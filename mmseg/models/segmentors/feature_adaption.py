import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
from collections import deque

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from ada.fft_domain_transfer import transform_img_source2target, ImgFetcher
from ..utils.dataset import TargetDataset, FeatDiscriminator, StructDiscriminator, ImageDomainTransformer
import yaml

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


@SEGMENTORS.register_module()
class FeatureAdaption(EncoderDecoder):
    """

    NOTE: Possible to train all objectives for both 'source' and 'target' models
    through the existing optimizer. Gradients will be correctly propagated to
    corresponding module.

    Ex: loss_task_source = loss_func()
        losses.update(loss_task_source)

        loss_task_target = loss_func()
        losses.update(loss_task_target)


    Example loss function:

        def loss_func(self, img, img_metas):
            # Backbone output
            out = self.encode_decode(img, img_metas)
            out = nn.Softmax2d()(out)
            # Generate label
            label = torch.zeros(out.shape).to('cuda:0')
            label[:,4] = 1.
            # Compute loss
            loss = - label * torch.log(out)
            loss = torch.mean(loss)
            # Create loss dictionary
            losses = {'decode.loss_test': loss}
            return losses

    losses_ = loss_func(img, img_metas)
    losses.update(losses_)

    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
    
        super(FeatureAdaption, self).__init__(backbone,
                                              decode_head,
                                              neck,
                                              auxiliary_head,
                                              train_cfg,
                                              test_cfg,
                                              pretrained)
        # Source model
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        # Target model (copy of source model)
        self.backbone_target = copy.deepcopy(self.backbone)
        self.decode_head_target = copy.deepcopy(self.decode_head)
        self.auxiliary_head_target = copy.deepcopy(self.auxiliary_head)
        if self.with_neck:
            self.neck_target = copy.deepcopy(self.neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        print("\n###################################")
        print("#  Initializing feature adaption")
        print("###################################\n")

        print("Read feature adaption parameters")
        with open("feat_adapt_params.yaml") as file:
            params = yaml.full_load(file)

        self.target_batch_size = int(params['target_batch_size'])
        self.discr_dropout_p = float(params['discr_dropout_p'])
        self.discr_acc_threshold = float(params['discr_acc_threshold'])
        self.lambda_seg = float(params['lambda_seg'])
        self.lambda_consis = float(params['lambda_consis'])
        self.lambda_discr = float(params['lambda_discr'])
        self.target_dataset = params['target_dataset']
        self.target_dataset_path = params['target_dataset_path']
        self.cropbox = params['cropbox']
        self.train_log_file = params['train_log_file']
        self.discr_input_dim = params['discr_input_dim']
        self.save_dir = params['save_dir']
        self.save_interval = params['save_interval']
        self.load_dir = params['load_dir']
        self.load_iter = params['load_iter']
        self.eval_model = params['eval_model']
        self.adaption_level = params['adaption_level']
        self.discr_type = params['discriminator']
        self.num_workers = int(params['num_workers'])
        print(f"target_batch_size:   {self.target_batch_size}")
        print(f"discr_dropout_p:     {self.discr_dropout_p}")
        print(f"discr_acc_threshold: {self.discr_acc_threshold}")
        print(f"lambda_seg:          {self.lambda_seg}")
        print(f"lambda_consis:       {self.lambda_consis}")
        print(f"lambda_discr:        {self.lambda_discr}")
        print(f"target_dataset:      {self.target_dataset}")
        print(f"target_dataset_path: {self.target_dataset_path}")
        print(f"cropbox:             {self.cropbox}")
        print(f"train_log_file:      {self.train_log_file}")
        print(f"discr_input_dim:     {self.discr_input_dim}")
        print(f"save_dir:            {self.save_dir}")
        print(f"save_interval:       {self.save_interval}")
        print(f"load_dir:            {self.load_dir}")
        print(f"load_iter:           {self.load_iter}")
        print(f"eval_model:          {self.eval_model}")
        print(f"adaption_level:      {self.adaption_level}")
        print(f"discriminator:       {self.discr_type}")
        print(f"num_workers:         {self.num_workers}\n")

        # Reset training log file
        with open(self.train_log_file, 'w') as file:
            file.write('# discr steps | model steps | discr acc\n')

        ############################
        #  DOMAIN TRANSFORMATIONS
        ############################
        #fft_beta = 2.35  # ???
        #self.domain_transformer = ImageDomainTransformer(
        #    self.target_dataset_path, fft_beta, self.cropbox)

        ##############
        #  DATASETS
        ##############
        self.dataset_target = TargetDataset(
            self.target_dataset_path, self.cropbox, self.target_dataset)
        self.dataloader_target = DataLoader(
            self.dataset_target, batch_size=self.target_batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.num_workers)
        # Dataloader iterators
        self.dataloader_target_iter = enumerate(self.dataloader_target)

        self.iter_idx = 0

        ####################
        #  DISCRIMINATORS
        ####################
        if self.discr_type == 'feature':
            self.discr = FeatDiscriminator(input_dim=self.discr_input_dim, output_dim=2, dropout_p=self.discr_dropout_p)
        elif self.discr_type == 'struct':
            self.discr = StructDiscriminator(input_dim=self.discr_input_dim, output_dim=1)
        else:
            raise Exception(f"Invalid discriminator type: {self.discr_type}")

        ################
        #  OBJECTIVES
        ################

        self.KLDivLoss = nn.KLDivLoss(reduction='mean')
        self.BCELoss = nn.BCEWithLogitsLoss()

        self.discr_acc_list = deque(maxlen=100)
        # So that generator is not optimized by chance
        for _ in range(100):
            self.discr_acc_list.append(0.)

        
    ############################
    #  SOURCE MODEL FUNCTIONS
    ############################

    def extract_feat_source(self, img):
        """
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def output_logits_source(self, out_feat, img_metas):
        """
        Args:
            out_feat (batch_n, C, H, W): Encoder output features.
            img_metas:
        
        Returns:
            Logit tensor (batch_n, C, H, W)
        """
        out_logit = self._decode_head_forward_test(out_feat, img_metas)
        return out_logit

    def _decode_head_forward_test_source(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train_source(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_source'))
        return losses

    def forward_train_source(self, x, img_metas, gt_semantic_seg, lambda_seg):
        """Forward function for training.

        Args:
            x (Tensor): Features outputted by encoder; extract_feat()
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            lambda_seg (float): Loss weight value.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        loss_decode = self._decode_head_forward_train_source(x, img_metas,
                                                      gt_semantic_seg)
        dict_key = 'decode_source.loss_seg'
        loss_decode[dict_key] = lambda_seg*loss_decode[dict_key]
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    ############################
    #  TARGET MODEL FUNCTIONS
    ############################

    def extract_feat_target(self, img):
        """
        """
        x = self.backbone_target(img)
        if self.with_neck:
            x = self.neck_target(x)
        return x

    def output_logits_target(self, out_feat, img_metas):
        """Returns output logit tensor of without resizing.

        Args:
            out_feat (batch_n, C, H, W): Encoder output features.
            img_metas:
        
        Returns:
            Logit tensor (batch_n, C, H, W)
        """
        out_logit = self._decode_head_forward_test_target(out_feat, img_metas)
        return out_logit

    def _decode_head_forward_test_target(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_target.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train_target(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head_target.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_target'))
        return losses
    
    def _auxiliary_head_forward_train_target(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head_target, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head_target):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}_target'))
        else:
            loss_aux = self.auxiliary_head_target.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux_target'))

        return losses
    
    def forward_train_target(self, x, img_metas, gt_semantic_seg, lambda_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            lambda_seg (float): Loss weight value.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        loss_decode = self._decode_head_forward_train_target(x, img_metas,
                                                            gt_semantic_seg)
        dict_key = 'decode_target.loss_seg'
        loss_decode[dict_key] = lambda_seg*loss_decode[dict_key]
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train_target(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
    
    #########################
    #  INFERENCE FUNCTIONS
    #########################

    def encode_decode_source(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_source(img)
        out = self._decode_head_forward_test_source(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def encode_decode_target(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_target(img)
        out = self._decode_head_forward_test_target(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def slide_inference(self, img, img_meta, rescale, model):
        raise NotImplementedError

    def whole_inference(self, img, img_meta, rescale, model):
        """Inference with full image."""

        if model == 'source':
            seg_logit = self.encode_decode_source(img, img_meta)
        elif model == 'target':
            seg_logit = self.encode_decode_target(img, img_meta)
        else:
            raise Exception(f'Undefined model type ({model})')

        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, model='source'):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, model)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, model)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output
    
    ###################
    #  TRAINING CODE
    ###################
        
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Loss term representation:
        - Dictionary with scalar tensors indexed by strings

        Ex: losses['decode.loss_seg'] --> tensor(2.9027, device='cuda:0)
            losses['decode.acc_seg': tensor([10.7810], device='cuda:0')

        How to add source model loss terms:
        1. Losses are represented by scalar tensors
        2. Loss tensors are stored in a loss dict w. key string containing 'loss'
        3. All loss tensors are summed at the end of each train step
        
        Ex: losses = dict()
            losses_ = loss_func()
            losses.update(losses_)
            loss, log_vars = _parse_losses(losses)

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.iter_idx += 1
        losses = dict()

        #optimization_state = self.state_machine.get_state()

        # Minibatch of normalized RGB image tensors with dim (N,C,H,W)
        img = data_batch['img']
        img_metas = data_batch['img_metas']
        gt_semantic_seg = data_batch['gt_semantic_seg']  # (N,1,H,W)

        _, img_target = self.dataloader_target_iter.__next__()
        img_target = img_target.to('cuda')

        #a = np.transpose(img[0].cpu().numpy(), (1,2,0)).astype(np.uint8)
        #b = np.transpose(img_target[0].cpu().numpy(), (1,2,0)).astype(np.uint8)
        #plt.subplot(1,2,1)
        #plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
        #plt.subplot(1,2,2)
        #plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        #plt.show()
        #exit()

        #img = self.domain_transformer.transform_img_batch(img)

        # Output syntax
        # out_TYPE_MODEL_INPUT

        ##################################
        #  1: Supervised label loss
        ##################################

        # Encoder features from 'source' domain
        out_feat_source_s = self.extract_feat_source(img)
        out_feat_target_s = self.extract_feat_target(img)

        # Source model
        loss = self.forward_train_source(
            out_feat_source_s, img_metas, gt_semantic_seg, self.lambda_seg)
        losses.update(loss)

        # Target model
        loss = self.forward_train_target(
            out_feat_target_s, img_metas, gt_semantic_seg, self.lambda_seg)
        losses.update(loss)

        ################################################################
        #  2. Target consistency loss
        #  Regularize target model by penalizing deviation from task.
        #  NOTE: Source model is NOT optimized (static distribution)
        ################################################################

        # Encoder features from 'target' domain
        with torch.no_grad():
            out_feat_source_t = self.extract_feat_source(img_target)
            out_logit_source = self.output_logits_source(out_feat_source_t, img_metas)
            out_prob_source = F.softmax(out_logit_source, dim=1)
            out_problog_source = F.log_softmax(out_logit_source, dim=1)

        out_feat_target_t = self.extract_feat_target(img_target)
        out_logit_target = self.output_logits_target(out_feat_target_t, img_metas)
        out_prob_target = F.softmax(out_logit_target, dim=1)
        out_problog_target = F.log_softmax(out_logit_target, dim=1)
        
        loss_cons = 0.5*(self.KLDivLoss(out_problog_source, out_prob_target)
                + self.KLDivLoss(out_problog_target, out_prob_source))
        loss_cons = self.lambda_consis * loss_cons
        losses_ = {'feature_adaptation.loss_consistency': loss_cons}
        losses.update(losses_)
        
        #############################
        #  3. Adapt model features
        #############################

        # Discriminator prediction
        discr_pred_target = self.discr(out_logit_target)

        # Dimensions for label
        N, _, d1, d2 = discr_pred_target.shape

        # Discriminator label
        # NOTE: Reverse labels to train model to fool discriminator
        source_label = torch.ones((N, 1, d1, d2), dtype=torch.float).to('cuda')

        loss_feat = self.BCELoss(discr_pred_target, source_label)
        loss_feat = self.lambda_discr * loss_feat
        losses_ = {'feature_adaptation.loss_feat': loss_feat}
        losses.update(losses_)

        ############################
        #  4. Train discriminator
        ############################

        discr_pred_source = self.discr(out_logit_source.detach())  # (2,1,3,9)
        discr_pred_target = self.discr(out_logit_target)  # (2,1,3,9)

        discr_pred = torch.cat((discr_pred_source, discr_pred_target))  # (4,1,3,9)

        # Dimensions for label
        N, _, d1, d2 = discr_pred_source.shape

        # Discriminator label
        source_label = torch.ones((N, 1, d1, d2), dtype=torch.float)
        target_label = torch.zeros((N, 1, d1, d2), dtype=torch.float)
        discr_label = torch.cat((source_label, target_label)).to('cuda')  # (4,1,3,9)

        loss_discr = self.BCELoss(discr_pred, discr_label)

        losses_ = {'feature_adaptation.loss_discr': loss_discr}
        losses.update(losses_)

        ############################
        #  Discriminator accuracy
        ############################

        discr_pred = discr_pred.detach().cpu().numpy()

        source_pred = np.zeros(discr_pred[0:N].shape)
        target_pred = np.zeros(discr_pred[N:].shape)
        # Only consider confident prediction
        source_pred[discr_pred[0:N] > 0.5] = 1.
        target_pred[discr_pred[N:] <= -0.5] = 1.
        
        correct_pred = 0.5*(np.mean(source_pred) + np.mean(target_pred))
        #self.discr_acc_list.append(correct_pred * 100.)

        losses_ = {'feature_adaptation.discr_acc': torch.tensor(correct_pred).to('cuda')}
        losses.update(losses_)

        #################
        #  Save models
        #################
        #if self.iter_idx % self.save_interval == 0:
        #    print('Saving model')
        #    self.save_model(self.iter_idx, self.save_dir, 'source')
        #    self.save_model(self.iter_idx, self.save_dir, 'target')
        #    self.save_model(self.iter_idx, self.save_dir, 'discr')

        # Reset iterators when cycled through
        if self.iter_idx % len(self.dataloader_target) == 0:
            self.dataloader_target_iter = enumerate(self.dataloader_target)

        # loss: Scalar tensor consisting of summed loss terms
        # - Loss value for source model back propagatino
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        # Returns source model loss for optimization + logging info
        return outputs
    
    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""

        #if self.load_iter != None:
        #    if self.eval_model == 'source':
        #        self.load_model(self.load_iter, self.load_dir, 'source')
        #    elif self.eval_model == 'target':
        #        self.load_model(self.load_iter, self.load_dir, 'target')
        #    else:
        #        raise Exception(f"Undefined model selected ({self.eval_model})")

        seg_logit = self.inference(img, img_meta, rescale, self.eval_model)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
'''    
    def save_model(self, iter_idx, path, tag):

        if tag == 'source':
            # Backbone
            file_path = os.path.join(path, f'backbone_source_{iter_idx}.pth') 
            torch.save(self.backbone.state_dict(), file_path)
            # Decode head
            file_path = os.path.join(path, f'decode_head_source_{iter_idx}.pth')
            torch.save(self.decode_head.state_dict(), file_path)
            #model_dict = {
            #    'backbone_source': self.backbone.state_dict(),
            #    'decode_head_source': self.decode_head.state_dict()
            #}
        elif tag == 'target':
            # Backbone
            file_path = os.path.join(path, f'backbone_target_{iter_idx}.pth') 
            torch.save(self.backbone_target.state_dict(), file_path)
            # Decode head
            file_path = os.path.join(path, f'decode_head_target_{iter_idx}.pth')
            torch.save(self.decode_head_target.state_dict(), file_path)
            #model_dict = {
            #    'backbone_target': self.backbone_target.state_dict(),
            #    'decode_head_target': self.decode_head_target.state_dict()
            #}
        elif tag == 'discr':
            file_path = os.path.join(path, f'discr_{iter_idx}.pth')
            torch.save(self.discr.state_dict(), file_path)
            #model_dict = {'discr': self.discr.state_dict()}
        else:
            raise Exception(f"Invalid model tag ({tag})")

        #file_path = os.path.join(path, f'feat_adapt_iter_{tag}_{iter_idx}.pkl')
        #with open(file_path, 'wb') as file:
        #    pickle.dump(model_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, iter_idx, path, tag):

        #file_path = os.path.join(path, f'feat_adapt_iter_{tag}_{iter_idx}.pkl')
        #with open(file_path, 'rb') as file:
        #    model_dict = pickle.load(file)

        model = torch.load('exp3_a_c/iter_80000.pth')['state_dict']

        self.backbone.load_state_dict(model, strict=False)

        print("Exiting")
        exit()

        if tag == 'source':
            # Backbone
            file_path = os.path.join(path, f'backbone_source_{iter_idx}.pth')
            state_dict = torch.load(file_path)
            self.backbone.load_state_dict(state_dict)
            # Decode head
            file_path = os.path.join(path, f'decode_head_source_{iter_idx}.pth')
            state_dict = torch.load(file_path)
            self.decode_head.load_state_dict(state_dict)
            #self.backbone.load_state_dict(model_dict['backbone_source'])
            #self.decode_head.load_state_dict(model_dict['decode_head_source'])
        elif tag == 'target':
            file_path = os.path.join(path, f'backbone_target_{iter_idx}.pth')
            state_dict = torch.load(file_path)
            self.backbone_target.load_state_dict(state_dict)
            # Decode head
            file_path = os.path.join(path, f'decode_head_target_{iter_idx}.pth')
            state_dict = torch.load(file_path)
            self.decode_head_target.load_state_dict(state_dict)
            #self.backbone_target.load_state_dict(model_dict['backbone_target'])
            #self.decode_head_target.load_state_dict(model_dict['decode_head_target'])
        elif tag =='discr':
            file_path = os.path.join(path, f'discr_{iter_idx}.pth')
            state_dict = torch.load(file_path)
            self.discr.load_state_dict(state_dict)
            #self.discr.load_state_dict(model_dict['discr'])
        else:
            raise Exception(f"Invalid model tag ({tag})")
'''
        