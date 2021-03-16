import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from ada.fft_domain_transfer import transform_img_source2target, ImgFetcher
from ..utils.dataset import FeatureAdaptionDatasetCityscapes, FeatDiscriminator, StructDiscriminator, ImageDomainTransformer
import yaml

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

        self.discr_lr = float(params['discr_lr'])
        self.model_lr = float(params['model_lr'])
        self.sgd_momentum = float(params['sgd_momentum'])
        self.batch_size = int(params['batch_size'])
        self.discr_dropout_p = float(params['discr_dropout_p'])
        self.discr_acc_threshold = float(params['discr_acc_threshold'])
        self.lambda_seg = float(params['lambda_seg'])
        self.lambda_consis = float(params['lambda_consis'])
        self.lambda_discr = float(params['lambda_discr'])
        self.cropbox = params['cropbox']  # (512, 1024)
        self.source_subset = params['source_subset']
        self.dataset_path_source = params['dataset_path_source']
        self.dataset_path_target = params['dataset_path_target']
        self.train_log_file = params['train_log_file']
        self.discr_input_dim = params['discr_input_dim']
        self.save_dir = params['save_dir']
        self.save_interval = params['save_interval']
        self.adaption_level = params['adaption_level']
        self.discr_type = params['discriminator']
        self.print_interval = int(params['print_interval'])
        self.num_workers = int(params['num_workers'])
        print(f"discr_lr:            {self.discr_lr}")
        print(f"model_lr:            {self.model_lr}")
        print(f"momentum:            {self.sgd_momentum}")
        print(f"batch_size:          {self.batch_size}")
        print(f"discr_dropout_p:     {self.discr_dropout_p}")
        print(f"discr_acc_threshold: {self.discr_acc_threshold}")
        print(f"lambda_seg:          {self.lambda_seg}")
        print(f"lambda_consis:       {self.lambda_consis}")
        print(f"lambda_discr:        {self.lambda_discr}")
        print(f"cropbox:             {self.cropbox}")
        print(f"source_subset:       {self.source_subset}")
        print(f"dataset_path_source: {self.dataset_path_source}")
        print(f"dataset_path_target: {self.dataset_path_target}")
        print(f"train_log_file:      {self.train_log_file}")
        print(f"discr_input_dim:     {self.discr_input_dim}")
        print(f"save_dir:            {self.save_dir}")
        print(f"save_interval:       {self.save_interval}")
        print(f"adaption_level:      {self.adaption_level}")
        print(f"discriminator:       {self.discr_type}")
        print(f"print_interval:      {self.print_interval}")
        print(f"num_workers:         {self.num_workers}\n")

        # Reset training log file
        with open(self.train_log_file, 'w') as file:
            file.write('# discr steps | model steps | discr acc\n')

        ############################
        #  DOMAIN TRANSFORMATIONS
        ############################
        fft_beta = 2.35  # ???
        self.domain_transformer = ImageDomainTransformer(
            self.dataset_path_target, fft_beta, self.cropbox)

        ##############
        #  DATASETS
        ##############
        self.dataset_target = FeatureAdaptionDatasetCityscapes(
            self.dataset_path_target, self.cropbox) 
            #target_adaption_path=self.dataset_path_source)
        self.dataloader_target = DataLoader(
            self.dataset_target, batch_size=self.batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.num_workers)
        # Dataloader iterators
        self.dataloader_target_iter = enumerate(self.dataloader_target)

        ####################
        #  DISCRIMINATORS
        ####################
        if self.discr_type == 'feature':
            self.discr = FeatDiscriminator(input_dim=self.discr_input_dim, output_dim=2, dropout_p=self.discr_dropout_p)
        elif self.discr_type == 'struct':
            self.discr = StructDiscriminator(input_dim=self.discr_input_dim, output_dim=1, dropout_p=self.discr_dropout_p)
        else:
            raise Exception(f"Invalid discriminator type: {self.discr_type}")

        ################
        #  OPTIMIZERS
        ################
        params = [p for p in self.discr.parameters() if p.requires_grad]
        self.optimizer_discr = torch.optim.Adam(params, lr=self.discr_lr, weight_decay=0.0005, betas=(0.9, 0.99)) #momentum=self.sgd_momentum)

        self.KLDivLoss = nn.KLDivLoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

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

    def encode_decode_source(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map.
            Args:
                img: 
                img_metas:
        
            Returns:
                Logit tensor (batch_n, C, H, W)
        """
        x = self.extract_feat_source(img)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def _decode_head_forward_train_source(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_source'))
        return losses

    def forward_train_source(self, x, img_metas, gt_semantic_seg):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
    
    #def model_forward(self, img, img_metas):
    #    """
    #    """
    #    out = self.encode_decode(img, img_metas)
    #    out = nn.Softmax2d()(out)
    #    return out

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

    def _decode_head_forward_test_target(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_target.forward_test(x, img_metas, self.test_cfg)
        return seg_logits
    
    def encode_decode_target(self, img, img_metas):
        """
        """
        x = self.extract_feat_target(img)
        out = self._decode_head_forward_test_target(x, img_metas)
        return out
    
    #def model_forward_target(self, img, img_metas):
    #    """
    #    """
    #    out = self.encode_decode_target(img, img_metas)
    #    out = nn.Softmax2d()(out)
    #    return out

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
    
    def forward_train_target(self, x, img_metas, gt_semantic_seg):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        loss_decode = self._decode_head_forward_train_target(x, img_metas,
                                                            gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train_target(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
    
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
        # Minibatch of normalized RGB image tensors with dim (N,C,H,W)
        img = data_batch['img']
        img_metas = data_batch['img_metas']
        gt_semantic_seg = data_batch['gt_semantic_seg']  # (N,1,H,W)

        _, img_target = self.dataloader_target_iter.__next__()
        img_target = img_target.to('cuda')

        #a = np.transpose(img[0].cpu().numpy(), (1,2,0)).astype(np.uint8)
        #b = np.transpose(img_target[0].cpu().numpy(), (1,2,0)).astype(np.uint8)
        #plt.subplot(1,2,1)
        #plt.imshow(a)
        #plt.subplot(1,2,2)
        #plt.imshow(b)
        #plt.show()
        #exit()

        #img = self.domain_transformer.transform_img_batch(img)

        # Encoder features

        out_feat_source = self.extract_feat_source(img)
        out_feat_target = self.extract_feat_target(img)

        losses = dict()

        ##################################
        #  1: Supervised label loss
        ##################################

        # Source model
        loss = self.forward_train_source(out_feat_source, img_metas, gt_semantic_seg)
        losses.update(loss)

        # Target model
        loss = self.forward_train_target(out_feat_target, img_metas, gt_semantic_seg)
        losses.update(loss)

        '''
        # Target model
        # img <-- source2target(img)
        losses_ = self.forward_train_target(img, img_metas, gt_semantic_seg)
        losses_['decode_target.loss_seg'] = self.lambda_seg * losses_['decode_target.loss_seg']
        #losses.update(losses_)

        ################################
        #  2. Target consistency loss
        ################################

        # Source model
        out_source = self.encode_decode(img_target, img_metas)
        out_source = resize(input=out_source, size=img_target.shape[2:], 
            mode='bilinear', align_corners=self.align_corners)
        out_source_prob = F.softmax(out_source, dim=1)
        out_source_problog = F.log_softmax(out_source, dim=1)

        # Target model
        out_target = self.encode_decode_target(img_target, img_metas)
        out_target = resize(input=out_target, size=img_target.shape[2:], 
            mode='bilinear', align_corners=self.align_corners)
        out_target_prob = F.softmax(out_target, dim=1)
        out_target_problog = F.log_softmax(out_target, dim=1)

        loss = (self.KLDivLoss(out_source_problog, out_target_prob)
                + self.KLDivLoss(out_target_problog, out_source_prob))
        
        loss = self.lambda_consis * loss
        losses_ = {'feature_adaption.loss_consistency': loss}
        #losses.update(losses_)

        #out_source_pred = out_source_prob.argmax(dim=1)
        
        #out_target = self.model_forward_target(img_target, img_metas)

        #print(out_source_pred.shape)
        #print(out_target.shape)

        #exit()
        '''
        

        #############################
        #  3. Adapt model features
        #############################

        ############################
        #  4. Train discriminator
        ############################

        #######################
        #  OPTIMIZATION STEP
        #######################
        #self.optimizer_discr.step()

        # loss: Scalar tensor consisting of summed loss terms
        # - Loss value for source model back propagatino
        loss, log_vars = self._parse_losses(losses)

        print(loss.item())
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        # Returns source model loss for optimization + logging info
        return outputs
    