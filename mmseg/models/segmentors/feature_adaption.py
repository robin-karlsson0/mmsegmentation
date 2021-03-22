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


class SequentialStateMachine():
    """Finite state machine handling which optimization to perform every step.

    States are represented as integeres [0, 1, ..., N-1]
    Tranitions occur sequentially: state n --> state n+1
    """
    def __init__(self, state_count:int, initial_state:int=0):
        self.state_count = state_count
        self.state = initial_state
    
    def get_state(self):
        """Returns current state and transition to new state.
        """
        current_state = self.state
        self._update_state()
        return current_state
    
    def _update_state(self):
        increment_state = self.state + 1
        if increment_state >= self.state_count:
            self.state = 0
        else:
            self.state = increment_state


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

        self.iter_idx = 0

        ####################
        #  DISCRIMINATORS
        ####################
        #if self.discr_type == 'feature':
        #    self.discr = FeatDiscriminator(input_dim=self.discr_input_dim, output_dim=2, dropout_p=self.discr_dropout_p)
        #elif self.discr_type == 'struct':
        #    self.discr = StructDiscriminator(input_dim=self.discr_input_dim, output_dim=1, dropout_p=self.discr_dropout_p)
        #else:
        #    raise Exception(f"Invalid discriminator type: {self.discr_type}")

        ################
        #  OPTIMIZERS
        ################
        #params = [p for p in self.discr.parameters() if p.requires_grad]
        #self.optimizer_discr = torch.optim.Adam(params, lr=self.discr_lr, weight_decay=0.0005, betas=(0.9, 0.99)) #momentum=self.sgd_momentum)

        self.KLDivLoss = nn.KLDivLoss(reduction='mean')
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        optimization_stages = 2
        self.state_machine = SequentialStateMachine(optimization_stages)

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

    #def encode_decode_source(self, img, img_metas):
    #    """Encode images with backbone and decode into a semantic segmentation
    #    map.
    #        Args:
    #            img: 
    #            img_metas:
    #    
    #        Returns:
    #            Logit tensor (batch_n, C, H, W)
    #    """
    #    x = self.extract_feat_source(img)
    #    out = self._decode_head_forward_test(x, img_metas)
    #    return out

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
    
    #def encode_decode_target(self, img, img_metas):
    #    """
    #    """
    #    x = self.extract_feat_target(img)
    #    out = self._decode_head_forward_test_target(x, img_metas)
    #    return out
    
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

        optimization_state = self.state_machine.get_state()

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

        # Output syntax
        # out_TYPE_MODEL_INPUT

        ##################################
        #  1: Supervised label loss
        ##################################

        if optimization_state == 0:

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

        elif optimization_state == 1:

            # Encoder features from 'target' domain
            with torch.no_grad():
                out_feat_source_t = self.extract_feat_source(img_target)
            out_feat_target_t = self.extract_feat_target(img_target)

            with torch.no_grad():
                out_logit_source = self.output_logits_source(out_feat_source_t, img_metas)
                out_prob_source = F.softmax(out_logit_source, dim=1)
                out_problog_source = F.log_softmax(out_logit_source, dim=1)

            out_logit_target = self.output_logits_target(out_feat_target_t, img_metas)
            out_prob_target = F.softmax(out_logit_target, dim=1)
            out_problog_target = F.log_softmax(out_logit_target, dim=1)
            
            loss_cons = 0.5*(self.KLDivLoss(out_problog_source, out_prob_target)
                    + self.KLDivLoss(out_problog_target, out_prob_source))
            loss_cons = self.lambda_consis * loss_cons
            losses_ = {'feature_adaption.loss_consistency': loss_cons}
            losses.update(losses_)
        
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

        if self.iter_idx % len(self.dataloader_target) == 0:
                self.dataloader_target_iter = enumerate(self.dataloader_target)

        # loss: Scalar tensor consisting of summed loss terms
        # - Loss value for source model back propagatino
        loss, log_vars = self._parse_losses(losses)

        #print(loss.item())
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        # Returns source model loss for optimization + logging info
        return outputs
    