import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import mmcv
import os
import copy
from collections import deque
import yaml


class FeatureAdaptionDataset(Dataset):

    def __init__(self, root_dir, crop):
        '''
        Args:
            label: Integer (source, target) = (1, 0)
        '''

        if os.path.isdir(root_dir):
            self.root_dir = root_dir
        else:
            raise FileNotFoundError(f"Dataset directories do not exists")

        self.samples_N = len(os.listdir(self.root_dir))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[123.675/255., 116.28/255., 103.53/255.], std=[58.395/255., 57.12/255., 57.375/255.]
            )
            
        ])

    def __len__(self):
        return self.samples_N

    def __getitem__(self, idx):
        
        filepath = os.path.join(self.root_dir, f"{idx}.png")
        img = mmcv.imread(filepath)

        img = self.transform(img)
        img = img.to('cuda')

        return img


class FeatDiscriminator(torch.nn.Module):
    '''Discriminating independent feature vectors using 1x1 convolutions.
    '''
    def __init__(self, input_dim=512, output_dim=2, dropout_p=0.5):
        '''
        '''
        super().__init__()

        dim1 = 512
        dim2 = int(dim1/2)

        self.D = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 1),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, 1),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim2, output_dim, 1)
        )

    def forward(self, x):
        out = self.D(x)
        return out


class StructDiscriminator(torch.nn.Module):
    '''Discriminating contextual features using 2D convolutions.
    '''
    def __init__(self, input_dim, output_dim=2, ch=64, dropout_p=0.):
        '''
        '''
        super().__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_dim, ch, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(p=dropout_p),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(p=dropout_p),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(p=dropout_p),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(p=dropout_p),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ch*8, output_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.D(x)
        return out


@SEGMENTORS.register_module()
class FeatureAdaption(EncoderDecoder):
    """Encoder Decoder segmentors.

    Each MMSegmentation 

    Two backbones
        self.backbone: Gets adapted so target features --> source features
        self.backbone_frozen: Static distribution of source features
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FeatureAdaption, self).__init__(backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, pretrained)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

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
        self.cropbox = params['cropbox']  # (512, 1024)
        self.dataset_path_source = params['dataset_path_source']
        self.dataset_path_target = params['dataset_path_target']
        self.train_log_file = params['train_log_file']
        self.discr_input_dim = params['discr_input_dim']
        self.save_interval = params['save_interval']
        self.adaption_level = params['adaption_level']
        self.discr_type = params['discriminator']
        print(f"discr_lr:            {self.discr_lr}")
        print(f"model_lr:            {self.model_lr}")
        print(f"momentum:            {self.sgd_momentum}")
        print(f"batch_size:          {self.batch_size}")
        print(f"discr_dropout_p:     {self.discr_dropout_p}")
        print(f"cropbox:             {self.cropbox}")
        print(f"dataset_path_source: {self.dataset_path_source}")
        print(f"dataset_path_target: {self.dataset_path_target}")
        print(f"train_log_file:      {self.train_log_file}")
        print(f"discr_input_dim:     {self.discr_input_dim}")
        print(f"save_interval:       {self.save_interval}")
        print(f"adaption_level:      {self.adaption_level}")
        print(f"discriminator:       {self.discr_type}\n")

        # Reset training log file
        with open(self.train_log_file, 'w') as file:
            file.write('# discr steps | model steps | discr acc\n')

        ##############
        #  DATASETS
        ##############
        # Datasets
        self.dataset_source = FeatureAdaptionDataset(self.dataset_path_source, self.cropbox)
        self.dataset_target = FeatureAdaptionDataset(self.dataset_path_target, self.cropbox)
        # Dataloaders
        self.dataloader_source = DataLoader(self.dataset_source, batch_size=self.batch_size)
        self.dataloader_target = DataLoader(self.dataset_target, batch_size=self.batch_size)

        print(f"Using {len(self.dataloader_source.dataset)} source images")
        print(f"Using {len(self.dataloader_target.dataset)} target images\n")

        ############
        #  MODELS
        ############

        # Must be initialized AFTER backbone weights set
        self.backbone_frozen = None  
        self.decode_head_frozen = None

        if self.discr_type == 'feature':
            self.discr = FeatDiscriminator(input_dim=self.discr_input_dim, output_dim=2, dropout_p=self.discr_dropout_p)
        elif self.discr_type == 'struct':
            self.discr = StructDiscriminator(input_dim=self.discr_input_dim, output_dim=2, dropout_p=self.discr_dropout_p)
        else:
            raise Exception(f"Invalid discriminator type: {self.discr_type}")
        self.discr = self.discr.to('cuda')

        self.initialize_backbone_frozen = True
        if self.adaption_level == 'output':
            self.initialize_decode_head_frozen = True
        else:
            self.initialize_decode_head_frozen = False

        ################
        #  OPTIMIZERS
        ################

        # Optimizer for 'backbone' parameters
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer_backbone = torch.optim.SGD(params, lr=self.model_lr, weight_decay=0.0005, momentum=self.sgd_momentum)

        # Optimizer for 'decoder' parameters
        params = [p for p in self.decode_head.parameters() if p.requires_grad]
        self.optimizer_decoder = torch.optim.SGD(params, lr=self.model_lr, weight_decay=0.0005, momentum=self.sgd_momentum)

        # Optimizer for 'discriminator' parameters
        params = [p for p in self.discr.parameters() if p.requires_grad]
        self.optimizer_discr = torch.optim.SGD(params, lr=self.discr_lr, weight_decay=0.0005, momentum=self.sgd_momentum)

        ##########
        #  LOSS
        ##########

        self.NLLLoss = torch.nn.NLLLoss(size_average=True, ignore_index=255)

        ################
        #  PARAMETERS
        ################

        # Loss weights
        self.lambda_discr = 1.
        self.lambda_gen = 0.1

        self.discr_acc_threshold = 60

        # Optimization variables
        self.iter_idx = 0
        self.gen_steps = 0
        self.gen_steps_tot = 0
        self.loss_disc_list = deque(maxlen=100)
        self.loss_gen_list = deque(maxlen=100)
        self.discr_acc = deque(maxlen=100)

        # So that generator is not optimized by chance
        for _ in range(100):
            self.discr_acc.append(0.)

        self.iter_save_interval = 2000

    def extract_feat_frozen(self, img):
        """Extract features from images."""
        x = self.backbone_frozen(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def _decode_head_forward_tes_frozen(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_frozen.forward_test(x, img_metas, self.test_cfg)
        return seg_logits
    
    def encode_decode_frozen(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map."""
        x = self.extract_feat_frozen(img)
        out = self._decode_head_forward_tes_frozen(x, img_metas)
        return out

    def write_log_entry(self, line):
        with open(self.train_log_file, 'a') as file:
            file.write(line)


    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

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
                DDP, it means the batch si ze on each GPU), which is used for
                averaging the logs.
        """

        img_metas = data_batch['img_metas'][0]

        # Copy backbone once after checkpoint loaded
        if self.initialize_backbone_frozen:
            self.backbone_frozen = copy.deepcopy(self.backbone)
            self.initialize_backbone_frozen = False

            if self.initialize_decode_head_frozen:
                self.decode_head_frozen = copy.deepcopy(self.decode_head)
                self.initialize_decode_head_frozen = False

        # Run loop until model save point reached
        do_not_save = True
        while do_not_save:

            for source_imgs, target_imgs in zip(self.dataloader_source, self.dataloader_target):

                self.iter_idx += 1

                if self.iter_idx % 100 == 0:
                    print(f"Iter {self.iter_idx} | Discr. L {np.mean(self.loss_disc_list):.6f} (Acc. {np.mean(self.discr_acc):.2f} %) | Gen. L {np.mean(self.loss_gen_list):.6f} (steps {self.gen_steps}/{self.gen_steps_tot})")
                    self.write_log_entry(f"{self.iter_idx}, {self.gen_steps_tot}, {np.mean(self.discr_acc):.2f}\n")

                    self.gen_steps = 0

                ############################
                #  OPTIMIZE DISCRIMINATOR
                ############################
                for params in self.discr.parameters():
                    params.requires_grad = True

                self.optimizer_discr.zero_grad()
                self.optimizer_backbone.zero_grad()

                # Generate model features
                if self.adaption_level == 'backbone':
                    with torch.no_grad():
                        source_x = self.extract_feat_frozen(source_imgs)[-1]  # (2,512,64,128)
                        target_x = self.extract_feat(target_imgs)[-1]
                elif self.adaption_level == 'output':
                    with torch.no_grad():
                        source_x = self.encode_decode_frozen(source_imgs, img_metas)  # (2,19,128,256)
                        target_x = self.encode_decode(target_imgs, img_metas)
                else:
                    raise ValueError(f"Given feature adaption level not supported ({self.adaption_level})")

                # Discriminator prediction
                source_pred = self.discr(source_x)  # (2,2,64,128)
                target_pred = self.discr(target_x)

                pred_discr = torch.cat((source_pred, target_pred))

                # Dimensions for label
                N, _, d1, d2 = source_pred.shape

                # Discriminator label
                source_label = torch.ones((N, d1, d2), dtype=torch.long)
                target_label = torch.zeros((N, d1, d2), dtype=torch.long)

                label = torch.cat((source_label, target_label)).to('cuda')  # (4, 64, 128)

                # Compute loss
                loss_discr = self.NLLLoss(F.log_softmax(pred_discr, dim=1), label)

                loss_discr = self.lambda_discr * loss_discr

                loss_discr.backward()

                # Compute discriminator accuracy
                pred_dis = torch.squeeze(pred_discr.max(1)[1])
                dom_acc = (pred_dis == label).float().mean().item() 
                self.discr_acc.append(dom_acc * 100.)

                self.loss_disc_list.append(loss_discr.item())

                ########################
                #  OPTIMIZE GENERATOR
                ########################

                # Only train generator if discriminator is accurate
                if np.mean(self.discr_acc) > self.discr_acc_threshold:

                    for params in self.discr.parameters():
                        params.requires_grad = False

                    self.optimizer_discr.zero_grad()
                    self.optimizer_backbone.zero_grad()

                    # Generate model features
                    if self.adaption_level == 'backbone':
                        target_x = self.extract_feat(target_imgs)[-1]
                    elif self.adaption_level == 'output':
                        target_x = self.encode_decode(target_imgs, img_metas)
                    else:
                        raise ValueError(f"Given feature adaption level not supported ({self.adaption_level})")

                    # Discriminator prediction
                    pred_discr = self.discr(target_x)

                    # Discriminator label
                    label = torch.ones((N, d1, d2), dtype=torch.long).to('cuda')

                    loss_gen = self.NLLLoss(F.log_softmax(pred_discr, dim=1), label)

                    loss_gen = self.lambda_gen * loss_gen

                    loss_gen.backward()

                    self.loss_gen_list.append(loss_gen.item())
                    self.gen_steps += 1
                    self.gen_steps_tot += 1
                
                #######################
                #  OPTIMIZATION STEP
                #######################
                self.optimizer_discr.step()
                self.optimizer_backbone.step()
                self.optimizer_decoder.step()
                
                if self.iter_idx % self.iter_save_interval == 0:
                    do_not_save = False
                    break

        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        # Zero loss to not interfere with feature adaption
        loss = torch.tensor(0., requires_grad=True)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        return outputs
