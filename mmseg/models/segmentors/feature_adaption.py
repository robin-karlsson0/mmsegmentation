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


class Discriminator(torch.nn.Module):
    '''
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

        ##############
        #  DATASETS
        ##############

        # Datasets
        self.cropbox = (512, 1024)
        self.dataset_path_source = '/media/robin/Data/feat_adapt_dataset/cityscapes'  #'/var/datasets/feat_adapt_dataset/a2d2'
        self.dataset_path_target = '/media/robin/Data/feat_adapt_dataset/a2d2'  #'/var/datasets/feat_adapt_dataset/cityscapes'
        self.dataset_source = FeatureAdaptionDataset(self.dataset_path_source, self.cropbox)
        self.dataset_target = FeatureAdaptionDataset(self.dataset_path_target, self.cropbox)
        # Dataloaders
        self.dataloader_source = DataLoader(self.dataset_source, batch_size=2)
        self.dataloader_target = DataLoader(self.dataset_target, batch_size=2)

        print(f"Using {len(self.dataloader_source.dataset)} source images")
        print(f"Using {len(self.dataloader_target.dataset)} target images\n")

        ############
        #  MODELS
        ############

        self.backbone_frozen = None  # Must be initialized AFTER backbone weights set

        self.discr = Discriminator(input_dim=512, output_dim=2, dropout_p=0.5)
        self.discr = self.discr.to('cuda')

        self.initialize_backbone_frozen = True

        ################
        #  OPTIMIZERS
        ################

        # Optimizer for 'backbone_adapt' parameters
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer_backbone = torch.optim.SGD(params, lr=1e-3, weight_decay=0.0005)#, momentum=0.9)

        # Optimizer for 'discriminator' parameters
        params = [p for p in self.discr.parameters() if p.requires_grad]
        self.optimizer_discr = torch.optim.SGD(params, lr=1e-3, weight_decay=0.0005)#, momentum=0.9)

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

        self.iter_save_interval = 2000

    def extract_feat_frozen(self, img):
        """Extract features from images."""
        x = self.backbone_frozen(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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

        # Copy backbone once after checkpoint loaded
        if self.initialize_backbone_frozen:
            self.backbone_frozen = copy.deepcopy(self.backbone)
            self.initialize_backbone_frozen = False

        # Run loop until model save point reached
        do_not_save = True
        while do_not_save:

            for source_imgs, target_imgs in zip(self.dataloader_source, self.dataloader_target):

                self.iter_idx += 1

                if self.iter_idx % 1 == 0:
                    print(f"Iter {self.iter_idx} | Discr. L {np.mean(self.loss_disc_list):.6f} (Acc. {np.mean(self.discr_acc):.2f} %) | Gen. L {np.mean(self.loss_gen_list):.6f} (steps {self.gen_steps}/{self.gen_steps_tot})")
                    self.gen_steps = 0

                ############################
                #  OPTIMIZE DISCRIMINATOR
                ############################

                self.optimizer_discr.zero_grad()

                # Generate model features
                source_x = self.extract_feat_frozen(source_imgs)[-1]  # (2,512,64,128)
                target_x = self.extract_feat(target_imgs)[-1]

                # Discriminator prediction
                source_pred = self.discr(source_x)  # (2,2,64,128)
                target_pred = self.discr(target_x)

                pred_discr = torch.cat((source_pred, target_pred))

                # Dimensions for label
                N, _, d1, d2 = source_pred.shape

                # Discriminator label
                source_label = torch.ones((N, d1, d2), dtype=torch.long)
                target_label = torch.zeros((N, d1, d2), dtype=torch.long)

                label = torch.cat((source_label, target_label)).to('cuda')  # (4, 128, 128)

                # Compute loss
                loss_discr = self.NLLLoss(F.log_softmax(pred_discr, dim=1), label)

                loss_discr = self.lambda_discr * loss_discr

                loss_discr.backward()

                self.optimizer_discr.step()

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

                    self.optimizer_discr.zero_grad()
                    self.optimizer_backbone.zero_grad()

                    # Generate model features
                    target_x = self.extract_feat(target_imgs)[-1]

                    # Discriminator prediction
                    pred_discr = self.discr(target_x)

                    # Discriminator label
                    label = torch.ones((N, d1, d2), dtype=torch.long).to('cuda')

                    loss_gen = self.NLLLoss(F.log_softmax(pred_discr, dim=1), label)

                    loss_gen = self.lambda_gen * loss_gen

                    loss_gen.backward()

                    self.optimizer_backbone.step()

                    self.loss_gen_list.append(loss_gen.item())
                    self.gen_steps += 1
                    self.gen_steps_tot += 1
                
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
