import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from ada.fft_domain_transfer import transform_img_source2target, ImgFetcher

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import mmcv
import os
import copy
from collections import deque
import yaml
import pickle
import random

class SemanticSegDataset(Dataset):

    def __init__(self, root_dir):
        '''
        '''

        if os.path.isdir(root_dir):
            self.root_dir = root_dir
        else:
            raise FileNotFoundError(f"Dataset directories do not exists \
                                     ({root_dir})")
        
        self.samples_N = 0
    
    def __len__(self):
        return self.samples_N

    def _random_crop(self, img, random_ratio_1, random_ratio_2):
        '''
        '''
        dim = len(img.shape)
        if dim == 3:
            crop_start_h = int(np.round((img.shape[1] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[2] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[:, h0:h1, w0:w1]
        elif dim == 2:
            crop_start_h = int(np.round((img.shape[0] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[1] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[h0:h1, w0:w1]
        else:
            raise ValueError(f"Unsupported dimensino for image crop ({img.shape})")

        return img


class SourceDatasetA2D2(SemanticSegDataset):
    '''
    dataloader_iter = enumerate(dataloader)
    
    _, batch = dataloader_iter.__next__()

    batch --> [imgs_tensor, labels_tensor]
        imgs_tensor   --> (N, C, H, W)
        labels_tensor --> (N, H, W)
    '''

    def __init__(self, root_dir, subset, crop, target_adaption_path=None, label_postfix='_labelTrainIds'):
        '''
        Args:
            subset: String specifying subset to load.
                    Options: 'train', 'val', 'test'
        '''
        super().__init__(root_dir)

        if subset not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid dataset 'subset' ({subset})")

        self.img_dir = os.path.join(root_dir, 'img_dir', subset)
        self.ann_dir = os.path.join(root_dir, 'ann_dir', subset)

        if ((os.path.isdir(self.img_dir) is False) or
            (os.path.isdir(self.ann_dir) is False)):
           raise FileNotFoundError(f"Invalid dataset paths ({self.img_dir}, \
                                     {self.ann_dir})")

        # Create list of image filenames
        self.img_list = os.listdir(self.img_dir)

        self.samples_N = len(self.img_list)

        self.label_postfix = label_postfix

        self.crop_h = crop[0]
        self.crop_w = crop[1]

        # Image adaption
        if target_adaption_path is not None:
            self.target_img_fetcher = ImgFetcher(target_adaption_path)
            self.fft_beta = 0.002  # A2D2 --> Cityscapes
        else:
            self.target_img_fetcher = None

        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        ])

    def __getitem__(self, idx):

        img_filename = self.img_list[idx]
        ann_filename = img_filename.replace('.png', self.label_postfix + '.png')

        img = mmcv.imread(os.path.join(self.img_dir, img_filename))
        ann = mmcv.imread(os.path.join(os.path.join(self.ann_dir, ann_filename)), flag='grayscale')

        # BGR --> RGB
        img = mmcv.bgr2rgb(img)

        # (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))

        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        img = self._random_crop(img, random_ratio_1, random_ratio_2)
        ann = self._random_crop(ann, random_ratio_1, random_ratio_2)

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=2)
            ann = np.flip(ann, axis=1)

        # Image adaption
        if self.target_img_fetcher is not None:
            trg_img_path = self.target_img_fetcher.get_img_path()
            trg_img = mmcv.imread(trg_img_path)
            trg_img = mmcv.bgr2rgb(trg_img)
            # (H,W,C) --> (C,H,W)
            trg_img = np.transpose(trg_img, (2,0,1))
            # Random crop
            trg_img = self._random_crop(trg_img, random_ratio_1, random_ratio_2)
            # (C,H,W) --> (H,W,C)
            img = img.transpose((1,2,0))
            trg_img = trg_img.transpose((1,2,0))
            img = transform_img_source2target(img, trg_img, beta=self.fft_beta)
            # PIL --> Numpy array
            img = np.array(img)
            # (H,W,C) --> (C,H,W)
            img = np.transpose(img, (2,0,1))

        img = torch.from_numpy(np.ascontiguousarray(img ,dtype=np.float32))
        ann = torch.from_numpy(np.ascontiguousarray(ann ,dtype=np.long))
        img = self.normalize(img)

        return img, ann


class FeatureAdaptionDatasetCityscapes(SemanticSegDataset):
    '''
    dataloader_iter = enumerate(dataloader)
    
    _, batch = dataloader_iter.__next__()

    batch --> [img_1, ..., img_N] for batch size N
    '''

    def __init__(self, root_dir, crop, target_adaption_path=None):
        '''
        Args:
            label: Integer (source, target) = (1, 0)
        '''
        super().__init__(root_dir)

        self.samples_N = len(os.listdir(self.root_dir))

        self.crop_h = crop[0]
        self.crop_w = crop[1]

        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        ])

        # Image adaption
        if target_adaption_path is not None:
            trg_img_dir = os.path.join(target_adaption_path, 'img_dir', 'train')
            self.target_img_fetcher = ImgFetcher(trg_img_dir)
            self.fft_beta = 0.0015  # Cityscapes --> A2D2
        else:
            self.target_img_fetcher = None

    def __getitem__(self, idx):
        
        filepath = os.path.join(self.root_dir, f"{idx}.png")
        img = mmcv.imread(filepath)

        # BGR --> RGB
        img = mmcv.bgr2rgb(img)

        # (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))

        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        img = self._random_crop(img, random_ratio_1, random_ratio_2)

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=2)

         # Image adaption
        if self.target_img_fetcher is not None:
            trg_img_path = self.target_img_fetcher.get_img_path()
            trg_img = mmcv.imread(trg_img_path)
            trg_img = mmcv.bgr2rgb(trg_img)
            # (H,W,C) --> (C,H,W)
            trg_img = np.transpose(trg_img, (2,0,1))
            # Random crop
            trg_img = self._random_crop(trg_img, random_ratio_1, random_ratio_2)
            # (C,H,W) --> (H,W,C)
            img = img.transpose((1,2,0))
            trg_img = trg_img.transpose((1,2,0))
            img = transform_img_source2target(img, trg_img, beta=self.fft_beta)
            # PIL --> Numpy array
            img = np.array(img)
            # (H,W,C) --> (C,H,W)
            img = np.transpose(img, (2,0,1))

        img = torch.from_numpy(np.ascontiguousarray(img , dtype=np.float32))
        img = self.normalize(img)

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
    def __init__(self, input_dim, output_dim=1, ch=64, dropout_p=0.):
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


def freeze_batchnorm(module):
    '''Makes the BN running statisitcs of a module static.
    '''
    for module in module.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum=0.
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            module.eval()


def save_model(backbone, decode_head, discr, iter_idx, path):
    file_path = os.path.join(path, f'feat_adapt_iter_{iter_idx}.pkl')
    model_dict = {'backbone': backbone.state_dict(),
                  'decode_head': decode_head.state_dict(),
                  'discr': discr.state_dict()}

    with open(file_path, 'wb') as file:
        pickle.dump(model_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(backbone, decode_head, discr, iter_idx, path):
    '''
    How to use:
        load_model(self.backbone, self.decode_head, self.discr, ...)
    '''
    file_path = os.path.join(path, f'feat_adapt_iter_{iter_idx}.pkl')
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    
    backbone.load_state_dict(checkpoint['backbone'])
    decode_head.load_state_dict(checkpoint['decode_head'])
    discr.load_state_dict(checkpoint['discr'])


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

        ##############
        #  DATASETS
        ##############
        # Datasets
        self.dataset_source = SourceDatasetA2D2(self.dataset_path_source, self.source_subset, self.cropbox)
        self.dataset_target = FeatureAdaptionDatasetCityscapes(self.dataset_path_target, self.cropbox)
        self.dataset_source_adapted = SourceDatasetA2D2(self.dataset_path_source, self.source_subset, self.cropbox, target_adaption_path=self.dataset_path_target)
        self.dataset_target_adapted = FeatureAdaptionDatasetCityscapes(self.dataset_path_target, self.cropbox, target_adaption_path=self.dataset_path_source)
        # Dataloaders
        self.dataloader_source = DataLoader(self.dataset_source, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        self.dataloader_target = DataLoader(self.dataset_target, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        self.dataloader_source_adapted = DataLoader(self.dataset_source_adapted, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        self.dataloader_target_adapted = DataLoader(self.dataset_target_adapted, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        # Dataloader iterators
        # Usage: _, batch = iter.__next()
        self.dataloader_source_iter = enumerate(self.dataloader_source)
        self.dataloader_target_iter = enumerate(self.dataloader_target)
        self.dataloader_source_adapted_iter = enumerate(self.dataloader_source_adapted)
        self.dataloader_target_adapted_iter = enumerate(self.dataloader_target_adapted)

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
            self.discr = StructDiscriminator(input_dim=self.discr_input_dim, output_dim=1, dropout_p=self.discr_dropout_p)
        else:
            raise Exception(f"Invalid discriminator type: {self.discr_type}")
        self.discr = self.discr.to('cuda')

        self.initialize_backbone_frozen = True
        if self.adaption_level == 'output':
            self.initialize_decode_head_frozen = True
        else:
            self.initialize_decode_head_frozen = False

        # Freeze batchnorm statistics
        freeze_batchnorm(self.backbone)
        freeze_batchnorm(self.decode_head)

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
        self.optimizer_discr = torch.optim.Adam(params, lr=self.discr_lr, weight_decay=0.0005, betas=(0.9, 0.99)) #momentum=self.sgd_momentum)

        #####################
        #  LOSS CRITERIONS
        #####################
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=255)
        self.KLDivLoss = nn.KLDivLoss()
        self.BCELoss = nn.BCEWithLogitsLoss()

        ################
        #  PARAMETERS
        ################

        # Optimization variables
        self.iter_idx = 0
        self.gen_steps = 0
        self.gen_steps_tot = 0
        self.loss_seg_list = deque(maxlen=100)
        self.loss_consis_list = deque(maxlen=100)
        self.loss_adv_feat_list = deque(maxlen=100)
        self.loss_discr_list = deque(maxlen=100)
        self.discr_acc_list = deque(maxlen=100)

        # So that generator is not optimized by chance
        for _ in range(100):
            self.discr_acc_list.append(0.)

        

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

    def _decode_head_forward_test_frozen(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_frozen.forward_test(x, img_metas, self.test_cfg)
        return seg_logits
    
    def encode_decode_frozen(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map."""
        x = self.extract_feat_frozen(img)
        out = self._decode_head_forward_test_frozen(x, img_metas)
        return out

    def write_log_entry(self, line):
        with open(self.train_log_file, 'a') as file:
            file.write(line)

    @staticmethod
    def _list_batch2tensor(list_batch):
        '''Transforms [tensor1, ..., tensorN] --> tensor (N, C, H, W)
        '''
        N = len(list_batch)
        H = list_batch[0].shape[1]
        W = list_batch[0].shape[2]
        tensor = torch.zeros((N, 3, H, W))
        for n in range(N):
            tensor[n] = list_batch[n]
        
        return tensor

    def model_forward_source(self, imgs, img_metas):
        '''
        Args:
            imgs: Tensor w. dim (N, C, H, W)
        '''
        if self.adaption_level == 'backbone':
            source_x = self.extract_feat_frozen(imgs)[-1]  # (2,512,64,128)
        elif self.adaption_level == 'output':
            source_x = self.encode_decode_frozen(imgs, img_metas)  # (2,19,128,256)
        else:
            raise ValueError(f"Given feature adaption level not supported ({self.adaption_level})")

        return source_x

    def model_forward_target(self, imgs, img_metas):
        '''
        Args:
            imgs: Tensor w. dim (N, C, H, W)
        '''
        if self.adaption_level == 'backbone':
            target_x = self.extract_feat(imgs)[-1]
        elif self.adaption_level == 'output':
            target_x = self.encode_decode(imgs, img_metas)
        else:
            raise ValueError(f"Given feature adaption level not supported ({self.adaption_level})")

        return target_x

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

        #######################################################################
        #  Be careful about what gradients that gets accumulated and updated
        #######################################################################

        # Copy backbone once after checkpoint loaded
        if self.initialize_backbone_frozen:
            self.backbone_frozen = copy.deepcopy(self.backbone)
            for params in self.backbone_frozen.parameters():
                params.requires_grad = False
            freeze_batchnorm(self.backbone_frozen)
            self.initialize_backbone_frozen = False
            
            if self.initialize_decode_head_frozen:
                self.decode_head_frozen = copy.deepcopy(self.decode_head)
                for params in self.decode_head_frozen.parameters():
                    params.requires_grad = False
                freeze_batchnorm(self.decode_head_frozen)
                self.initialize_decode_head_frozen = False

        # Run loop until finished
        while True:

            #_, batch_source = self.dataloader_source_iter.__next__()
            #_, batch_target = self.dataloader_target_iter.__next__()
            #_, batch_source_adapted = self.dataloader_source_adapted_iter.__next__()
            #_, batch_target_adapted = self.dataloader_target_adapted_iter.__next__()

            self.iter_idx += 1

            if self.iter_idx % self.print_interval == 0:
                seg_mean = np.mean(self.loss_seg_list)
                consis_mean = np.mean(self.loss_consis_list)
                adv_feat_mean = np.mean(self.loss_adv_feat_list)
                discr_mean = np.mean(self.loss_discr_list)
                discr_acc = np.mean(self.discr_acc_list)
                print(f"Iter {self.iter_idx} | Seg {seg_mean:.3f} | Consis {consis_mean:.3f}| Adv. feat {adv_feat_mean:.3f} | Discr {discr_mean:.3f} (acc {discr_acc:.0f}%)")
                self.write_log_entry(f"{self.iter_idx}, {seg_mean:.6f}, {consis_mean:.6f}, {adv_feat_mean:.6f}, {discr_mean:.6f}\n")        
                
                #self.gen_steps = 0
            
            self.optimizer_discr.zero_grad()
            self.optimizer_backbone.zero_grad()
            self.optimizer_decoder.zero_grad()

            # Learning gradients switch
            #     Backbone grad: O
            #     Decoder grad:  O
            #     Discr. grad:   X
            for params in self.backbone.parameters():
                params.requires_grad = True
            for params in self.decode_head.parameters():
                params.requires_grad = True
            for params in self.discr.parameters():
                params.requires_grad = False

            ##################################
            #  1: Supervised label loss
            ##################################
            # Output <-- target_model(adapted_source_img)
            # should correspond to source label

            #_, batch_source_adapted = self.dataloader_source_adapted_iter.__next__()
            _, batch_source_adapted = self.dataloader_source_iter.__next__()
            imgs_source_adapted = batch_source_adapted[0].to('cuda')
            labels_source_adapted = batch_source_adapted[1].to('cuda')

            # Adapted source samples w. Target model
            out_source_adapted_target = self.model_forward_target(imgs_source_adapted, img_metas)
            #out_source_adapted_target = self.model_forward_source(imgs_source_adapted, img_metas)
            out_source_adapted_target_resize = resize(input=out_source_adapted_target, size=imgs_source_adapted.shape[2:], mode='bilinear', align_corners=self.align_corners)

            loss = self.CrossEntropyLoss(out_source_adapted_target_resize, labels_source_adapted)
            loss = self.lambda_seg * loss
            loss.backward(retain_graph=True)
            self.loss_seg_list.append(loss.item())

            #a = np.transpose(imgs_source_adapted.cpu().numpy()[0], (1,2,0))
            #b = out_source_adapted_target_resize.argmax(dim=1)
            #b = b.detach().cpu().numpy()[0]
            #c = labels_source_adapted.cpu().numpy()[0]

            #c[c == 255] = -1
            
            #plt.subplot(1,3,1)
            #plt.imshow(a)
            #plt.subplot(1,3,2)
            #plt.imshow(b)
            #plt.subplot(1,3,3)
            #plt.imshow(c)
            #plt.show()
            #continue

            ################################
            #  2. Target consistency loss
            ################################
            # Output <-- target_model(target_img)
            # Output <-- source_model(adapted_target_img)
            # should be the same

            _, batch_target = self.dataloader_target_iter.__next__()
            _, batch_target_adapted = self.dataloader_target_adapted_iter.__next__()

            # Unpack 'list batch' --> image tensors
            imgs_target = self._list_batch2tensor(batch_target).to('cuda')
            imgs_target_adapted = self._list_batch2tensor(batch_target_adapted).to('cuda')

            # 'Target' output
            out_target = self.model_forward_target(imgs_target, img_metas)
            out_target_resize = resize(input=out_target, size=imgs_target.shape[2:], mode='bilinear', align_corners=self.align_corners)
            out_target_prob = F.softmax(out_target_resize, dim=1)
            out_target_problog = F.log_softmax(out_target_resize, dim=1)

            # 'Adapted target' output
            out_target_adapted = self.model_forward_source(imgs_target_adapted, img_metas)
            out_target_adapted_resize = resize(input=out_target_adapted, size=imgs_target.shape[2:], mode='bilinear', align_corners=self.align_corners)
            out_target_adapted_prob = F.softmax(out_target_adapted_resize, dim=1)
            out_target_adapted_problog = F.log_softmax(out_target_adapted_resize, dim=1)

            loss = self.KLDivLoss(out_target_problog, out_target_adapted_prob) + self.KLDivLoss(out_target_adapted_problog, out_target_prob)
            loss = self.lambda_consis * loss
            loss.backward(retain_graph=True)
            self.loss_consis_list.append(loss.item())

            #############################
            #  3. Adapt model features
            #############################
            # Train target model to make discr. missclasify target as source

            # Only train generator if discriminator is accurate <-- ???????
            if np.mean(self.discr_acc_list) > self.discr_acc_threshold:

                # Discriminator prediction
                target_pred = self.discr(out_target)

                # Dimensions for label
                N, _, d1, d2 = target_pred.shape

                # Discriminator label
                # NOTE: Reverse labels to train model to fool discriminator
                source_label = torch.ones((N, 1, d1, d2), dtype=torch.float).to('cuda')

                loss = self.BCELoss(target_pred, source_label)
                loss = self.lambda_discr * loss
                loss.backward(retain_graph=True)
                self.loss_adv_feat_list.append(loss.item())

            # Learning gradients switch
            #     Backbone grad: X
            #     Decoder grad:  X
            #     Discr. grad:   O
            for params in self.backbone.parameters():
                params.requires_grad = False
            for params in self.decode_head.parameters():
                params.requires_grad = False
            for params in self.discr.parameters():
                params.requires_grad = True

            ############################
            #  4. Train discriminator
            ############################
            # Train discr. to correctly classify source and target features

            out_source_adapted = self.model_forward_source(imgs_source_adapted, img_metas)

            out_source_adapted = out_source_adapted.detach()
            out_target = out_target.detach()

            # Discriminator prediction
            source_pred = self.discr(out_source_adapted)  # (2,1,3,3)
            target_pred = self.discr(out_target)
            pred_discr = torch.cat((source_pred, target_pred))

            # Dimensions for label
            N, _, d1, d2 = source_pred.shape

            # Discriminator label
            source_label = torch.ones((N, 1, d1, d2), dtype=torch.float)
            target_label = torch.zeros((N, 1, d1, d2), dtype=torch.float)
            label = torch.cat((source_label, target_label)).to('cuda')  # (4,1,3,3)

            loss = self.BCELoss(pred_discr, label)
            #loss = self.lambda_discr * loss  <-- Isolated module ==> no scaled loss???

            loss.backward(retain_graph=True)
            self.loss_discr_list.append(loss.item())

            # Compute discriminator accuracy
            pred_discr = pred_discr.detach().cpu().numpy()

            source_pred = np.zeros(pred_discr[0:N].shape)
            target_pred = np.zeros(pred_discr[N:].shape)
            # Only consider confident prediction
            source_pred[pred_discr[0:N] > 0.5] = 1.
            target_pred[pred_discr[N:] <= -0.5] = 1.
            
            correct_pred = 0.5*(np.mean(source_pred) + np.mean(target_pred))
            self.discr_acc_list.append(correct_pred * 100.)

            #######################
            #  OPTIMIZATION STEP
            #######################
            self.optimizer_discr.step()
            self.optimizer_backbone.step()
            self.optimizer_decoder.step()
            
            #################
            #  SAVE MODELS
            #################
            if self.iter_idx % self.save_interval == 0:
                print('Saving model')
                save_model(self.backbone, self.decode_head, self.discr, 
                           self.iter_idx, self.save_dir)

            # Reset iterators when cycled through
            if self.iter_idx % len(self.dataloader_source_adapted) == 0:
                self.dataloader_source_adapted_iter = enumerate(self.dataloader_source_adapted)    
            if self.iter_idx % len(self.dataloader_target) == 0:
                self.dataloader_target_iter = enumerate(self.dataloader_target)
            if self.iter_idx % len(self.dataloader_target_adapted) == 0:
                self.dataloader_target_adapted_iter = enumerate(self.dataloader_target_adapted)

        # Never ever again go beyond this point
        exit()

        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        # Zero loss to not interfere with feature adaption
        loss = torch.tensor(0., requires_grad=True)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        return outputs


    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image.
        
        Modified to replace modules with stored adapted modules.

        NOTE: Assumes that the intended iteration index is stored in a text file
        located in the root directory.
        
        """

        # Load adapted modules
        with open("iter_idx.txt", 'r') as file:
            load_idx = int(file.readline())
        if load_idx >= 0:
            load_model(self.backbone, self.decode_head, self.discr, load_idx, self.save_dir)

        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

