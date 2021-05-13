import numpy as np
import mmcv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob

from ada.fft_domain_transfer import transform_img_source2target, ImgFetcher

import matplotlib.pyplot as plt
import copy

class ImageDomainTransformer:
    """
    """

    def __init__(self, target_img_path, fft_beta, crop):
        """
        Args:
            target_img_path:
            fft_beta: A2D2 --> Cityscapes | 0.002
            crop: Tuple of integers (height, width)
        """

        if os.path.isdir(target_img_path) == False:
            raise Exception(f"Target image directory is invalid {target_img_path}")
        if len(os.listdir(target_img_path)) == 0:
            raise Exception(f"Target image directory is empty")
        
        self.target_img_fetcher = ImgFetcher(target_img_path)
        self.fft_beta = fft_beta

        self.crop_h = crop[0]
        self.crop_w = crop[1]

        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        ])
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(
                mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375],
                std=[1/58.395, 1/57.12, 1/57.375])
        ])
    
    def transform(self, img):
        """

        NOTE: Default RGB img array dim: (H,W,C)

        Arg:
            img: Numpy RGB img w. dim (H,W,C)
        
        Returns:
            Transformed Numpy RGB image with dim (H,W,C)
        """
        # Load 'target' image
        trg_img_path = self.target_img_fetcher.get_img_path()
        trg_img = mmcv.imread(trg_img_path, channel_order='rgb')
        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        trg_img = self._random_crop(trg_img, random_ratio_1, random_ratio_2)

        #img_copy = copy.deepcopy(img)

        # Transform source image
        img = transform_img_source2target(img, trg_img, beta=self.fft_beta)

        # PIL --> Numpy array
        img = np.array(img, dtype=np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(img)
        #plt.subplot(1,3,2)
        #plt.imshow(trg_img)
        #plt.subplot(1,3,3)
        #plt.imshow(img_copy)
        #plt.show()
        #exit()

        return img
    
    def transform_tensor(self, tensor):
        """
        Args:
            tensor: BGR image tensor with dim (C, H, W).
        """
        # Undo normalization
        tensor = self.inv_normalize(tensor)

        # Convert to Numpy image array dim (C,H,W) --> (H,W,C)
        img = tensor.cpu().numpy()
        img = img.astype(np.uint8)
        img = np.transpose(img, (1,2,0))

        # Transform image
        img = self.transform(img)

        # Transpose image (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))
        tensor = torch.from_numpy(np.ascontiguousarray(img ,dtype=np.float32)).to("cuda")
        tensor = self.normalize(tensor)

        return tensor

    def transform_img_batch(self, img_batch):
        """Returns the input image tensor (img) in the target domain spectrum.

        Args:
            img_list: Tensors RGB image with dim (N,C,H,W).
        """
        transformed_img_batch = torch.zeros(img_batch.shape).to("cuda")
        for batch_idx in range(img_batch.shape[0]):
            tensor = img_batch[batch_idx]
            tensor = self.transform_tensor(tensor)
            transformed_img_batch[batch_idx] = tensor
        
        return transformed_img_batch
    
    def _random_crop(self, img, random_ratio_1, random_ratio_2):
        """
        Args:
            img: Numpy image with dim (H,W,C).
        """
        dim = len(img.shape)
        if dim == 3:
            crop_start_h = int(np.round((img.shape[0] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[1] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[h0:h1, w0:w1, :]
        elif dim == 2:
            crop_start_h = int(np.round((img.shape[0] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[1] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[h0:h1, w0:w1]
        else:
            raise ValueError(f"Unsupported dimension for image crop ({img.shape})")

        return img


class SemanticSegDataset(Dataset):

    def __init__(self, root_dir):
        """
        """

        if os.path.isdir(root_dir):
            self.root_dir = root_dir
        else:
            raise FileNotFoundError(f"Dataset directories do not exists \
                                     ({root_dir})")
        
        self.samples_N = 0
    
    def __len__(self):
        return self.samples_N

    def _random_crop(self, img, random_ratio_1, random_ratio_2):
        """
        Args:
            img: Numpy image with dim (H,W,C).
        """
        dim = len(img.shape)
        if dim == 3:
            crop_start_h = int(np.round((img.shape[0] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[1] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[h0:h1, w0:w1, :]
        elif dim == 2:
            crop_start_h = int(np.round((img.shape[0] - self.crop_h) * random_ratio_1))
            crop_start_w = int(np.round((img.shape[1] - self.crop_w) * random_ratio_2))

            h0 = crop_start_h
            h1 = crop_start_h+self.crop_h
            w0 = crop_start_w
            w1 = crop_start_w+self.crop_w

            img = img[h0:h1, w0:w1]
        else:
            raise ValueError(f"Unsupported dimension for image crop ({img.shape})")

        return img

# OBSOLETE
class SourceDatasetA2D2(SemanticSegDataset):
    """
    dataloader_iter = enumerate(dataloader)
    
    _, batch = dataloader_iter.__next__()

    batch --> [imgs_tensor, labels_tensor]
        imgs_tensor   --> (N, C, H, W)
        labels_tensor --> (N, H, W)
    """

    def __init__(self, root_dir, subset, crop, target_adaption_path=None, label_postfix='_labelTrainIds'):
        """
        Args:
            subset: String specifying subset to load.
                    Options: 'train', 'val', 'test'
        """
        super().__init__(root_dir)

        if subset not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid dataset 'subset' ({subset})")

        self.img_dir = os.path.join(root_dir, 'img_dir', subset)
        self.ann_dir = os.path.join(root_dir, 'ann_dir', subset)

        if ((os.path.isdir(self.img_dir) is False) or
            (os.path.isdir(self.ann_dir) is False)):
           raise FileNotFoundError(f"Invalid dataset paths ({self.img_dir}, \
                                     {self.ann_dir})")

        # Create list of image filenames (splice/subdir/files)
        self.img_list = glob.glob(f'{self.img_dir}/*/*.png')

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

# OBSOLETE
class FeatureAdaptionDatasetCityscapes(SemanticSegDataset):
    """
    dataloader_iter = enumerate(dataloader)
    
    _, batch = dataloader_iter.__next__()

    batch --> [img_1, ..., img_N] for batch size N

    TODO: Refactor
    """

    def __init__(self, root_dir, crop, target_adaption_path=None):
        """
        Args:
            label: Integer (source, target) = (1, 0)
        """
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
            self.fft_beta = 0.0015 #0.0015  # Cityscapes --> A2D2
        else:
            self.target_img_fetcher = None

    def __getitem__(self, idx):
        
        filepath = os.path.join(self.root_dir, f"{idx}.png")
        img = mmcv.imread(filepath, channel_order='rgb')
        # BGR --> RGB
        #img = mmcv.bgr2rgb(img)

        # (H,W,C) --> (C,H,W)
        #img = np.transpose(img, (2,0,1))

        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        img = self._random_crop(img, random_ratio_1, random_ratio_2)

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1)

         # Image adaption
        if self.target_img_fetcher is not None:
            trg_img_path = self.target_img_fetcher.get_img_path()
            trg_img = mmcv.imread(trg_img_path, channel_order='rgb')
            #trg_img = mmcv.bgr2rgb(trg_img)
            # (H,W,C) --> (C,H,W)
            #trg_img = np.transpose(trg_img, (2,0,1))
            # Random crop
            trg_img = self._random_crop(trg_img, random_ratio_1, random_ratio_2)
            # (C,H,W) --> (H,W,C)
            #img = img.transpose((1,2,0))
            #trg_img = trg_img.transpose((1,2,0))
            img = transform_img_source2target(img, trg_img, beta=self.fft_beta)
            # PIL --> Numpy array
            img = np.array(img)

        # Transpose image (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))
        img = torch.from_numpy(np.ascontiguousarray(img , dtype=np.float32))
        img = self.normalize(img)

        return img

# OBSOLETE
class FeatureAdaptionDatasetA2D2(SemanticSegDataset):
    """
    dataloader_iter = enumerate(dataloader)
    
    _, batch = dataloader_iter.__next__()

    batch --> [img_1, ..., img_N] for batch size N
    """
    def __init__(self, root_dir, crop, target_adaption_path=None):
        """
        Args:
            label: Integer (source, target) = (1, 0)
        """
        super().__init__(root_dir)

        self.samples = glob.glob(f'{self.root_dir}/img_dir/train/*.png')
        self.samples_N = len(self.samples)

        self.crop_h = crop[0]
        self.crop_w = crop[1]

        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        ])

    def __getitem__(self, idx):

        filepath = self.samples[idx]
        img = mmcv.imread(filepath, channel_order='bgr')

        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        img = self._random_crop(img, random_ratio_1, random_ratio_2)

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1)
        
        # Transpose image (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))
        img = torch.from_numpy(np.ascontiguousarray(img , dtype=np.float32))
        img = self.normalize(img)

        return img


class TargetDataset(SemanticSegDataset):
    '''Dataset class returning input images from the 'target' dataset.
    '''
    def __init__(self, root_dir, crop, dataset):
        '''
        Args:
            root_dir (str): Absolute path to the root of each dataset.
                            ../a2d2/camera_lidar_semantic/
                            ../cityscapes/
            crop (int,int): Tuple of (height, width).
            dataset (str): 'a2d2' or 'cityscapes'
        '''
        if dataset == 'a2d2':
            self.samples = glob.glob(f'{root_dir}/img_dir/train/*/*.png')
        elif dataset == 'cityscapes':
            self.samples = glob.glob(f'{root_dir}/leftImg8bit/train/*/*.png')
        else:
            raise Exception('Invalid dataset type:', dataset)

        # Check for empty sample list
        if not self.samples:
            raise Exception(f"No samples loaded in TargetDataset\n  {root_dir}\n  {dataset}")

        self.samples_N = len(self.samples)

        self.crop_h = crop[0]
        self.crop_w = crop[1]

        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        ])

    def __getitem__(self, idx):
        '''Returns a cropped, flipped, and normalized sample by index.
        NOTE: Images are read as BGR to match the mmsegmentation dataloader images.
        '''
        filepath = self.samples[idx]
        img = mmcv.imread(filepath, channel_order='bgr')

        # Random crop
        random_ratio_1 = np.random.random()
        random_ratio_2 = np.random.random()
        img = self._random_crop(img, random_ratio_1, random_ratio_2)

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1)
        
        # Transpose image (H,W,C) --> (C,H,W)
        img = np.transpose(img, (2,0,1))
        img = torch.from_numpy(np.ascontiguousarray(img , dtype=np.float32))
        img = self.normalize(img)

        return img


class FeatDiscriminator(torch.nn.Module):
    """Discriminating independent feature vectors using 1x1 convolutions.
    """
    def __init__(self, input_dim=512, output_dim=2, dropout_p=0.5):
        """
        """
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
    """Discriminating contextual features using 2D convolutions.
    """
    def __init__(self, input_dim, output_dim=1, ch=64):
        """
        """
        super().__init__()

        self.D = nn.Sequential(
            # 1: 64
            nn.Conv2d(input_dim, ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 2: 128
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 3: 256
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 4: 512
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 5: 512
            nn.Conv2d(ch*8, ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 6: 1
            nn.Conv2d(ch*8, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.D(x)
        return out
