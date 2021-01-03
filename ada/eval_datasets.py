import numpy as np
import glob
import os.path as osp
import re


def get_cityscapes_eval_samples(root_path, dirs=['train', 'val']):
    '''Creates two lists of paths to all images and labels.

    Args:
        root_path: Path to Cityscapes root.
        dirs: List of directories to add samples from
    
    Returns:
        Lists of ordered image and label paths.
    '''

    img_paths = []
    label_paths = []

    for dir in dirs:
        img_paths += glob.glob(osp.join(root_path, f'leftImg8bit/{dir}/*/*.png'))
        label_paths += glob.glob(osp.join(root_path, f'gtFine/{dir}/*/*_labelTrainIds.png'))

    img_paths.sort()
    label_paths.sort()

    return img_paths, label_paths
