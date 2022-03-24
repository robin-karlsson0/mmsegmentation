# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np
from PIL import Image

random.seed(14)

ID_CITYSCAPES_DICT = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
    34: 2,
    35: 4,
    36: 255,
    37: 5,
    38: 255,
    39: 255,
    40: 255,
    41: 255,
    42: 255,
    43: 255,
    44: 255,
    -1: 255,
}


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    label_filepath = label_filepath.replace('.png', '_labelTrainIds.png')
    return label_filepath


def convert_to_trainid(label_filepath):

    label = np.array(Image.open(label_filepath))
    label_copy = label.copy()
    for clsID, trID in ID_CITYSCAPES_DICT.items():
        label_copy[label == clsID] = trID

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath)
    label_img = label_copy.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def ann2img(filepath):
    filepath = filepath.replace('/data_2d_semantics/', '/data_2d_raw/')
    filepath = filepath.replace('/train/', '/')
    filepath = filepath.replace('/semantic/', '/data_rect/')
    return filepath


def restructure_kitti360_directory(kitti360_path,
                                   num_val_samples=500,
                                   use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the KITTI-360 dataset conform to the mmsegmentation
    frameworks expected dataset structure.

    └── img_dir
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── ann_dir
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        |   ...
        ...
    Args:
        kitti360_path: Absolute path to the 'KITTI-360/' directory.
        num_val_samples: Number of randomlselected samples used for validation.
        use_symlinks: Symbolically link existing files in the original dataset
                      directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(kitti360_path, 'ann_dir', 'val'))

    config_path = os.path.join(kitti360_path, 'data_2d_semantics', 'train')
    train_filename = '2013_05_28_drive_train_frames.txt'
    train_frames = np.loadtxt(
        os.path.join(config_path, train_filename), dtype=str)

    val_filename = '2013_05_28_drive_val_frames.txt'
    val_frames = np.loadtxt(os.path.join(config_path, val_filename), dtype=str)

    train_idx = 0
    for idx in range(train_frames.shape[0]):
        img_path = str(train_frames[idx, 0])
        ann_path = str(train_frames[idx, 1])
        ann_path = modify_label_filename(ann_path)

        img_path = os.path.join(kitti360_path, img_path)
        ann_path = os.path.join(kitti360_path, ann_path)

        img_name = f'train_{train_idx}.png'
        ann_name = f'train_{train_idx}.png'

        img_linkpath = osp.join(kitti360_path, 'img_dir', 'train', img_name)
        ann_linkpath = osp.join(kitti360_path, 'ann_dir', 'train', ann_name)

        if use_symlinks:
            # NOTE: Can only create new symlinks if no priors ones exists
            try:
                symlink(img_path, img_linkpath)
            except FileExistsError:
                pass
            try:
                symlink(ann_path, ann_linkpath)
            except FileExistsError:
                pass

        else:
            copyfile(img_path, img_linkpath)
            copyfile(ann_path, ann_linkpath)

        train_idx += 1

    val_idx = 0
    for idx in range(val_frames.shape[0]):
        img_path = str(val_frames[idx, 0])
        ann_path = str(val_frames[idx, 1])
        ann_path = modify_label_filename(ann_path)

        img_path = os.path.join(kitti360_path, img_path)
        ann_path = os.path.join(kitti360_path, ann_path)

        img_name = f'val_{val_idx}.png'
        ann_name = f'val_{val_idx}.png'

        img_linkpath = osp.join(kitti360_path, 'img_dir', 'val', img_name)
        ann_linkpath = osp.join(kitti360_path, 'ann_dir', 'val', ann_name)

        if use_symlinks:
            # NOTE: Can only create new symlinks if no priors ones exists
            try:
                symlink(img_path, img_linkpath)
            except FileExistsError:
                pass
            try:
                symlink(ann_path, ann_linkpath)
            except FileExistsError:
                pass

        else:
            copyfile(img_path, img_linkpath)
            copyfile(ann_path, ann_linkpath)

        val_idx += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary Vistas annotations to trainIds')
    parser.add_argument(
        'kitti360_path',
        help='Mapillary vistas segmentation data absolute path\
                           (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-convert',
        dest='convert',
        action='store_false',
        help='Skips converting label images')
    parser.set_defaults(convert=True)
    parser.add_argument(
        '--no-restruct',
        dest='restruct',
        action='store_false',
        help='Skips restructuring directory structure')
    parser.set_defaults(restruct=True)
    parser.add_argument(
        '--num-val-samples',
        type=int,
        default=500,
        help='Number of random samples to form validation set.')
    parser.add_argument(
        '--choice', default='cityscapes', help='Label conversion type choice')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    parser.add_argument(
        '--no-symlink',
        dest='symlink',
        action='store_false',
        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """A script for making the KITTI-360 dataset compatible with
    mmsegmentation.

    Directory structure:
        KITTI-360/
            data_2d_raw/
                2013_05_28_drive_0000_sync/
                    image_00/
                        data_rect/
                            0000000000.png
                            ...
                    ...
                ...
            data_2d_semantics/
                train/
                    2013_05_28_drive_0000_sync/
                        image_00/
                            semantic/
                                0000000253.png
                    ...

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/KITTI-360)

    Add `--nproc N` for multiprocessing using N threads.
    Example usage:
        python tools/convert_datasets/kitti360.py
            abs_path/to/mapillary_vistas
    """
    args = parse_args()
    kitti360_path = args.kitti360_path
    out_dir = args.out_dir if args.out_dir else kitti360_path
    mmcv.mkdir_or_exist(out_dir)

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:
        # Create a list of filepaths to all original labels
        # NOTE: Original label files have a number before '.png'
        label_filepaths = glob.glob(
            osp.join(kitti360_path,
                     'data_2d_semantics/train/*/image_00/semantic/*[0-9].png'))

        seg_choice = args.choice
        if seg_choice == 'cityscapes':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_to_trainid,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_to_trainid, label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_kitti360_directory(out_dir, args.num_val_samples,
                                       args.symlink)


if __name__ == '__main__':
    main()
