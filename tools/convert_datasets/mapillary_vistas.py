# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX = '_trainIds.png'

#   key: RGB color, value: trainidx
SEG_COLOR_DICT_V12 = {
    '[165, 42, 42]': 0,
    '[0, 192, 0]': 1,
    '[196, 196, 196]': 2,
    '[190, 153, 153]': 3,
    '[180, 165, 180]': 4,
    '[90, 120, 150]': 5,
    '[102, 102, 156]': 6,
    '[128, 64, 255]': 7,
    '[140, 140, 200]': 8,
    '[170, 170, 170]': 9,
    '[250, 170, 160]': 10,
    '[96, 96, 96]': 11,
    '[230, 150, 140]': 12,
    '[128, 64, 128]': 13,
    '[110, 110, 110]': 14,
    '[244, 35, 232]': 15,
    '[150, 100, 100]': 16,
    '[70, 70, 70]': 17,
    '[150, 120, 90]': 18,
    '[220, 20, 60]': 19,
    '[255, 0, 0]': 20,
    '[255, 0, 100]': 21,
    '[255, 0, 200]': 22,
    '[200, 128, 128]': 23,
    '[255, 255, 255]': 24,
    '[64, 170, 64]': 25,
    '[230, 160, 50]': 26,
    '[70, 130, 180]': 27,
    '[190, 255, 255]': 28,
    '[152, 251, 152]': 29,
    '[107, 142, 35]': 30,
    '[0, 170, 30]': 31,
    '[255, 255, 128]': 32,
    '[250, 0, 30]': 33,
    '[100, 140, 180]': 34,
    '[220, 220, 220]': 35,
    '[220, 128, 128]': 36,
    '[222, 40, 40]': 37,
    '[100, 170, 30]': 38,
    '[40, 40, 40]': 39,
    '[33, 33, 33]': 40,
    '[100, 128, 160]': 41,
    '[142, 0, 0]': 42,
    '[70, 100, 150]': 43,
    '[210, 170, 100]': 44,
    '[153, 153, 153]': 45,
    '[128, 128, 128]': 46,
    '[0, 0, 80]': 47,
    '[250, 170, 30]': 48,
    '[192, 192, 192]': 49,
    '[220, 220, 0]': 50,
    '[140, 140, 20]': 51,
    '[119, 11, 32]': 52,
    '[150, 0, 255]': 53,
    '[0, 60, 100]': 54,
    '[0, 0, 142]': 55,
    '[0, 0, 90]': 56,
    '[0, 0, 230]': 57,
    '[0, 80, 100]': 58,
    '[128, 64, 64]': 59,
    '[0, 0, 110]': 60,
    '[0, 0, 70]': 61,
    '[0, 0, 192]': 62,
    '[32, 32, 32]': 63,
    '[120, 10, 10]': 64,
    '[0, 0, 0]': 65
}

TRAINIDX_MV2CITY = {
    13: 0,  # Road
    24: 0,
    41: 0,
    2: 1,  # Sidewalk
    15: 1,
    17: 2,  # Building
    6: 3,  # Wall
    3: 4,  # Fence
    45: 5,  # Pole
    47: 5,
    48: 6,  # Traffic light
    50: 7,  # Traffic sign
    30: 8,  # Vegetation
    29: 9,  # Terrain
    27: 10,  # Sky
    19: 11,  # Person
    20: 12,  # Rider
    21: 12,
    22: 12,
    55: 13,  # Car
    61: 14,  # Truck
    54: 15,  # Bus
    58: 16,  # Train
    57: 17,  # Motorcycle
    52: 18,  # Bicycle
}

# Takes longer to compute mask for all RGBs than finding unique RGBs
RGBS = [
    np.fromstring(rgb[1:-1], dtype=int, sep=',').tolist()
    for rgb in SEG_COLOR_DICT_V12.keys()
]

KEEP_IDXS = list(TRAINIDX_MV2CITY.keys())


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-compatible label filename."""
    # Ensure that label filenames are modified only once
    if 'TrainIds.png' in label_filepath:
        return label_filepath

    if label_choice == 'cityscapes':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX)
    else:
        raise ValueError
    return label_filepath


def convert_cityscapes_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label following the Cityscapes 'trainids' format.

    The new image is saved into the same directory as the original image having
    an additional suffix.
    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    # Read label file as Numpy array (H, W, 3)
    orig_label = mmcv.imread(label_filepath, channel_order='rgb')

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    mod_label = ignore_id * np.ones((H, W), dtype=int)

    # Create a list of RGB values = [ [R,G,B]_0, ... ] existing in label
    # NOTE: Takes longer to compute mask for all RGBs than finding unique RGBs
    rgbs = orig_label.reshape(-1, 3)
    rgbs = np.unique(rgbs, axis=0)
    rgbs = [rgbs[idx].tolist() for idx in range(rgbs.shape[0])]

    for rgb in rgbs:
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = (orig_label == rgb).all(-1)
        # Segment masked elements with 'trainIds' value
        mv_idx = SEG_COLOR_DICT_V12[str(rgb)]
        if mv_idx in KEEP_IDXS:
            city_idx = TRAINIDX_MV2CITY[mv_idx]
            mod_label[mask] = city_idx

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, 'cityscapes')
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def get_idx2rgb_conversion(vistas_path, annotation_version):
    """Creates an 'RGB' --> 'trainidx' mapping according to the config file.

    NOTE: Only use once for hardcoding mappings. Parallelization of function
          does not allow multiple arguments.
    Args:
        vistas_path: Path to Mapillary Vistas root directory.
        annotation_version: String specifying annotation version.

    Returns:
        Dict with keys str([R, G, B]) providing 'trainidx' integers
    """
    if annotation_version == 'v1.2':
        config_filename = 'config_v1.2.json'
    elif annotation_version == 'v2.0':
        config_filename = 'config_v2.0.json'
    else:
        raise Exception(f'Unsupported annotation ver. ({annotation_version})')

    config_filepath = os.path.join(vistas_path, config_filename)
    if not os.path.isfile(config_filepath):
        raise IOError(f'JSON file not found\n\t{config_filepath}')
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    labels = config['labels']

    rgb2idx = {}
    for idx, label in enumerate(labels):
        rgb = label['color']
        rgb2idx[str(rgb)] = idx

    return rgb2idx


def restructure_vistas_directory(vistas_path,
                                 train_on_val_and_test=False,
                                 use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the Mapillary Vistas dataset conform to the mmsegmentation
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
        vistas_path: Absolute path to the Mapillary Vistas 'vistas/' directory.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'testing'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'testing'))

    for split in ['training', 'validation', 'testing']:
        img_filepaths = glob.glob(f'{vistas_path}/{split}/images/*.jpg')

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:

            img_filename = img_filepath.split('/')[-1]

            ann_filename = img_filename[:-4] + LABEL_SUFFIX
            ann_filepath = f'{vistas_path}/{split}/v2.0/labels/{ann_filename}'

            img_linkpath = f'{vistas_path}/img_dir/{split}/{img_filename}'
            if split == 'testing':
                ann_linkpath = None
            else:
                ann_linkpath = f'{vistas_path}/ann_dir/{split}/{ann_filename}'

            if use_symlinks:
                # NOTE: Can only create new symlinks if no priors ones exists
                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    if split != 'testing':
                        symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

            else:
                copyfile(img_filepath, img_linkpath)
                if split != 'testing':
                    copyfile(ann_filepath, ann_linkpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary Vistas annotations to trainIds')
    parser.add_argument(
        'vistas_path',
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
        '--annotation-version',
        default='v1.2',
        help='Which annotation set to use (v1.2 or v2.0)')
    parser.add_argument(
        '--choice',
        default='cityscapes',
        help='Label conversion type choice: \'cityscapes\' (19 classes)')
    parser.add_argument(
        '--train-on-val-and-test',
        dest='train_on_val_and_test',
        action='store_true',
        help='Use validation and test samples as training samples')
    parser.set_defaults(train_on_val_and_test=False)
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
    """A script for making the Mapillary Vistas dataset compatible with
    mmsegmentation.

    Directory structure:
        mapillary_vistas/
            testing/
                images/
                    ___QXeb8e952hTD6EaQVEQ.jpg
                    ...
            training/
                images/
                    __CRyFzoDOXn6unQ6a3DnQ.jpg
                    ...
                v1.2/
                    labels/
                        __CRyFzoDOXn6unQ6a3DnQ.png
                        ...
                v2.0/
                    labels/
                        __CRyFzoDOXn6unQ6a3DnQ.png
                        ...
            Validation/
                images/
                    _1Gn_xkw7sa_i9GU4mkxxQ.jpg
                    ...
                v1.2/
                    labels/
                        _1Gn_xkw7sa_i9GU4mkxxQ.png
                        ...
                v2.0/
                    labels/
                        _1Gn_xkw7sa_i9GU4mkxxQ.png
                        ...
            config_v1.2.json  <-- 65 object classes
            config_v2.0.json  <-- 124 object classes

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/mapillary_vistas)

    Segmentation label conversion:
        The function 'convert_TYPE_trainids()' converts all RGB segmentation to
        their corresponding categorical segmentation and saves them as new
        label image files.
        Label choice 'cityscapes' (default) results in labels with 19 classes
        with the filename suffix '_city_trainIds.png'.

    Dataset split:
        Arranges samples into 'train', 'val', and 'test' splits according to
        predetermined directory structure
    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is useful in the synthetic-to-real domain adaptation
    experiment setting.
    Add `--nproc N` for multiprocessing using N threads.
    Example usage:
        python tools/convert_datasets/mapillary_vistas.py
            abs_path/to/mapillary_vistas
    """
    args = parse_args()
    vistas_path = args.vistas_path
    out_dir = args.out_dir if args.out_dir else vistas_path
    ann_ver = args.annotation_version
    mmcv.mkdir_or_exist(out_dir)

    # Get mapping to convert RGB --> trainidx integer map
    # idx2rgb = get_idx2rgb_conversion(vistas_path, ann_ver)
    # print(idx2rgb)

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:

        # Create a list of filepaths to all original labels
        suffix_wo_png = LABEL_SUFFIX[:-4]
        label_filepaths = glob.glob(
            osp.join(vistas_path,
                     f'*/{ann_ver}/labels/*[!{suffix_wo_png}].png'))

        seg_choice = args.choice
        if seg_choice == 'cityscapes':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_cityscapes_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_cityscapes_trainids,
                                    label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_vistas_directory(out_dir, args.train_on_val_and_test,
                                     args.symlink)


if __name__ == '__main__':
    main()
