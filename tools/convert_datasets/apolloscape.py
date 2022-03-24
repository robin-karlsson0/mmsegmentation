import argparse
import glob
import os
import os.path as osp
import random
from logging import root
from os import symlink
from shutil import copyfile

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import PIL.Image as Image

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX_CITYSCAPES = '_labelTrainIds.png'

# Ref: http://apolloscape.auto/scene.html
CLASS_TO_TRAINID = {
    0: 255,
    1: 255,
    17: 10,
    33: 13,
    161: 13,
    34: 17,
    162: 17,
    35: 18,
    163: 18,
    36: 11,
    164: 11,
    37: 12,
    165: 12,
    38: 14,
    166: 14,
    39: 15,
    167: 15,
    40: 255,
    168: 255,
    49: 0,
    50: 1,
    65: 255,
    66: 255,
    67: 4,
    81: 6,
    82: 5,
    83: 7,
    84: 3,
    85: 255,
    86: 255,
    97: 2,
    98: 255,
    99: 255,
    100: 255,
    113: 8,
    255: 255,
}
# Class   	            Class   Cityscapes trainids
# others	            0  	    255
# rover	                1	    255
# sky	                17	    10
# car	                33	    13
# car_groups	        161	    13
# motorbicycle	        34	    17
# motorbicycle_group	162	    17
# bicycle	            35	    18
# bicycle_group	        163	    18
# person	            36	    11
# person_group	        164	    11
# rider	                37	    12
# rider_group	        165	    12
# truck	                38	    14
# truck_group	        166	    14
# bus	                39	    15
# bus_group	            167	    15
# tricycle	            40	    255
# tricycle_group	    168	    255
# road	                49	    0
# siderwalk	            50	    1
# traffic_cone	        65	    255
# road_pile	            66	    255
# fence	                67	    4
# traffic_light	        81	    6
# pole	                82	    5
# traffic_sign	        83	    7
# wall	                84	    3
# dustbin	            85	    255
# billboard	            86	    255
# building	            97	    2
# bridge	            98	    255
# tunnel	            99	    255
# overpass	            100	    255
# vegatation	        113	    8
# unlabeled	            255	    255


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-combatible label filename."""
    if label_choice == 'cityscapes':
        label_filepath = label_filepath.replace('.png',
                                                LABEL_SUFFIX_CITYSCAPES)
    else:
        raise ValueError

    return label_filepath


def convert_cityscapes_trainids(ann_filepath, ignore_id=255, overwrite=True):
    """Saves a new semantic label replacing RGB values with label categories.
    The new image is saved into the same directory as the original image having
    an additional suffix.
    Args:
        ann_filepath: Path to the original semantic annotation label.
        ignore_id: Default value for unlabeled elements.
    """
    label = np.array(Image.open(ann_filepath))
    label_copy = ignore_id * np.ones(label.shape, dtype=int)
    for clsID, trID in CLASS_TO_TRAINID.items():
        label_copy[label == clsID] = trID

    # Save new 'trainids' semantic label
    ann_filepath = modify_label_filename(ann_filepath, 'cityscapes')
    label_img = label_copy.astype(np.uint8)
    mmcv.imwrite(label_img, ann_filepath)


def create_split_dir(img_filepaths,
                     ann_filepaths,
                     split,
                     root_path,
                     use_symlinks=True):
    """Creates dataset split directory from given file lists using symbolic
    links or copying files.
    Args:
        img_filepaths: List of filepaths as strings.
        ann_filepaths:
        split: String denoting split (i.e. 'train', 'val', or 'test'),
        root_path: Dataset root directory (.../camera_lidar_semantic/)
        use_symlinks: Symbolically link existing files in the original
                      dataset directory. If false, files will be copied.
    Raises:
        FileExistError: In case of pre-existing files when trying to create new
                        symbolic links.
    """
    assert split in ['train', 'val', 'test']

    for img_filepath, ann_filepath in zip(img_filepaths, ann_filepaths):
        # Partitions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[-1]
        ann_filename = ann_filepath.rpartition('/')[-1]

        img_link_path = osp.join(root_path, 'images', split, img_filename)
        ann_link_path = osp.join(root_path, 'annotations', split, ann_filename)

        if use_symlinks:
            # NOTE: Can only create new symlinks if no priors ones exists
            try:
                symlink(img_filepath, img_link_path)
            except FileExistsError:
                pass
            try:
                symlink(ann_filepath, ann_link_path)
            except FileExistsError:
                pass

        else:
            copyfile(img_filepath, img_link_path)
            copyfile(ann_filepath, ann_link_path)


def restructure_directory(root_path,
                          train_samples,
                          val_samples,
                          test_samples,
                          label_choice,
                          use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the dataset conform to the mmsegmentation frameworks
    expected dataset structure.
    lane_segmentation/
    └── images
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── annotations
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        |   ...
        ...
    
    Args:
        root_path: Absolute path to the Apolloscape 'lane_segmentation'
                   directory.
        val_ratio: Float value representing ratio of validation samples.
        test_ratio: Float value representing ratio of test samples.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_choice:
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(root_path, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'ann_dir', 'test'))

    # if label_choice == 'cityscapes':
    #     label_suffix = LABEL_SUFFIX_CITYSCAPES
    # else:
    #     raise ValueError

    train_idx = 0
    for idx in range(train_samples.shape[0]):

        img_path = str(train_samples[idx, 0])
        img_path = img_path.replace(' ', '_')
        ann_path = str(train_samples[idx, 1])
        ann_path = ann_path.replace(' ', '_')
        ann_path = modify_label_filename(ann_path, label_choice)

        img_path = os.path.join(root_path, img_path)
        ann_path = os.path.join(root_path, ann_path)

        img_name = f'train_{train_idx}.png'
        ann_name = f'train_{train_idx}.png'

        img_linkpath = osp.join(root_path, 'img_dir', 'train', img_name)
        ann_linkpath = osp.join(root_path, 'ann_dir', 'train', ann_name)

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
    for idx in range(val_samples.shape[0]):

        img_path = str(val_samples[idx, 0])
        img_path = img_path.replace(' ', '_')
        ann_path = str(val_samples[idx, 1])
        ann_path = ann_path.replace(' ', '_')
        ann_path = modify_label_filename(ann_path, label_choice)

        img_path = os.path.join(root_path, img_path)
        ann_path = os.path.join(root_path, ann_path)

        img_name = f'val_{val_idx}.png'
        ann_name = f'val_{val_idx}.png'

        img_linkpath = osp.join(root_path, 'img_dir', 'val', img_name)
        ann_linkpath = osp.join(root_path, 'ann_dir', 'val', ann_name)

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

    test_idx = 0
    for idx in range(test_samples.shape[0]):

        img_path = str(test_samples[idx, 0])
        img_path = img_path.replace(' ', '_')
        ann_path = str(test_samples[idx, 1])
        ann_path = ann_path.replace(' ', '_')
        ann_path = modify_label_filename(ann_path, label_choice)

        img_path = os.path.join(root_path, img_path)
        ann_path = os.path.join(root_path, ann_path)

        img_name = f'test_{test_idx}.png'
        ann_name = f'test_{test_idx}.png'

        img_linkpath = osp.join(root_path, 'img_dir', 'test', img_name)
        ann_linkpath = osp.join(root_path, 'ann_dir', 'test', ann_name)

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

        test_idx += 1


# def generate_crop(imgpath, H=2710, W=3384, H_cropped=1010):
#     """Generate cropped versions of images containing the bottom part."""
#     bbox = np.array([0, H - H_cropped, W, H])
#     img = mmcv.imread(imgpath, channel_order='bgr')
#     img = mmcv.imcrop(img, bbox)
#     # Create cropped image path while avoiding recursive duplication
#     file_ending = imgpath.split('_')[-1]
#     if file_ending == '5.jpg' or file_ending == '6.jpg':
#         crop_imgpath = imgpath.replace('.jpg', '_crop.jpg')
#     elif file_ending == 'bin.png':
#         crop_imgpath = imgpath.replace('bin.png', 'bin_crop.png')
#     else:
#         raise ValueError(f'Invalid file ending: {file_ending}')
#
#     mmcv.imwrite(img, crop_imgpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Apolloscape Lane Segmentation annotations to \
                     TrainIds')
    parser.add_argument(
        'root_path',
        help='Apolloscape Lane Segmentation directory absolute path \
              (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    # parser.add_argument(
    #     '--no-crop-gen',
    #     dest='crop_gen',
    #     action='store_false',
    #     help='Skips generating cropped versions of images and labels')
    # parser.set_defaults(crop_gen=True)
    # parser.add_argument(
    #     '--no-crops',
    #     dest='use_crops',
    #     action='store_false',
    #     help='Use original full images instead of crops')
    # parser.set_defaults(use_crops=True)
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
        '--choice',
        default='cityscapes',
        help='Label conversion type choice: \'cityscapes\' (19 classes)')
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
    """A script for making Apolloscape's Scene Parsing dataset compatible
    with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e.
            data/apolloscape_lane_segmentation)

    Example usage:
        python tools/convert_datasets/apolloscape_lane_seg.py
            path/to/apolloscape/scene_parsing/
    """
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    # Load file lists
    # NOTE: Remember to change "space" to underscore "_"
    road01_ins_train = np.loadtxt(
        os.path.join(root_path, 'road01_ins_train.lst'),
        delimiter='\t',
        dtype=str)
    road01_ins_val = np.loadtxt(
        os.path.join(root_path, 'road01_ins_val.lst'),
        delimiter='\t',
        dtype=str)
    road01_ins_test = np.loadtxt(
        os.path.join(root_path, 'road01_ins_test.lst'),
        delimiter='\t',
        dtype=str)

    road02_ins_train = np.loadtxt(
        os.path.join(root_path, 'road02_ins_train.lst'),
        delimiter='\t',
        dtype=str)
    road02_ins_val = np.loadtxt(
        os.path.join(root_path, 'road02_ins_val.lst'),
        delimiter='\t',
        dtype=str)
    road02_ins_test = np.loadtxt(
        os.path.join(root_path, 'road02_ins_test.lst'),
        delimiter='\t',
        dtype=str)

    road03_ins_train = np.loadtxt(
        os.path.join(root_path, 'road03_ins_train.lst'),
        delimiter='\t',
        dtype=str)
    road03_ins_val = np.loadtxt(
        os.path.join(root_path, 'road03_ins_val.lst'),
        delimiter='\t',
        dtype=str)
    road03_ins_test = np.loadtxt(
        os.path.join(root_path, 'road03_ins_test.lst'),
        delimiter='\t',
        dtype=str)

    train_samples = np.concatenate(
        [road01_ins_train, road02_ins_train, road03_ins_train])
    val_samples = np.concatenate(
        [road01_ins_val, road02_ins_val, road03_ins_val])
    test_samples = np.concatenate(
        [road01_ins_test, road02_ins_test, road03_ins_test])

    all_samples = np.concatenate([train_samples, val_samples, test_samples])

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    # img_filepaths = glob.glob(
    #     osp.join(root_path,
    #              'ColorImage_road*/ColorImage/Record*/Camera_*/*[0-9].jpg'))
    # label_filepaths = glob.glob(
    #     osp.join(root_path,
    #              'Labels_road*/Label/Record*/Camera_*/*[0-9]_bin.png'))

    # if args.crop_gen:
    #     print('Generating cropped images')
    #     if args.nproc > 1:
    #         mmcv.track_parallel_progress(generate_crop, img_filepaths,
    #                                      args.nproc)
    #         mmcv.track_parallel_progress(generate_crop, label_filepaths,
    #                                      args.nproc)
    #     else:
    #         mmcv.track_progress(generate_crop, img_filepaths)
    #         mmcv.track_progress(generate_crop, label_filepaths)

    # if args.use_crops:
    #     print('Use cropped images')
    #     img_filepaths = [
    #         path.replace('.jpg', '_crop.jpg') for path in img_filepaths
    #     ]
    #     label_filepaths = [
    #         path.replace('.png', '_crop.png') for path in label_filepaths
    #     ]

    # Convert segmentation images to 'TrainIds' values
    if args.convert:
        print('Converting segmentation labels')

        ann_filepaths = all_samples[:, 1].tolist()
        ann_filepaths = [
            os.path.join(root_path, path) for path in ann_filepaths
        ]
        ann_filepaths = [path.replace(' ', '_') for path in ann_filepaths]

        seg_choice = args.choice
        if seg_choice == 'cityscapes':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_cityscapes_trainids,
                                             ann_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_cityscapes_trainids, ann_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_directory(out_dir, train_samples, val_samples,
                              test_samples, args.choice, args.symlink)


if __name__ == '__main__':
    main()
