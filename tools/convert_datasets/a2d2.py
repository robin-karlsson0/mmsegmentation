import argparse
import glob
import mmcv
import numpy as np
import os.path as osp
from os import symlink
from shutil import copyfile

# Dictionaries specifying which A2D2 segmentation color corresponds to

# A2D2 'trainId' value
#   key: RGB color, value: trainId
#
# The following segmentation classes are ignored (i.e. trainIds 255):
# - Ego car:      A calibrated system should a priori know what input region
#                 corresponds to the ego vehicle.
# - Blurred area: Ambiguous semantic.
# - Rain dirt:    Ambiguous semantic.
SEG_COLOR_DICT_A2D2 = {
    (255, 0, 0): 28,  # Car 1
    (200, 0, 0): 28,  # Car 2
    (150, 0, 0): 28,  # Car 3
    (128, 0, 0): 28,  # Car 4
    (182, 89, 6): 27,  # Bicycle 1
    (150, 50, 4): 27,  # Bicycle 2
    (90, 30, 1): 27,  # Bicycle 3
    (90, 30, 30): 27,  # Bicycle 4
    (204, 153, 255): 26,  # Pedestrian 1
    (189, 73, 155): 26,  # Pedestrian 2
    (239, 89, 191): 26,  # Pedestrian 3
    (255, 128, 0): 30,  # Truck 1
    (200, 128, 0): 30,  # Truck 2
    (150, 128, 0): 30,  # Truck 3
    (0, 255, 0): 32,  # Small vehicles 1
    (0, 200, 0): 32,  # Small vehicles 2
    (0, 150, 0): 32,  # Small vehicles 3
    (0, 128, 255): 19,  # Traffic signal 1
    (30, 28, 158): 19,  # Traffic signal 2
    (60, 28, 100): 19,  # Traffic signal 3
    (0, 255, 255): 20,  # Traffic sign 1
    (30, 220, 220): 20,  # Traffic sign 2
    (60, 157, 199): 20,  # Traffic sign 3
    (255, 255, 0): 29,  # Utility vehicle 1
    (255, 255, 200): 29,  # Utility vehicle 2
    (233, 100, 0): 16,  # Sidebars
    (110, 110, 0): 12,  # Speed bumper
    (128, 128, 0): 14,  # Curbstone
    (255, 193, 37): 6,  # Solid line
    (64, 0, 64): 22,  # Irrelevant signs
    (185, 122, 87): 17,  # Road blocks
    (0, 0, 100): 31,  # Tractor
    (139, 99, 108): 1,  # Non-drivable street
    (210, 50, 115): 8,  # Zebra crossing
    (255, 0, 128): 34,  # Obstacles / trash
    (255, 246, 143): 18,  # Poles
    (150, 0, 150): 2,  # RD restricted area
    (204, 255, 153): 33,  # Animals
    (238, 162, 173): 9,  # Grid structure
    (33, 44, 177): 21,  # Signal corpus
    (180, 50, 180): 3,  # Drivable cobblestone
    (255, 70, 185): 23,  # Electronic traffic
    (238, 233, 191): 4,  # Slow drive area
    (147, 253, 194): 24,  # Nature object
    (150, 150, 200): 5,  # Parking area
    (180, 150, 200): 13,  # Sidewalk
    (72, 209, 204): 255,  # Ego car <-- IGNORED
    (200, 125, 210): 11,  # Painted driv. instr.
    (159, 121, 238): 10,  # Traffic guide obj.
    (128, 0, 255): 7,  # Dashed line
    (255, 0, 255): 0,  # RD normal street
    (135, 206, 255): 25,  # Sky
    (241, 230, 255): 15,  # Buildings
    (96, 69, 143): 255,  # Blurred area <-- IGNORED
    (53, 46, 82): 255,  # Rain dirt <-- IGNORED
}

# The following data directories are used for validation samples
VAL_SPLIT = (
    '20181204_170238',  # Representative 'countryside' data partition
    '20181107_132730',  # Representative 'urban' data partition
)


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    label_filepath = label_filepath.replace('_label_', '_camera_')
    label_filepath = label_filepath.replace('.png', '_labelTrainIds.png')
    return label_filepath


def convert_a2d2_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label using the A2D2 label categories.

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

    seg_colors = list(SEG_COLOR_DICT_A2D2.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_A2D2[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath)
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


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
        root_path: A2D2 dataset root directory (.../camera_lidar_semantic/)
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.

    Raises:
        FileExistError: In case of pre-existing files when trying to create new
                        symbolic links.
    """
    assert split in ['train', 'val', 'test']

    for img_filepath, ann_filepath in zip(img_filepaths, ann_filepaths):
        # Partions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[2]
        ann_filename = ann_filepath.rpartition('/')[2]

        img_link_path = osp.join(root_path, 'img_dir', split, img_filename)
        ann_link_path = osp.join(root_path, 'ann_dir', split, ann_filename)

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


def restructure_a2d2_directory(a2d2_path,
                               val_split,
                               train_on_val=False,
                               use_symlinks=True,
                               label_suffix='_labelTrainIds.png'):
    """Creates a new directory structure and link existing files into it.

    Required to make the A2D2 dataset conform to the mmsegmentation frameworks
    expected dataset structure.

    my_dataset
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
        ... ...

    Args:
        a2d2_path: Absolute path to the A2D2 'camera_lidar_semantic' directory.
        val_split: List of directories used for validation samples.
        train_on_val: Use validation samples as training samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'test'))

    # Lists containing all images and labels to symlinked
    img_filepaths = sorted(glob.glob(osp.join(a2d2_path, '*/camera/*/*.png')))
    ann_filepaths = sorted(
        glob.glob(osp.join(a2d2_path, '*/label/*/*{}'.format(label_suffix))))

    # Split filepaths into 'training' and 'validation'
    if train_on_val:
        train_img_paths = img_filepaths
        train_ann_paths = ann_filepaths
    else:
        # Create new lists that skips validation directories
        # NOTE: '/.../' is added to so substrings only match with directories
        train_img_paths = [
            x for x in img_filepaths
            if not any(f'/{y}/' in x for y in val_split)
        ]
        train_ann_paths = [
            x for x in ann_filepaths
            if not any(f'/{y}/' in x for y in val_split)
        ]
    # Create new lists that includes only validation directories
    # NOTE: '/.../' is added to so substrings only match with directories
    val_img_paths = [
        x for x in img_filepaths if any(f'/{y}/' in x for y in val_split)
    ]
    val_ann_paths = [
        x for x in ann_filepaths if any(f'/{y}/' in x for y in val_split)
    ]

    create_split_dir(
        train_img_paths,
        train_ann_paths,
        'train',
        a2d2_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        val_img_paths,
        val_ann_paths,
        'val',
        a2d2_path,
        use_symlinks=use_symlinks)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to TrainIds')
    parser.add_argument(
        'a2d2_path',
        help='A2D2 segmentation data absolute path\
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
        '--train-on-val',
        dest='train_on_val',
        action='store_true',
        help='Use validation samples as training samples')
    parser.set_defaults(train_on_val=False)
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
    """Program for making Audi's A2D2 dataset compatible with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/a2d2)

    Segmentation label conversion:
        The A2D2 labels are instance segmentations (i.e. car_1, car_2, ...),
        while semantic segmentation requires categorical segmentations.

        The function 'convert_TYPE_trainids()' converts all instance
        segmentation to their corresponding categorical segmentation and saves
        them as new label image files..

    NOTE: The following segmentation classes are ignored (i.e. trainIds 255):
          - Ego car:  A calibrated system should a priori know what input
                      region corresponds to the ego vehicle.
          - Blurred area: Ambiguous semantic.
          - Rain dirt: Ambiguous semantic.

    Directory restructuring:
        A2D2 files are not arranged in the required 'train/val/test' directory
        structure.

        The function 'restructure_a2d2_directory' creates a new compatible
        directory structure in the root directory, and fills it with symbolic
        links or file copies to the input and segmentation label images.

    Example usage:
        python tools/convert_datasets/a2d2.py path/to/camera_lidar_semantic
    """
    args = parse_args()
    a2d2_path = args.a2d2_path
    out_dir = args.out_dir if args.out_dir else a2d2_path
    mmcv.mkdir_or_exist(out_dir)

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    label_filepaths = glob.glob(osp.join(a2d2_path, '*/label/*/*[0-9].png'))

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:
        if args.nproc > 1:
            mmcv.track_parallel_progress(convert_a2d2_trainids,
                                         label_filepaths, args.nproc)
        else:
            mmcv.track_progress(convert_a2d2_trainids, label_filepaths)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_a2d2_directory(out_dir, VAL_SPLIT, args.train_on_val,
                                   args.symlink)


if __name__ == '__main__':
    main()
