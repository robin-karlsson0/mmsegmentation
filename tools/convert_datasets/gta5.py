import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX = '_trainIds.png'

# Cityscapes-like 'trainId' value
#   key: RGB color, value: trainId (from Cityscapes)
SEG_COLOR_DICT_CITYSCAPES = {
    (0, 0, 0): 255,  # Ignore
    (111, 74, 0): 255,  # Dynamic --> Ignore
    (81, 0, 81): 255,  # Ground --> Ignore
    (128, 64, 128): 0,  # Road
    (244, 35, 232): 1,  # Sidewalk
    (250, 170, 160): 255,  # Parking --> Ignore
    (230, 150, 140): 255,  # Rail track --> Ignore
    (70, 70, 70): 2,  # Building
    (102, 102, 156): 3,  # Wall
    (190, 153, 153): 4,  # Fence
    (180, 165, 180): 255,  # Guard rail
    (150, 100, 100): 255,  # Bridge
    (150, 120, 90): 255,  # Tunnel
    (153, 153, 153): 5,  # Pole
    (250, 170, 30): 6,  # Traffic light
    (220, 220, 0): 7,  # Traffic sign
    (107, 142, 35): 8,  # Vegetation
    (152, 251, 152): 9,  # Terrain
    (70, 130, 180): 10,  # Sky
    (220, 20, 60): 11,  # Person
    (255, 0, 0): 12,  # Rider
    (0, 0, 142): 13,  # Car
    (0, 0, 70): 14,  # Truck
    (0, 60, 100): 15,  # Bus
    (0, 0, 90): 255,  # Caravan
    (0, 0, 110): 255,  # Trailer
    (0, 80, 100): 16,  # Train
    (0, 0, 230): 17,  # Motorcycle (license plate --> Car)
    (119, 11, 32): 18,  # Bicycle
}


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

    seg_colors = list(SEG_COLOR_DICT_CITYSCAPES.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_CITYSCAPES[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, 'cityscapes')
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
        root_path: GTA 5 dataset root directory (.../gta5/)
        use_symlinks: Symbolically link existing files in the original GTA 5
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


def restructure_gta5_directory(gta5_path,
                               val_ratio,
                               test_ratio,
                               label_choice,
                               train_on_val_and_test=False,
                               use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the GTA 5 dataset conform to the mmsegmentation frameworks
    expected dataset structure. my_dataset.

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
    Samples are randomly split into a 'train', 'validation', and 'test' split
    according to the argument sample ratios.
    Args:
        gta5_path: Absolute path to the GTA 5 'gta5/' directory.
        val_ratio: Float value representing ratio of validation samples.
        test_ratio: Float value representing ratio of test samples.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    for r in [val_ratio, test_ratio]:
        assert r >= 0. and r < 1., 'Invalid ratio {}'.format(r)

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'images'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'images', 'train'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'images', 'val'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'images', 'test'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'annotations', 'train'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'annotations', 'val'))
    mmcv.mkdir_or_exist(osp.join(gta5_path, 'annotations', 'test'))

    # Lists containing all images and labels to symlinked
    img_filepaths = sorted(glob.glob(osp.join(gta5_path, 'images/*.png')))

    if label_choice == 'cityscapes':
        label_suffix = LABEL_SUFFIX
    else:
        raise ValueError
    ann_filepaths = sorted(
        glob.glob(osp.join(gta5_path, 'labels/*{}'.format(label_suffix))))

    # Randomize order of (image, label) pairs
    pairs = list(zip(img_filepaths, ann_filepaths))
    random.shuffle(pairs)
    img_filepaths, ann_filepaths = zip(*pairs)

    # Split data according to given ratios
    total_samples = len(img_filepaths)
    train_ratio = 1.0 - val_ratio - test_ratio

    train_idx_end = int(np.ceil(train_ratio * (total_samples - 1)))
    val_idx_end = train_idx_end + int(np.ceil(val_ratio * total_samples))

    # Train split
    if train_on_val_and_test:
        train_img_paths = img_filepaths
        train_ann_paths = ann_filepaths
    else:
        train_img_paths = img_filepaths[:train_idx_end]
        train_ann_paths = ann_filepaths[:train_idx_end]
    # Val split
    val_img_paths = img_filepaths[train_idx_end:val_idx_end]
    val_ann_paths = ann_filepaths[train_idx_end:val_idx_end]
    # Test split
    test_img_paths = img_filepaths[val_idx_end:]
    test_ann_paths = ann_filepaths[val_idx_end:]

    create_split_dir(
        train_img_paths,
        train_ann_paths,
        'train',
        gta5_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        val_img_paths,
        val_ann_paths,
        'val',
        gta5_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        test_img_paths,
        test_ann_paths,
        'test',
        gta5_path,
        use_symlinks=use_symlinks)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA 5 annotations to trainIds')
    parser.add_argument(
        'gta5_path',
        help='GTA 5 segmentation data absolute path\
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
        '--choice',
        default='cityscapes',
        help='Label conversion type choice: \'cityscapes\' (18 classes)')
    parser.add_argument(
        '--val', default=0.103, type=float, help='Validation set sample ratio')
    parser.add_argument(
        '--test', default=0.197, type=float, help='Test set sample ratio')
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
    """A script for making the GTA 5 dataset compatible with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/gta5)

    Segmentation label conversion:

        The function 'convert_TYPE_trainids()' converts all RGB segmentation to
        their corresponding categorical segmentation and saves them as new
        label image files.

        Label choice 'cityscapes' (default) results in labels with 18 classes
        with the filename suffix '_trainIds.png'.

    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is usefull in the synthetic-to-real domain adaptation
    experiment setting.

    Add `--nproc N` for multiprocessing using N threads.

    Directory restructuring:
        GTA 5 files are not arranged in the required 'train/val/test' directory
        structure.

        The function 'restructure_gta5_directory' creates a new compatible
        directory structure in the root directory, The optional argument
        `--no-symlink` creates copies of the label images instead of symbolic
        links.

    Example usage:
        python tools/convert_datasets/gta.py path/to/gta5
    """
    args = parse_args()
    gta5_path = args.gta5_path
    out_dir = args.out_dir if args.out_dir else gta5_path
    mmcv.mkdir_or_exist(out_dir)

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    label_filepaths = glob.glob(osp.join(gta5_path, 'labels/*[0-9].png'))

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:
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
        restructure_gta5_directory(out_dir, args.val, args.test, args.choice,
                                   args.train_on_val_and_test, args.symlink)


if __name__ == '__main__':
    main()
