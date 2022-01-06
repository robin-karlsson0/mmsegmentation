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
LABEL_SUFFIX_BINARY = '_BinaryTrainIds.png'

# Dictionaries specifying which Apolloscape segmentation color corresponds to
# which 'trainId' value
#     Key: RGB color --> Value: trainId integer
# Ref: https://github.com/ApolloScapeAuto/dataset-api/blob/master/
#          lane_segmentation/helpers/laneMarkDetection.py

# Only color non-markings black for speedup
SEG_COLOR_DICT_BINARY = {
    (0, 0, 0): 0,  # Void
    (0, 153, 153): 255,  # Noise <-- Ignored
    (255, 255, 255): 0,  # Ignored <-- Void (ego-car)
}
#    (70, 130, 180): 1,
#    (220, 20, 60): 1,
#    (128, 0, 128): 1,
#    (255, 0, 0): 1,
#    (0, 0, 60): 1,
#    (0, 60, 100): 1,
#    (0, 0, 142): 1,
#    (119, 11, 32): 1,
#    (244, 35, 232): 1,
#    (0, 0, 160): 1,
#    (153, 153, 153): 1,
#    (220, 220, 0): 1,
#    (250, 170, 30): 1,
#    (102, 102, 156): 1,
#    (128, 0, 0): 1,
#    (128, 64, 128): 1,
#    (238, 232, 170): 1,
#    (190, 153, 153): 1,
#    (0, 0, 230): 1,
#    (128, 128, 0): 1,
#    (128, 78, 160): 1,
#    (150, 100, 100): 1,
#    (255, 165, 0): 1,
#    (180, 165, 180): 1,
#    (107, 142, 35): 1,
#    (201, 255, 229): 1,
#    (0, 191, 255): 1,
#    (51, 255, 51): 1,
#    (250, 128, 114): 1,
#    (127, 255, 0): 1,
#    (255, 128, 0): 1,
#    (0, 255, 255): 1,
#    (178, 132, 190): 1,
#    (128, 128, 64): 1,
#    (102, 0, 204): 1,


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if 'TrainIds.png' in label_filepath:
        return label_filepath

    # label_filepath = label_filepath.replace('_label_', '_camera_')
    if label_choice == 'binary':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX_BINARY)
    else:
        raise ValueError
    return label_filepath


def convert_binary_trainids(label_filepath, ignore_id=1, overwrite=True):
    """Saves a new semantic label replacing RGB values with label categories.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    # Save new 'trainids' semantic label
    new_label_filepath = modify_label_filename(label_filepath, 'binary')

    # Skip existing files
    if osp.isfile(new_label_filepath) and not overwrite:
        return

    # Read label file as Numpy array (H, W, 3)
    try:
        orig_label = mmcv.imread(label_filepath, channel_order='rgb')
    except Exception as e:
        print(f"{e}\nFailed to read file \'{label_filepath}\' --> Skip\n")
        return

    # Empty array with all elements set as the default label
    H, W, _ = orig_label.shape
    mod_label = ignore_id * np.ones((H, W), dtype=int)

    seg_colors = list(SEG_COLOR_DICT_BINARY.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_BINARY[seg_color]

    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, new_label_filepath)


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
                          val_ratio,
                          test_ratio,
                          label_choice,
                          train_on_val_and_test=False,
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

    Samples are randomly split into a 'train', 'validation', and 'test' split
    according to the argument sample ratios.

    Args:
        root_path: Absolute path to the Apolloscape 'lane_segmentation'
                   directory.
        val_ratio: Float value representing ratio of validation samples.
        test_ratio: Float value representing ratio of test samples.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """
    for r in [val_ratio, test_ratio]:
        assert r >= 0. and r < 1., 'Invalid ratio {}'.format(r)

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(root_path, 'images'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'images', 'train'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'images', 'val'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'images', 'test'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'annotations', 'train'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'annotations', 'val'))
    mmcv.mkdir_or_exist(osp.join(root_path, 'annotations', 'test'))

    # Lists containing all images and labels to symlinked

    img_filepaths = sorted(
        glob.glob(
            osp.join(root_path,
                     'ColorImage_road*/ColorImage/Record*/Camera_*/*.jpg')))

    if label_choice == 'binary':
        label_suffix = LABEL_SUFFIX_BINARY
    else:
        raise ValueError
    ann_filepaths = sorted(
        glob.glob(
            osp.join(root_path,
                     '*/Label/Record*/Camera_*/*{}'.format(label_suffix))))

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
        root_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        val_img_paths,
        val_ann_paths,
        'val',
        root_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        test_img_paths,
        test_ann_paths,
        'test',
        root_path,
        use_symlinks=use_symlinks)


def generate_crop(imgpath, H=2710, W=3384, H_cropped=1010):
    """Generate cropped versions of images containing the bottom part."""
    bbox = np.array([0, H - H_cropped, W, H])
    img = mmcv.imread(imgpath, channel_order='bgr')
    img = mmcv.imcrop(img, bbox)
    # Create cropped image path
    file_extension = osp.splitext(imgpath)[-1]
    if file_extension == '.jpg':
        crop_imgpath = imgpath.replace('.jpg', '_crop.jpg')
    elif file_extension == '.png':
        crop_imgpath = imgpath.replace('.png', '_crop.png')
    else:
        raise ValueError

    mmcv.imwrite(img, crop_imgpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Apolloscape Lane Segmentation annotations to \
                     TrainIds')
    parser.add_argument(
        'root_path',
        help='Apolloscape Lane Segmentation directory absolute path \
              (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-crop-gen',
        dest='crop_gen',
        action='store_false',
        help='Skips generating cropped versions of images and labels')
    parser.set_defaults(crop_gen=True)
    parser.add_argument(
        '--no-crops',
        dest='use_crops',
        action='store_false',
        help='Use original full images instead of crops')
    parser.set_defaults(use_crops=True)
    parser.add_argument(
        '--no-convert',
        dest='convert',
        action='store_false',
        help='Skips converting label images')
    parser.set_defaults(convert=True)
    parser.add_argument(
        '--no-overwrite',
        dest='overwrite',
        action='store_false',
        help='Do not overwrite existing modified label files')
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        '--no-restruct',
        dest='restruct',
        action='store_false',
        help='Skips restructuring directory structure')
    parser.set_defaults(restruct=True)
    parser.add_argument(
        '--choice',
        default='binary',
        help='Label conversion type choice: \'binary\' (2 classes)')
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
    """A script for making Apolloscape's Lane Segmentation dataset compatible
    with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e.
            data/apolloscape_lane_segmentation)

    Directory restructuring:
        A2D2 files are not arranged in the required 'train/val/test' directory
        structure.

        The function 'restructure_a2d2_directory' creates a new compatible
        directory structure in the root directory, The optional argument
        `--no-symlink` creates copies of the label images instead of symbolic
        links.

    Example usage:
        python tools/convert_datasets/apolloscape_lane_seg.py
            path/to/camera_lidar_semantic
    """
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    img_filepaths = glob.glob(
        osp.join(root_path,
                 'ColorImage_road*/ColorImage/Record*/Camera_*/*[0-9].jpg'))
    label_filepaths = glob.glob(
        osp.join(root_path,
                 'Labels_road*/Label/Record*/Camera_*/*[0-9]_bin.png'))

    if args.crop_gen:
        print('Generating cropped images')
        if args.nproc > 1:
            mmcv.track_parallel_progress(generate_crop, img_filepaths,
                                         args.nproc)
            mmcv.track_parallel_progress(generate_crop, label_filepaths,
                                         args.nproc)
        else:
            mmcv.track_progress(generate_crop, img_filepaths)
            mmcv.track_progress(generate_crop, label_filepaths)

    if args.use_crops:
        print('Use cropped images')
        img_filepaths = [
            path.replace('.jpg', '_crop.jpg') for path in img_filepaths
        ]
        label_filepaths = [
            path.replace('.png', '_crop.png') for path in label_filepaths
        ]

    # Convert segmentation images to 'TrainIds' values
    if args.convert:
        print('Converting segmentation labels')
        seg_choice = args.choice
        if seg_choice == 'binary':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_binary_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_binary_trainids, label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_directory(out_dir, args.val, args.test, args.choice,
                              args.train_on_val_and_test, args.symlink)


if __name__ == '__main__':
    main()

# Add instructions to replace 'space' --> '_' in directory names
