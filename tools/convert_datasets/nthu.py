import argparse
import glob
import os.path as osp
import random
from os import symlink

import mmcv

random.seed(14)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Organize NTHU dataset to MMSegmentation format')
    parser.add_argument(
        'nthu_path',
        help='NTHU dataset absolute path (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-symlink',
        dest='symlink',
        action='store_false',
        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """A script for making the NTHU dataset compatible with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/nthu)

    Original directory structure:

    NTHU_Dataset/
        Rio/
            Images/
                Train/
                    <Images without labels>
                Test/
                    pano_00002_2_180.png
                    ...
            Labels/
                Test/
                    pano_00002_2_180_bgr.png
                    pano_00002_2_180_city.png <-- Use
                    pano_00002_2_180_eval.png
                    pano_00002_2_180_visual.png
                    ...
        Rome/
            ...
        Taipei/
            ...
        Tokyo/
            ...

    Converted directory structure:

    NTHU_Dataset/
        img_dir/
            train/
                <Empty>
            val/
                <Empty>
            test/
                rio_pano_00002_2_180.png  <Added city>
                ...
        ann_dir/
            train/
                <Empty>
            val/
                <Empty>
            test/
                rio_pano_00002_2_180_labelTrainIds.png  <Added label suffix>
                ...

    Example usage:
        python tools/convert_datasets/nthu.py path/to/nthu
    """
    args = parse_args()
    nthu_path = args.nthu_path
    out_dir = args.out_dir if args.out_dir else nthu_path
    mmcv.mkdir_or_exist(out_dir)

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    # Symbolically link images to new directory structure
    cities = ['Rio', 'Rome', 'Taipei', 'Tokyo']
    for city in cities:
        imgs = glob.glob(f'{nthu_path}/{city}/Images/Test/*.png')
        anns = glob.glob(f'{nthu_path}/{city}/Labels/Test/*_city.png')

        for idx in range(len(imgs)):
            # Image
            old_img_path = imgs[idx]
            img_filename = old_img_path.split('/')[-1]
            new_img_path = f'{out_dir}/img_dir/test/{city}_{img_filename}'
            symlink(old_img_path, new_img_path)
            # Annotation w. new suffix
            old_ann_path = anns[idx]
            ann_filename = old_ann_path.split('/')[-1]
            ann_filename = ann_filename.replace('_city.png',
                                                '_labelTrainIds.png')
            new_ann_path = f'{out_dir}/ann_dir/test/{city}_{ann_filename}'
            symlink(old_ann_path, new_ann_path)


if __name__ == '__main__':
    main()
