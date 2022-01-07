import argparse
import os

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recursively checks for broken images within a directory')
    parser.add_argument(
        'dir_path', type=str, help='Root of directory structure')
    parser.add_argument('--valid-exts', nargs='+', type=str, default=[])
    args = parser.parse_args()
    return args


def recursively_find_files(dir_path, valid_extensions):
    """Returns a list of valid file paths within a directory tree."""
    filepaths = []
    for path, subdirs, filenames in os.walk(dir_path):
        for filename in filenames:
            extension = filename.split('.')[-1]
            if extension in valid_extensions:
                filepath = os.path.join(path, filename)
                filepaths.append(filepath)
    return filepaths


def check_img(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        print(f'Invalid image: {img_path}')
        print(e)


def check_imgs(img_paths, print_interval=1000):
    N = len(img_paths)
    for idx, img_path in enumerate(img_paths):
        if idx % print_interval == 0:
            print(f'{idx}/{N} ({idx/N*100:.0f} %)', end='\r')
        check_img(img_path)


if __name__ == '__main__':

    args = parse_args()
    dir_path = args.dir_path
    valid_exts = args.valid_exts

    filepaths = recursively_find_files(dir_path, valid_exts)

    check_imgs(filepaths)
