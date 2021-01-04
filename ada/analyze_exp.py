import numpy as np
import matplotlib.pyplot as plt
import argparse
from ada.file_io import read_compressed_pickle
import os.path as osp


def get_miou(result):
    return result["miou"]


def get_img_path(result):
    return result["img_path"]


def miou_histogram(result, exp_name, output_dir, bins=20, range=(0,1)):
    mious = get_miou(result)

    plt.hist(mious, bins=bins, range=range)

    plt.xlabel("mIoU")
    plt.ylabel("#samples")

    filepath = osp.join(output_dir, f"miou_{exp_name}.png")
    plt.savefig(filepath)
    

def miou_list(result, exp_name, output_dir):

    mious = get_miou(result)
    img_paths = get_img_path(result)

    if len(mious) != len(img_paths):
        raise Exception(f'Number of "mIoU entries" and "image paths" differ ({len(mious)} vs {len(img_paths)})')

    for i in range(10):
        print(mious[i])

    for i in range(10):
        print(img_paths[i])

    img_paths = [x for _, x in sorted(zip(mious,img_paths))]

    mious.sort()

    text = ""
    for i in range(len(mious)):
        text += f"{mious[i]}, {img_paths[i]}\n"

    # Write to file
    filepath = osp.join(output_dir, f"miou_{exp_name}.csv")
    f = open(filepath, "w")
    f.write(text)
    f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Display result of evaluation')
    parser.add_argument(
        'result', type=str,
        help='Path to result file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    exp_name = args.result[:-4]

    result = read_compressed_pickle(args.result)

    # Perform analyzis
    miou_histogram(result, exp_name, ".")
    miou_list(result, exp_name, ".")
