import numpy as np
import matplotlib.pyplot as plt
import argparse
from ada.file_io import read_compressed_pickle
import os.path as osp


def get_miou(result):
    return result["miou"]


def get_img_path(result):
    return result["img_path"]


def miou_histogram(result_dict, output_dir, bins=20, range=(0,1)):
    mious = get_miou(result_dict)

    plt.hist(mious, bins=bins, range=range)

    plt.xlabel("mIoU")
    plt.ylabel("#samples")

    filepath = osp.join(output_dir, f"sample_miou.png")
    plt.savefig(filepath)
    

def miou_list(result_dict, output_dir):

    mious = get_miou(result_dict)
    img_paths = get_img_path(result_dict)

    if len(mious) != len(img_paths):
        raise Exception(f'Number of "mIoU entries" and "image paths" differ ({len(mious)} vs {len(img_paths)})')

    img_paths = [x for _, x in sorted(zip(mious,img_paths))]

    mious.sort()

    text = ""
    for i in range(len(mious)):
        text += f"{mious[i]}, {img_paths[i]}\n"

    # Write to file
    filepath = osp.join(output_dir, f"sample_miou.csv")
    f = open(filepath, "w")
    f.write(text)
    f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Display result of evaluation')
    parser.add_argument(
        'result_path', type=str,
        help='Path to result file')
    parser.add_argument(
        'output_dir', type=str,
        help='Path to output dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    result_path = args.result_path
    output_dir = args.output_dir

    result_dict = read_compressed_pickle(result_path)

    # Perform analyzis
    miou_histogram(result_dict, output_dir)
    miou_list(result_dict, output_dir)
