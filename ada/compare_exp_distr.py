import numpy as np
import matplotlib.pyplot as plt
import argparse
from ada.file_io import read_compressed_pickle
import os


def get_miou(result):
    return result["miou"]


def miou_histogram(result_dict1, result_dict2, output_dir, bins=20, range=(0,1)):
    
    mious1 = get_miou(result_dict1)
    mious2 = get_miou(result_dict2)

    plt.hist(mious1, bins=bins, range=range, alpha=0.5, label='a')
    plt.hist(mious2, bins=bins, range=range, alpha=0.5, label='b')

    plt.xlabel("mIoU")
    plt.ylabel("#samples")
    plt.legend()

    filepath = os.path.join(output_dir, f"compare_miou.png")
    plt.savefig(filepath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare results of two evaluations')
    parser.add_argument(
        'result1_path', type=str, help='Path to first result file')
    parser.add_argument(
        'result2_path', type=str, help='Path to second result file')
    parser.add_argument(
        'output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    result1_path = args.result1_path
    result2_path = args.result2_path
    output_dir = args.output_dir

    if os.path.isfile(result1_path) is False:
        raise FileNotFoundError(f"Result file 1 not found: {result1_path}")
    if os.path.isfile(result2_path) is False:
        raise FileNotFoundError(f"Result file 2 not found: {result2_path}")
    if os.path.isdir(output_dir) is False:
        raise FileNotFoundError(f"Output directory invalid: {output_dir}")

    result_dict1 = read_compressed_pickle(args.result1_path)
    result_dict2 = read_compressed_pickle(args.result2_path)

    miou_histogram(result_dict1, result_dict2, output_dir)




