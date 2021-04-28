import numpy as np
from PIL import Image
import argparse
import fnmatch
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt


class LabelClassCounter():
    '''Counts the number of pixel for each class in a semantic segmentation
    dataset using multiprocessing.

    How to use:
    1. Initialize object
        
        class_counter = LabelClassCounter(args.class_count, args.nproc)
    
    2. Count label distribution

        class_counts = class_counter.count_dataset(args.label_root_dir, args.postfix)
        class_counts --> [2.03607194e+09, 3.36037810e+08, ... ]

    The program automatically searches for labels through the directory tree
    according to given postfix (i.e. '_gtFine_labelIds.png')

    NOTE: Implemented as a class for multiprocessing with dynamic class count.
    '''
    def __init__(self, class_count, nproc):
        '''
        Args:
            class_count: Number of classes.
            nproc: Number of parallel threads.
        '''        
        self.class_count = class_count
        self.nproc = nproc
    
    def count_sample(self, label_path):
        '''Returns a pixel count list index by class for the given sample.
        '''
        counts = np.array([0]*self.class_count, dtype=np.double)

        try:
            label = np.array(Image.open(label_path))
        except:
            print(f"Could not read label ({label_path})")
            return counts

        for k in range(self.class_count):
            counts[k] = np.count_nonzero(label == k)

        return counts

    def count_dataset(self, dir_root_path, postfix='.png'):
        '''
        Args:
            label_root_dir: Root of directory tree containing label images.
            postfix: String ending common to all desired files.
                ex: '_gtFine_labelTrainIds.png'
        '''
        if not os.path.isdir(dir_root_path):
            raise IOError(f'Path is not a directory ({dir_root_path})')
        
        # Parse the directory tree for a list of label image paths
        label_paths = self.get_file_paths(dir_root_path, postfix)

        print(f"Found {len(label_paths)} labels")

        with Pool(self.nproc) as p:
            class_counts = p.map(self.count_sample, label_paths)

        # Creates a (#samples, #classes) matrix
        class_counts = np.array(class_counts, dtype=np.double)
        class_counts = np.sum(class_counts, axis=0)

        return class_counts

    @staticmethod
    def get_file_paths(dir_root, postfix):
        '''Parses a directory tree for file paths with the same postfix.
        Args:
            dir_root: Path to directory root.
            postfix: String ending common to all desired files.
                ex: '_gtFine_labelIds.png'
        '''
        file_paths = []

        for root, _, files in os.walk(dir_root):
            for filename in fnmatch.filter(files, '*' + postfix):
                file_paths.append(os.path.join(root, filename))
        
        return file_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count ratio of classes in a set of labels')
    parser.add_argument(
        'label_root_dir', type=str, help='Example cityscapes/gtFine/train')
    parser.add_argument('class_count', type=int, help='Number of classes')
    parser.add_argument(
        '--postfix', default='.png', help='String ending commong to all files')
    parser.add_argument('--nproc', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    class_counter = LabelClassCounter(args.class_count, args.nproc)
    class_counts = class_counter.count_dataset(args.label_root_dir, args.postfix)

    print("Class counts N_i")
    print(class_counts)

    # w_i = 1 / log(N_i)
    # w_i = C * w_i / sum(w)
    # Ref: https://github.com/openseg-group/OCNet.pytorch/issues/14
    w = 1 / np.log(class_counts)
    w = args.class_count * w / np.sum(w)

    print("Class weight coefficients w_i")
    print(w)
