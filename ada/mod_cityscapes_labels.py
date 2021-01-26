import numpy as np
import glob
import os
import argparse
from PIL import Image
import cv2
import copy


# Cityscapes '
ROAD = 0
SIDEWALK = 1
BUILDING = 2
WALL = 3
FENCE = 4
POLE = 5
TRAFFIC_LIGHT = 6
TRAFFIC_SIGN = 7
VEGETATION = 8
TERRAIN = 9
SKY = 10
PERSON = 11
RIDER = 12
CAR = 13
TRUCK = 14
BUS = 15
TRAIN = 16
MOTORCYCLE = 17
BICYCLE = 18
IGNORE = 255


def modify_rider_seg(label):
    '''Replaces 'rider' pixels with the ride (i.e. 'motorcycle' or 'bicycle')

    The algorithm does the following:
    1) Instance segmentation of rider pixels.
    2) Generation of a vicinity mask.
    3) Determination of ride, depending on number of 'motorcycle' and 'bicycle'
       pixels in the vicinity mask.
    4) Overwrites the original 'rider' segmentation with the correct 'ride'.

    Ref: https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html
    '''
    # Replace 'rider' pixels in this new label
    new_label = copy.deepcopy(label)

    # 'Rider' instance segmentation
    rider_seg = np.zeros(label.shape, dtype=np.uint8)
    rider_seg[label == RIDER] = 1

    _, instance_seg = cv2.connectedComponents(rider_seg)

    # Max value indicate number of instances
    instance_N = np.max(instance_seg)

    # Overwrite 'rider' instances one-by-one
    kernel = np.ones((3,3),np.uint8)
    for n in range(1,instance_N+1):

        label_tmp = copy.deepcopy(label)

        mask = np.zeros(rider_seg.shape, dtype=np.uint8)
        mask[instance_seg == n] = 1

        mask_dilated = cv2.dilate(mask, kernel, iterations=10)

        label_tmp[mask_dilated == 0] = 0

        # Count number of 'bicycle' pixels within mask
        bicycle_seg = np.zeros(rider_seg.shape, dtype=np.uint8)
        bicycle_seg[label_tmp == BICYCLE] = 1

        bicycle_n = np.sum(bicycle_seg)
        # Count number of 'motorcycle' pixels within mask
        motorcycle_seg = np.zeros(rider_seg.shape, dtype=np.uint8)
        motorcycle_seg[label_tmp == MOTORCYCLE] = 1
        motorcycle_n = np.sum(motorcycle_seg)

        # Overwrite original label with mask as motorcycle
        if motorcycle_n > bicycle_n:
            new_label[mask == 1] = MOTORCYCLE
        elif bicycle_n > motorcycle_n:
            new_label[mask == 1] = BICYCLE
        # Undetermined cases should be ignored to not influence results
        else:
            new_label[mask == 1] = IGNORE

    return new_label


def modify_label(label):
    '''Performs modifications and returns new label.
    Args:
        label: np array of dim (1024, 2048) and integer values as seg classes.
    '''

    # Switch 'wall' --> 'building'
    label[label == WALL] = BUILDING

    # Skip 'pole'
    label[label == POLE] = IGNORE

    # Skip 'traffic_light'
    label[label == TRAFFIC_LIGHT] = IGNORE

    # Switch 'terrain' --> 'vegetation'
    label[label == TERRAIN] = VEGETATION

    # Switch 'bus' --> 'truck'
    label[label == BUS] = TRUCK

    # Skip 'train'
    label[label == TRAIN] = IGNORE

    # Switch 'rider' --> 'motorcycle' or 'bicycle' depending on context
    if np.any(label == RIDER):
        label = modify_rider_seg(label)

    return label


def modify_cityscapes_labels(dataset_path):
    '''Creates and store new labels following mutual Cityscapes-A2D2 semantics.

    NOTE: Remember to modify 'mmseg/datasets/cityscapes.py' to read new labels!
        Old suffix: _gtFine_labelTrainIds
        New suffix: _gtFine_labelTrainIds_mod
    '''
    print(f"Dataset path: {dataset_path}")

    LABEL_SUFFIX = '_gtFine_labelTrainIds'
    NEW_SUFFIX = '_gtFine_labelTrainIds_mod'

    # Read all 'labelTrainIds' files
    label_paths = glob.glob(f'{dataset_path}/gtFine/*/*/*{LABEL_SUFFIX}.png')

    # Generate modified labels one-by-one, and save with new suffix
    label_iter = 1
    tot_labels = len(label_paths)
    for label_path in label_paths:

        print(f"{label_iter} ({label_iter/tot_labels*100.:.0f}%)\r", end="")

        label = np.array(Image.open(label_path))

        label_mod = modify_label(label)

        # Modifies label path and saves modified label as a png file
        label_path_mod = label_path.replace(LABEL_SUFFIX, NEW_SUFFIX)
        Image.fromarray(label_mod).save(label_path_mod)

        label_iter += 1

    print(f"\nModified {label_iter} labels")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    modify_cityscapes_labels(args.dataset_path)
