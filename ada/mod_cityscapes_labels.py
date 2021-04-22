import numpy as np
import glob
import os
import argparse
from PIL import Image
import cv2
import copy


# Cityscapes 'id' indices
UNLABELED = 0
EGO_VEHICLE = 1
RECTIFICATION_BORDER = 2
OUT_OF_ROI = 3
STATIC = 4
DYNAMIC = 5
GROUND = 6  # Surface at tram stop etc.
ROAD = 7
SIDEWALK = 8
PARKING = 9
RAIL_TRACK = 10
BUILDING = 11
WALL = 12
FENCE = 13
GUARD_RAIL = 14
BRIDGE = 15
TUNNEL = 16
POLE = 17
POLEGROUP = 18
TRAFFIC_LIGHT = 19
TRAFFIC_SIGN = 20
VEGETATION = 21
TERRAIN = 22
SKY = 23
PERSON = 24
RIDER = 25
CAR = 26
TRUCK = 27
BUS = 28
CARAVAN = 29
TRAILER = 30
TRAIN = 31
MOTORCYCLE = 32
BICYCLE = 33
LICENSE_PLATE = -1
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
    # Switch 'rider' --> 'motorcycle' or 'bicycle' depending on context
    if np.any(label == RIDER):
        label = modify_rider_seg(label)

    new_label = 255*np.ones(label.shape, dtype=np.uint8)

    # Road
    new_label[label == ROAD] = 0
    # Pedestrians
    new_label[label == PERSON] = 1
    # Vehicles
    new_label[label == CAR] = 2
    new_label[label == TRUCK] = 2
    new_label[label == BUS] = 2
    new_label[label == CARAVAN] = 2
    new_label[label == TRAIN] = 2
    # Motorcycles
    new_label[label == MOTORCYCLE] = 3
    # Bicycles
    new_label[label == BICYCLE] = 4
    # Sky
    new_label[label == SKY] = 5
    # Traffic signs
    new_label[label == TRAFFIC_SIGN] = 6
    # Other
    new_label[label == DYNAMIC] = 7
    new_label[label == STATIC] = 7
    new_label[label == GROUND] = 7
    new_label[label == SIDEWALK] = 7
    new_label[label == PARKING] = 7
    new_label[label == RAIL_TRACK] = 7
    new_label[label == BUILDING] = 7
    new_label[label == WALL] = 7
    new_label[label == FENCE] = 7
    new_label[label == GUARD_RAIL] = 7
    new_label[label == BRIDGE] = 7
    new_label[label == TUNNEL] = 7
    new_label[label == POLE] = 7
    new_label[label == POLEGROUP] = 7
    new_label[label == TRAFFIC_LIGHT] = 7
    new_label[label == VEGETATION] = 7
    new_label[label == TERRAIN] = 7

    return new_label


def modify_cityscapes_labels(dataset_path):
    '''Creates and store new labels following mutual Cityscapes-A2D2 semantics.

    NOTE: Remember to modify 'mmseg/datasets/cityscapes.py' to read new labels!
        Old suffix: labelIds
        New suffix: _gtFine_labelTrainIds_mod
    '''
    print(f"Dataset path: {dataset_path}")

    LABEL_SUFFIX = '_gtFine_labelIds'
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
