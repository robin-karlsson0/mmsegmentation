import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os.path as osp
import mmcv
from mmseg.core.evaluation import mean_iou
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor
from ada.eval_datasets import get_cityscapes_eval_samples, get_a2d2_eval_samples
from ada.file_io import write_compressed_pickle


CITYSCAPES_NUM_CLASSES = 19


def init_model(config_filepath, checkpoint_filepath, to_gpu=True):
    '''Initializes a model from a given configuration and checkpoint file.
    
    Exampels:
        config_filepath: deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py
        checkpoint_filepath: deeplabv3plus_resnet_50_40k_cityscapes.pth
    '''
    cfg = mmcv.Config.fromfile(config_filepath)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, checkpoint_filepath, map_location='cpu')
    model.cfg = cfg
    model.eval()
    
    if to_gpu:
        device = torch.device("cuda")
        model.to(device)
    
    return model


def eval_sample(model, img_path, label_path, num_classes, ignore_index=255):
    '''
    Returns:
        miou (float): Mean iou value for all classes in sample.
        iou (list): List of iou values per class for given sample.
    '''

    img = mmcv.imread(img_path)
    label = mmcv.imread(label_path, flag='grayscale')

    output = inference_segmentor(model, img)[0]
    _, _, iou = mean_iou(output, label, num_classes, ignore_index)
    miou = np.nanmean(iou)
    return miou, iou
    

def eval_samples(model, img_paths, label_paths, num_classes, ignore_index=255):
    '''Computes IoU values for a set of (image, label, id) triplets.
    '''
    if len(img_paths) != len(label_paths):
        raise Exception(f'Number of images and labels differ ({len(img_paths)} vs {len(label_paths)})')

    # Dictionary to store result for each sample
    result = {}
    result['miou'] = []
    result['iou'] = []
    result['img_path'] = []

    tot_idx = len(img_paths)

    for idx in range(tot_idx):

        img_path = img_paths[idx]
        label_path = label_paths[idx]

        miou, iou = eval_sample(model, img_path, label_path, num_classes, ignore_index)

        result['miou'].append(miou)
        result['iou'].append(iou)
        result['img_path'].append(img_path)

        print(f'Sample {idx}/{tot_idx} ({idx/tot_idx*100.:.0f}%)\r', end='')

    print(f'Sample {idx+1}/{tot_idx} ({(idx+1)/tot_idx*100.:.0f}%)')
    return result 


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate model sample-by-sample')
    parser.add_argument(
        'eval_path', type=str,
        help='Absolute path to root of evaluation samples')
    parser.add_argument(
        'config_filepath', type=str,
        help='Path to model configuration file')
    parser.add_argument(
        'checkpoint_dir', type=str,
        default='checkpoints',
        help='Path to checkpoint file directory')
    parser.add_argument(
        'checkpoint', type=str,
        help='Checkpoint filename')
    parser.add_argument(
        'dataset', type=str,
        help='Name of dataset to evaluate on (i.e. cityscapes, a2d2)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    eval_path = args.eval_path
    config_filepath = args.config_filepath
    checkpoint_directory = args.checkpoint_dir
    checkpoint_filename = args.checkpoint
    dataset_name = args.dataset

    print(f"Evaluate model\n    {checkpoint_filename}")
    print(f"On dataset\n    {dataset_name}\n")

    checkpoint_filepath = osp.join(checkpoint_directory, checkpoint_filename)

    # Load model
    model = init_model(config_filepath, checkpoint_filepath)

    # Get dataset filepaths
    if dataset_name == 'cityscapes':
        img_paths, label_paths = get_cityscapes_eval_samples(eval_path, dirs=['val'])
    elif dataset_name == 'a2d2':
        img_paths, label_paths = get_a2d2_eval_samples(eval_path, dirs=['train', 'val'])
    else:
        print(f"ERROR: invalid 'dataset_name' given ({dataset_name})")
        exit()

    if len(img_paths) == 0:
        print(f"ERROR: no image paths found (len(img_paths) --> {len(img_paths)})")
        exit()
    if len(label_paths) == 0:
        print(f"ERROR: no label paths found (len(label_paths) --> {len(label_paths)})")
        exit()
    
    # Evaluate
    result = eval_samples(model, img_paths, label_paths, CITYSCAPES_NUM_CLASSES)

    write_compressed_pickle(result, f"eval_{dataset_name}_{checkpoint_filename[:-4]}", ".")
