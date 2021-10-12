import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vissl(ckpt):
    """
    NOTE: Only extracts ResNet 50 backbone (for now?)

    Args:
        ckpt (OrderedDict): State dict w. original notations.
    """

    new_ckpt = OrderedDict()

    for key, value in ckpt.items():

        # Only convert backbone layers
        if key.find('backbone') == -1:
            continue

        # Remove prefix
        key = key.replace('_feature_blocks.model.', '')

        # Change notation for the 'stem'
        key = key.replace('layer0.0', 'conv1')
        key = key.replace('layer0.1', 'bn1')

        new_ckpt[key] = value

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in VISSL pretrained Dense SwAV models to \
                     MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    state_dict = checkpoint['classy_state_dict']['base_model']['model'][
        'trunk']

    weight = convert_vissl(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
