# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2MarkingsDataset(CustomDataset):
    """A2D2 dataset following the Cityscapes 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentation in 38
    categories. Each pixel in an image is given a label describing the type of
    object it represents, e.g. pedestrian, car, vegetation, etc.

    NOTE: Instance segmentations and some segmentation classes are collapsed to
          follow the Cityscapes 'trainids' label format.
          Ex: 'Car 1' and 'Car 2' --> 'Car'
              'Non-drivable street' --> 'Road'
              'Speed bumper' --> 'Road'

          The segmentation conversion is defined in the following file:
              tools/convert_datasets/a2d2.py

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_labelTrainIds.png' for A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('road', 'markings')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(A2D2MarkingsDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_markingsTrainIds.png',
            **kwargs)
        assert osp.exists(self.img_dir) is not None
