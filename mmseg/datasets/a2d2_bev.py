import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2DatasetBEV(CustomDataset):
    """The A2D2 dataset following the Cityscapes 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentations having 18
    classes. This dataset configuration merges the original A2D2 classes into a
    subset resembling the Cityscapes classes. The official A2D2 paper presents
    benchmark results in an unspecified but presumptively similar
    Cityscapes-like class taxonomy.

    The 18 class segmentation conversion is defined in the following file:
        tools/convert_datasets/a2d2.py

    Instance segmentations and some segmentation classes are merged to comply
    with the categorical 'trainids' label format.
        Ex: 'Car 1' and 'Car 2' --> 'Car'

    The color palette approximately follows the Cityscapes coloring.

    The following segmentation classes are ignored (i.e. trainIds 255):
    - Ego car:  A calibrated system should a priori know what input
                region corresponds to the ego vehicle.
    - Blurred area: Ambiguous semantic.
    - Rain dirt: Ambiguous semantic.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_BEVLabelTrainIds.png' for the BEV class A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('road', 'marking', 'vehicle', 'pedestrian', 'sky', 'other')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153]]

    def __init__(self, **kwargs):
        super(A2D2DatasetBEV, self).__init__(
            img_suffix='.png', seg_map_suffix='_bevTrainIds.png', **kwargs)
        assert osp.exists(self.img_dir) is not None
