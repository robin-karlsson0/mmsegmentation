import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ApolloscapeLanesegDatasetBEV(CustomDataset):
    """"""
    CLASSES = ('Background', 'Road markings')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(ApolloscapeLanesegDatasetBEV, self).__init__(
            img_suffix='_crop.jpg',
            seg_map_suffix='_bin_crop_BinaryTrainIds.png',
            **kwargs)
        assert osp.exists(self.img_dir) is not None
