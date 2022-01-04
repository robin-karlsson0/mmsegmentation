import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ApolloscapeLaneSegDatasetBinary(CustomDataset):
    """
    """

    CLASSES = ('Not-marking', 'Marking')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(ApolloscapeLaneSegDatasetBinary, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_bin_BinaryTrainIds.png', **kwargs)
        assert osp.exists(self.img_dir) is not None
