# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import faiss
import numpy as np
import torch

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ClusterHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        super(ClusterHead, self).__init__(**kwargs)

        # Setup k-means cluster estimator
        with open('/home/robin/projects/vissl/kmeans_centroids.pkl',
                  'rb') as f:
            kmeans_centroids = pickle.load(f)
        self.index = faiss.IndexFlatL2(kernel_size)
        self.index.add(kmeans_centroids)

        # Load cluster to class mapping
        with open('/home/robin/projects/vissl/cluster2class_mapping.pkl',
                  'rb') as f:
            self.cluster2class_mapping = pickle.load(f)

    def forward(self, inputs):
        """Forward function."""
        inputs = inputs[0][0]  # --> dim (D, H, W)

        # Transform 'tensor' --> 'row vectors' dim (H*W, D)
        D, H, W = inputs.shape
        inputs = torch.reshape(inputs, (D, -1)).T
        inputs = inputs.cpu().numpy()
        inputs = np.ascontiguousarray(inputs)

        # Replace 'embedding features' --> 'cluster idx'
        _, cluster_idxs = self.index.search(inputs, 1)

        # Transform 'row vectors' --> '2d map' dim (H, W)
        cluster_idx_map = np.reshape(cluster_idxs, (H, W, -1))
        cluster_idx_map = cluster_idx_map[:, :, 0]  # (h, w, 1) --> (h, w)

        N_CLASSES = 27
        output = np.zeros((N_CLASSES, H, W))

        unique_idxs = np.unique(cluster_idx_map)
        for idx in unique_idxs:
            class_idx = self.cluster2class_mapping[idx]
            mask = (cluster_idx_map == idx)
            output[class_idx] = np.logical_or(output[class_idx], mask)

        # for idx in range(27):
        #     output[idx] *= idx
        # output = np.sum(output, axis=0)

        output = torch.tensor(output).unsqueeze(0)

        return output
