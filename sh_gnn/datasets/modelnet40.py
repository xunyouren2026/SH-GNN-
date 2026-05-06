"""ModelNet40 point cloud classification dataset projected onto sphere."""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale
from torch_geometric.nn import knn_graph
from typing import List, Tuple, Optional
import os
from .base import SphericalGraphDataset


class ModelNet40Spherical(SphericalGraphDataset):
    """
    ModelNet40 dataset where each point cloud is projected onto the unit sphere,
    and a graph is built using k-nearest neighbors in spherical coordinates.
    """

    def __init__(
        self,
        root: str = "./data/ModelNet40",
        train: bool = True,
        num_points: int = 1024,
        k: int = 20,
        l_max: int = 10,
        in_features: int = 3,
        out_features: int = 40,
        transform=None,
    ):
        self.num_points = num_points
        self.k = k
        super().__init__(
            root=root,
            name="modelnet40",
            train=train,
            transform=transform,
            l_max=l_max,
            in_features=in_features,
            out_features=out_features,
        )

    def _process(self) -> Tuple[List[Data], List[Data]]:
        # Use PyG's ModelNet loader to get raw point clouds
        # Note: ModelNet does not have separate train/test splits? Actually it does.
        # We'll load both train and test sets.
        train_dataset = ModelNet(
            root=self.raw_dir,
            name="40",
            train=True,
            transform=SamplePoints(self.num_points),
            pre_transform=NormalizeScale(),
        )
        test_dataset = ModelNet(
            root=self.raw_dir,
            name="40",
            train=False,
            transform=SamplePoints(self.num_points),
            pre_transform=NormalizeScale(),
        )

        def process_split(dataset) -> List[Data]:
            data_list = []
            for obj in dataset:
                pos = obj.pos  # (N, 3) already normalized
                # Project to unit sphere (should already be on sphere after NormalizeScale)
                # but ensure radius = 1
                pos = pos / (torch.norm(pos, dim=1, keepdim=True) + 1e-8)

                # Build kNN graph in Euclidean (spherical) space
                edge_index = knn_graph(pos, k=self.k, loop=False)

                # Compute edge attributes (distance, theta, phi)
                edge_attr = self._compute_edge_attr_from_positions(
                    pos, edge_index)

                # Node features: positions (x,y,z) as features
                x = pos

                # Spherical harmonic coefficients as signals: we don't have them precomputed,
                # so we use zeros or compute from data? For dynamic sparsity we need signals.
                # We'll compute approximate SH coefficients from the point cloud via quadrature.
                # For ModelNet, we can skip signals and rely on fixed L_eff = l_max.
                signals = torch.zeros(x.size(0), (self.l_max + 1) ** 2)

                # Labels: class index
                y = obj.y  # (1,)

                # For classification, we want graph-level output, but our model outputs node-level.
                # We'll take mean pooling. So y is repeated for each node? Better to set y as scalar
                # but Data expects y per node if multiple nodes. We'll store scalar in data.y.
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    signals=signals,
                    y=y,
                )
                data_list.append(data)
            return data_list

        train_list = process_split(train_dataset)
        test_list = process_split(test_dataset)
        return train_list, test_list

    def _download(self):
        # ModelNet will be downloaded automatically by ModelNet class
        pass
