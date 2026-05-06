"""QM9 molecular property prediction dataset as spherical graphs."""

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9 as PyGQM9
from torch_geometric.nn import radius_graph
from typing import List, Tuple, Optional
from tqdm import tqdm
import os
from .base import SphericalGraphDataset


class QM9Spherical(SphericalGraphDataset):
    """
    QM9 dataset where molecules are represented as graphs with atomic positions,
    edges within a radius cutoff, and spherical edge attributes.
    """

    def __init__(
        self,
        root: str = "./data/QM9",
        train: bool = True,
        radius: float = 3.0,
        l_max: int = 4,
        in_features: int = 1,      # atomic number
        out_features: int = 1,      # single target (e.g., dipole moment)
        target_idx: int = 0,        # 0: dipole moment
        transform=None,
    ):
        self.radius = radius
        self.target_idx = target_idx
        super().__init__(
            root=root,
            name="qm9",
            train=train,
            transform=transform,
            l_max=l_max,
            in_features=in_features,
            out_features=out_features,
        )

    def _process(self) -> Tuple[List[Data], List[Data]]:
        # Load full QM9 dataset
        full_dataset = PyGQM9(root=self.raw_dir)
        # Split into train (80%) and test (20%)
        num_total = len(full_dataset)
        num_train = int(0.8 * num_total)
        indices = torch.randperm(num_total)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]

        def process_split(indices) -> List[Data]:
            data_list = []
            for idx in tqdm(indices, desc="Processing QM9"):
                mol = full_dataset[idx]
                pos = mol.pos  # (N, 3)
                z = mol.z.float().unsqueeze(1)  # (N, 1) atomic number as feature
                # Edge graph using radius graph
                edge_index = radius_graph(pos, r=self.radius, loop=False)
                # Edge attributes
                edge_attr = self._compute_edge_attr_from_positions(
                    pos, edge_index)
                # Signals: we don't have precomputed SH coefficients; use zeros
                signals = torch.zeros(pos.size(0), (self.l_max + 1) ** 2)
                # Target: e.g., dipole moment (index 0)
                y = mol.y[:, self.target_idx: self.target_idx + 1]  # (1,)
                data = Data(
                    x=z,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    signals=signals,
                    y=y,
                )
                data_list.append(data)
            return data_list

        train_list = process_split(train_indices)
        test_list = process_split(test_indices)
        return train_list, test_list
