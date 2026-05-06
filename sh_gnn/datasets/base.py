"""Base spherical graph dataset."""

import torch
from torch_geometric.data import InMemoryDataset, Data
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
import os


class SphericalGraphDataset(InMemoryDataset, ABC):
    def __init__(self, root: str, name: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 l_max: int = 10, in_features: int = 3, out_features: int = 1):
        self.name = name
        self.train = train
        self.l_max = l_max
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.name}_train.pt", f"{self.name}_test.pt"]

    def download(self):
        pass

    def process(self):
        train_list, test_list = self._process()
        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])

    @abstractmethod
    def _process(self) -> Tuple[List[Data], List[Data]]:
        pass

    def _compute_edge_attr_from_positions(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        rel = pos[col] - pos[row]
        dist = torch.norm(rel, dim=1, keepdim=True)
        rel_norm = rel / (dist + 1e-8)
        theta = torch.acos(torch.clamp(rel_norm[:, 2], -1.0, 1.0))
        phi = torch.atan2(rel_norm[:, 1], rel_norm[:, 0]) + torch.pi
        phi = phi % (2 * torch.pi)
        return torch.cat([dist, theta.unsqueeze(1), phi.unsqueeze(1)], dim=1)
