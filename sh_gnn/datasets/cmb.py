"""CMB spherical dataset (optional, requires healpy)."""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import os
from .base import SphericalGraphDataset

# 条件导入 healpy
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False
    print("Warning: healpy not installed. CMB dataset will be unavailable.")


class CMBSpherical(SphericalGraphDataset):
    """CMB dataset (requires healpy)."""

    def __init__(self, root: str = "./data/CMB", train: bool = True,
                 nside: int = 64, l_max: int = 128, use_smica: bool = True,
                 transform=None):
        if not HEALPY_AVAILABLE:
            raise ImportError(
                "healpy not installed. Install: pip install healpy")
        self.nside = nside
        self.use_smica = use_smica
        npix = hp.nside2npix(nside)
        self.npix = npix
        super().__init__(
            root=root, name=f"cmb_nside{nside}", train=train,
            transform=transform, l_max=l_max, in_features=3, out_features=1
        )

    @property
    def raw_file_names(self) -> List[str]:
        if self.use_smica:
            return ["SMICA_R3.00_full.fits"]
        else:
            return ["simulated_cmb_map.fits"]

    def download(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"Generating dummy CMB map at {raw_path}")
            npix = hp.nside2npix(self.nside)
            dummy = np.random.randn(npix)
            hp.write_map(raw_path, dummy)

    def _process(self) -> Tuple[List[Data], List[Data]]:
        map_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        cmb_map = hp.read_map(map_path, verbose=False)
        current_nside = hp.npix2nside(len(cmb_map))
        if current_nside != self.nside:
            cmb_map = hp.ud_grade(cmb_map, nside_out=self.nside)
        npix = len(cmb_map)

        theta, phi = hp.pix2ang(self.nside, np.arange(npix))
        x_cart = np.sin(theta) * np.cos(phi)
        y_cart = np.sin(theta) * np.sin(phi)
        z_cart = np.cos(theta)
        pos_cart = np.stack([x_cart, y_cart, z_cart], axis=1)

        theta_t = torch.tensor(theta, dtype=torch.float32).unsqueeze(1)
        phi_t = torch.tensor(phi, dtype=torch.float32).unsqueeze(1)
        temp_t = torch.tensor(cmb_map, dtype=torch.float32).unsqueeze(1)
        node_features = torch.cat([theta_t, phi_t, temp_t], dim=1)

        neighbors = hp.get_all_neighbours(self.nside, np.arange(npix))
        edge_list = []
        for i in range(npix):
            for nb in neighbors[:, i]:
                if nb >= 0 and nb != i:
                    edge_list.append([i, nb])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        pos_cart_t = torch.tensor(pos_cart, dtype=torch.float32)
        edge_attr = self._compute_edge_attr_from_positions(
            pos_cart_t, edge_index)

        alm = hp.map2alm(cmb_map, lmax=self.l_max)
        num_sh = (self.l_max + 1) ** 2
        alms_real = np.zeros(num_sh, dtype=np.float32)
        for l in range(self.l_max + 1):
            for m in range(0, l + 1):
                idx = l * l + l + m
                if m == 0:
                    val = alm[l, 0].real
                else:
                    val_pos = np.sqrt(2.0) * alm[l, m].real
                    val_neg = np.sqrt(2.0) * alm[l, m].imag
                    alms_real[l * l + l - m] = val_neg
                alms_real[idx] = val_pos
        signals = torch.tensor(
            alms_real, dtype=torch.float32).unsqueeze(0).expand(npix, -1)

        y = temp_t
        data = Data(x=node_features, edge_index=edge_index,
                    edge_attr=edge_attr, signals=signals, y=y)
        return [data], []
