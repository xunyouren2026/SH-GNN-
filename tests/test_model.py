"""Integration tests for SH-GNN model."""

import torch
import pytest
from sh_gnn import SHGNN, Config
from torch_geometric.data import Data


def test_model_forward():
    cfg = Config()
    cfg.l_max = 4
    cfg.in_features = 3
    cfg.out_features = 1
    model = SHGNN(cfg)
    N = 10
    E = 30
    x = torch.randn(N, cfg.in_features)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 3)
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, y=torch.randn(N, 1))
    task_out, alms, phys_loss = model(data, return_phys_loss=True)
    assert task_out.shape == (N, cfg.out_features)
    assert alms.shape == (N, (cfg.l_max+1)**2)
    assert phys_loss.isfinite()


def test_equivariance_property():
    # This is a simplified test: apply random rotation to all points and see if output transforms accordingly.
    # For rigorous test, we would need to rotate the spherical harmonics basis.
    # Placeholder: model should produce same task output when rotating input positions? Not exactly; classification should be invariant.
    pass
