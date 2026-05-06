"""Unit tests for mathematical components."""

import torch
import numpy as np
import pytest
from sh_gnn.spherical_harmonics import spherical_harmonics, compute_sh_basis
from sh_gnn.wigner_d import stable_wigner_d
from sh_gnn.dynamic_sparse import DynamicSparseScheduler
from sh_gnn.phys_loss import PhysConstraintLoss
from math import pi


def test_sh_orthogonality():
    l_max = 5
    N = 1000
    theta = torch.rand(N) * pi
    phi = torch.rand(N) * 2 * pi
    Y = compute_sh_basis(l_max, theta, phi)
    inner = (Y.T @ Y) / N
    identity = torch.eye(inner.shape[0])
    diff = inner - identity
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-5)


def test_wigner_d_orthogonality():
    for l in [1, 2, 5, 10]:
        beta = torch.tensor(0.5)
        d = stable_wigner_d(l, beta)
        ortho = d @ d.T
        assert torch.allclose(ortho, torch.eye(ortho.shape[0]), atol=1e-6)


def test_dynamic_sparse():
    scheduler = DynamicSparseScheduler(l_max=10, epsilon=0.1)
    coeffs = torch.zeros(1, (11)**2)
    coeffs[0, 0] = 1.0
    L_eff = scheduler.compute_L_eff(coeffs)
    assert L_eff == max(scheduler.min_l, 0)
    assert L_eff == 2


def test_phys_loss_finite():
    loss_fn = PhysConstraintLoss(l_max=4)
    alms = torch.randn(2, (4+1)**2)
    loss = loss_fn(alms, L_eff=4)
    assert loss.isfinite().item()
