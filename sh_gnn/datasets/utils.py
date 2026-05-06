"""Utility functions for dataset processing."""

import torch
import numpy as np
from typing import Tuple, Optional
from math import pi


def sphere_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates to Cartesian.

    Args:
        theta: polar angle (0..pi)
        phi: azimuthal angle (0..2π)
    Returns:
        (x, y, z) tensor of same shape.
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_sphere(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian to spherical coordinates (theta, phi).

    Args:
        xyz: (..., 3)
    Returns:
        theta, phi tensors of same batch shape.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.norm(xyz, dim=-1)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    phi = torch.atan2(y, x) + pi
    phi = phi % (2 * pi)
    return theta, phi


def compute_angular_distance(theta1: torch.Tensor, phi1: torch.Tensor,
                             theta2: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    """Great-circle distance between two points on sphere."""
    cos_dist = torch.sin(theta1) * torch.sin(theta2) * \
        torch.cos(phi1 - phi2) + torch.cos(theta1) * torch.cos(theta2)
    return torch.acos(torch.clamp(cos_dist, -1.0, 1.0))


def knn_sphere(pos_cart: torch.Tensor, k: int, include_self: bool = False) -> torch.Tensor:
    """
    K-nearest neighbors on sphere using angular distance.
    """
    # Compute angular distance matrix
    theta, phi = cartesian_to_sphere(pos_cart)
    N = pos_cart.shape[0]
    # Use indexing; for large N, use cKDTree. Here we implement naive for simplicity.
    dist_matrix = torch.zeros(N, N, device=pos_cart.device)
    for i in range(N):
        for j in range(N):
            if i == j and not include_self:
                dist_matrix[i, j] = float('inf')
            else:
                dist_matrix[i, j] = compute_angular_distance(
                    theta[i], phi[i], theta[j], phi[j])
    # Get k smallest indices per row
    knn_indices = dist_matrix.topk(k, dim=1, largest=False).indices
    # Build edge_index
    row = torch.arange(N, device=pos_cart.device).repeat_interleave(k)
    col = knn_indices.flatten()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index
