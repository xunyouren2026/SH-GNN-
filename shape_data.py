"""
Shared data generation utilities for structured synthetic 3D shapes.
"""
import torch
import numpy as np
import math
from torch_geometric.data import Data

NUM_CLASSES = 10
NUM_TRAIN = 800
NUM_TEST = 200
NUM_POINTS = 256
K_NEIGHBORS = 16
NOISE_LEVEL = 0.05

CLASS_NAMES = ["Sphere", "Cube", "Cylinder", "Cone", "Torus",
               "Ellipsoid", "Plane", "Pyramid", "Hemisphere", "Diamond"]


def compute_sh_signals(points, l_max):
    """Compute real spherical harmonic coefficients for each point as signals.

    This enables DynamicSparseScheduler to adaptively determine L_eff
    based on actual signal energy distribution (Parseval's identity).
    """
    n = points.shape[0]
    # Normalize points to unit sphere for SH computation
    norms = torch.norm(points, dim=1, keepdim=True) + 1e-8
    p = points / norms
    theta = torch.acos(torch.clamp(p[:, 2], -1.0, 1.0))  # [0, pi]
    phi = torch.atan2(p[:, 1], p[:, 0])  # [-pi, pi]

    num_sh = (l_max + 1) ** 2
    signals = torch.zeros(n, num_sh)

    for l in range(l_max + 1):
        # Normalization factor
        norm_factor = math.sqrt((2 * l + 1) / (4 * math.pi))
        for m in range(-l, l + 1):
            idx = l * l + (l + m)
            # Real spherical harmonics (Condon-Shortley)
            abs_m = abs(m)
            # Compute associated Legendre P_l^|m|(cos theta)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            if abs_m == 0:
                plm = _legendre(l, 0, cos_t)
            else:
                plm = ((-1) ** abs_m) * (sin_t ** abs_m) * _legendre(l, abs_m, cos_t)

            ylm = norm_factor * plm
            if m > 0:
                ylm = ylm * math.sqrt(2) * torch.cos(m * phi)
            elif m < 0:
                ylm = ylm * math.sqrt(2) * torch.sin(abs(m) * phi)
            signals[:, idx] = ylm

    return signals


def _legendre(l, m, x):
    """Compute associated Legendre polynomial P_l^m(x) via stable recurrence."""
    if m > l:
        return torch.zeros_like(x)
    # Starting value P_m^m
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    # Recurrence upward
    pmm1 = x * ((2 * m + 1) * pmm)
    if l == m + 1:
        return pmm1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = pll
    return pll


def generate_sphere(n, radius=1.0):
    theta = torch.rand(n) * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_cube(n, size=1.0):
    points = []
    per_face = n // 6
    for _ in range(6):
        u = (torch.rand(per_face) - 0.5) * 2 * size
        v = (torch.rand(per_face) - 0.5) * 2 * size
        face = torch.zeros(per_face, 3)
        if _ == 0: face[:, 0] = size; face[:, 1] = u; face[:, 2] = v
        elif _ == 1: face[:, 0] = -size; face[:, 1] = u; face[:, 2] = v
        elif _ == 2: face[:, 1] = size; face[:, 0] = u; face[:, 2] = v
        elif _ == 3: face[:, 1] = -size; face[:, 0] = u; face[:, 2] = v
        elif _ == 4: face[:, 2] = size; face[:, 0] = u; face[:, 1] = v
        elif _ == 5: face[:, 2] = -size; face[:, 0] = u; face[:, 1] = v
        points.append(face)
    return torch.cat(points, dim=0)[:n]


def generate_cylinder(n, radius=0.7, height=1.4):
    n_side = int(n * 0.7)
    n_top = (n - n_side) // 2
    n_bot = n - n_side - n_top
    theta = torch.rand(n_side) * 2 * np.pi
    h = (torch.rand(n_side) - 0.5) * height
    side = torch.stack([radius*torch.cos(theta), radius*torch.sin(theta), h], dim=1)
    r = torch.sqrt(torch.rand(n_top)) * radius
    t = torch.rand(n_top) * 2 * np.pi
    top = torch.stack([r*torch.cos(t), r*torch.sin(t), torch.full((n_top,), height/2)], dim=1)
    r = torch.sqrt(torch.rand(n_bot)) * radius
    t = torch.rand(n_bot) * 2 * np.pi
    bot = torch.stack([r*torch.cos(t), r*torch.sin(t), torch.full((n_bot,), -height/2)], dim=1)
    return torch.cat([side, top, bot], dim=0)[:n]


def generate_cone(n, radius=0.8, height=1.6):
    points = []
    n_side = int(n * 0.8)
    n_base = n - n_side
    t = torch.rand(n_side)
    theta = torch.rand(n_side) * 2 * np.pi
    r = radius * (1 - t)
    h = height * t - height / 2
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    r = torch.sqrt(torch.rand(n_base)) * radius
    th = torch.rand(n_base) * 2 * np.pi
    points.append(torch.stack([r*torch.cos(th), r*torch.sin(th), torch.full((n_base,), -height/2)], dim=1))
    return torch.cat(points, dim=0)[:n]


def generate_torus(n, R=0.8, r=0.3):
    theta = torch.rand(n) * 2 * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=1)


def generate_ellipsoid(n, a=1.0, b=0.6, c=0.4):
    theta = torch.rand(n) * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = a * torch.sin(theta) * torch.cos(phi)
    y = b * torch.sin(theta) * torch.sin(phi)
    z = c * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_plane(n, size=1.2):
    x = (torch.rand(n) - 0.5) * 2 * size
    y = (torch.rand(n) - 0.5) * 2 * size
    z = (torch.rand(n) - 0.5) * 0.05
    return torch.stack([x, y, z], dim=1)


def generate_pyramid(n, base=1.0, height=1.4):
    points = []
    n_side = int(n * 0.8)
    n_base = n - n_side
    per_face = n_side // 4
    apex = torch.tensor([0, 0, height/2])
    corners = [
        torch.tensor([base/2, base/2, -height/2]),
        torch.tensor([-base/2, base/2, -height/2]),
        torch.tensor([-base/2, -base/2, -height/2]),
        torch.tensor([base/2, -base/2, -height/2]),
    ]
    for i in range(4):
        c1, c2 = corners[i], corners[(i+1)%4]
        u = torch.rand(per_face)
        v = torch.rand(per_face)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = c1.unsqueeze(0) * (1-u-v).unsqueeze(1) + c2.unsqueeze(0) * v.unsqueeze(1) + apex.unsqueeze(0) * u.unsqueeze(1)
        points.append(pts)
    u = (torch.rand(n_base) - 0.5) * base
    v = (torch.rand(n_base) - 0.5) * base
    points.append(torch.stack([u, v, torch.full((n_base,), -height/2)], dim=1))
    return torch.cat(points, dim=0)[:n]


def generate_hemisphere(n, radius=1.0):
    theta = torch.rand(n) * (np.pi / 2)
    phi = torch.rand(n) * 2 * np.pi
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_diamond(n, radius=1.0):
    points = []
    n_half = n // 2
    t = torch.rand(n_half)
    theta = torch.rand(n_half) * 2 * np.pi
    r = radius * (1 - t)
    h = t * radius
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    t = torch.rand(n - n_half)
    theta = torch.rand(n - n_half) * 2 * np.pi
    r = radius * (1 - t)
    h = -t * radius
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    return torch.cat(points, dim=0)[:n]


SHAPE_GENERATORS = [
    generate_sphere, generate_cube, generate_cylinder, generate_cone,
    generate_torus, generate_ellipsoid, generate_plane, generate_pyramid,
    generate_hemisphere, generate_diamond,
]


def random_rotation(points):
    axis = torch.randn(3)
    axis = axis / (torch.norm(axis) + 1e-8)
    angle = (torch.rand(1) * 2 * np.pi).item()
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = torch.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return points @ R.T


def random_scale(points, scale_range=(0.8, 1.2)):
    s = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return points * s


def add_noise(points, noise_level=0.05):
    noise = torch.randn_like(points) * noise_level
    return points + noise


def pytorch_knn_graph(x, k, loop=False):
    dist = torch.cdist(x, x)
    if not loop:
        dist.fill_diagonal_(float('inf'))
    _, indices = dist.topk(k, dim=1, largest=False)
    row = torch.arange(x.size(0)).unsqueeze(1).expand(-1, k)
    edge_index = torch.stack([row, indices], dim=0).reshape(2, -1)
    return edge_index


def generate_dataset(num_samples, num_points=NUM_POINTS, k=K_NEIGHBORS,
                     noise_level=NOISE_LEVEL, augment=True, l_max=6):
    data_list = []
    samples_per_class = num_samples // NUM_CLASSES

    for cls_id in range(NUM_CLASSES):
        for i in range(samples_per_class):
            points = SHAPE_GENERATORS[cls_id](num_points)

            if augment:
                points = random_rotation(points)
                points = random_scale(points)
                points = add_noise(points, noise_level)

            points = points / (torch.norm(points, dim=1, keepdim=True) + 1e-8)

            edge_index = pytorch_knn_graph(points, k=k, loop=False)

            row, col = edge_index
            rel = points[col] - points[row]
            dist = torch.norm(rel, dim=1, keepdim=True)
            rel_norm = rel / (dist + 1e-8)
            edge_theta = torch.acos(torch.clamp(rel_norm[:, 2], -1.0, 1.0))
            edge_phi = torch.atan2(rel_norm[:, 1], rel_norm[:, 0]) + np.pi
            edge_phi = edge_phi % (2 * np.pi)
            edge_attr = torch.cat([dist, edge_theta.unsqueeze(1), edge_phi.unsqueeze(1)], dim=1)

            signals = compute_sh_signals(points, l_max)

            y = cls_id

            data = Data(
                x=points,
                edge_index=edge_index,
                edge_attr=edge_attr,
                signals=signals,
                y=torch.tensor([y], dtype=torch.long),
            )
            data_list.append(data)

    return data_list
