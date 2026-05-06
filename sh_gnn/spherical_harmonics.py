"""Numerically stable real spherical harmonics."""

import torch
import math
from math import pi, sqrt

_LOG_FACT_CACHE = {}


def _log_fact(n: int) -> float:
    if n < 0:
        return -float('inf')
    if n not in _LOG_FACT_CACHE:
        if n == 0:
            _LOG_FACT_CACHE[n] = 0.0
        else:
            _LOG_FACT_CACHE[n] = _LOG_FACT_CACHE.get(n-1, 0.0) + math.log(n)
    return _LOG_FACT_CACHE[n]


def _stable_associated_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    abs_m = abs(m)
    x = x.clamp(-1.0, 1.0)
    sin_theta = torch.sqrt(1.0 - x * x + 1e-12)

    p_mm = sin_theta ** abs_m
    if abs_m > 0:
        double_fact = 1.0
        for k in range(1, 2 * abs_m, 2):
            double_fact *= k
        p_mm = double_fact * p_mm
    if abs_m % 2 == 1:
        p_mm = -p_mm

    if l == abs_m:
        return p_mm

    p_prev = p_mm
    p_curr = x * (2 * abs_m + 1) * p_prev
    if l == abs_m + 1:
        return p_curr

    for k in range(abs_m + 1, l):
        p_next = ((2 * k + 1) * x * p_curr - (k + abs_m)
                  * p_prev) / (k - abs_m + 1)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def spherical_harmonics(l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    abs_m = abs(m)
    log_norm = 0.5 * (math.log(2*l+1) - math.log(4*math.pi))
    log_norm += 0.5 * (_log_fact(l - abs_m) - _log_fact(l + abs_m))
    norm = torch.exp(torch.tensor(
        log_norm, dtype=theta.dtype, device=theta.device))

    x = torch.cos(theta)
    p_lm = _stable_associated_legendre(l, m, x)

    if m >= 0:
        angular = torch.cos(m * phi)
    else:
        angular = torch.sin(abs_m * phi)

    return norm * p_lm * angular


def compute_sh_basis(l_max: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    N = theta.shape[0]
    num_sh = (l_max + 1) ** 2
    Y = torch.zeros(N, num_sh, device=theta.device, dtype=theta.dtype)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l * l + l + m
            Y[:, idx] = spherical_harmonics(l, m, theta, phi)
    return Y
