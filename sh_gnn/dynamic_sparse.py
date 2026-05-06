"""Adaptive angular truncation based on Parseval's identity."""

import torch
from typing import Tuple


class DynamicSparseScheduler:
    """
    Computes effective angular degree L_eff such that sum_{l=0}^{L_eff} E_l >= 1 - epsilon.
    E_l = sum_m |a_{lm}|^2.
    """

    def __init__(self, l_max: int, epsilon: float = 1e-3, min_l: int = 2):
        self.l_max = l_max
        self.epsilon = epsilon
        self.min_l = min_l

    def compute_L_eff(self, sh_coeffs: torch.Tensor) -> int:
        """
        sh_coeffs: (..., (l_max+1)^2) spherical harmonic coefficients.
        Returns L_eff (scalar integer, max over batch).
        """
        energy_per_l = []
        for l in range(self.l_max + 1):
            m_start = l * l
            m_end = (l + 1) * (l + 1)
            alms_l = sh_coeffs[..., m_start:m_end]
            energy_l = torch.sum(torch.abs(alms_l) ** 2, dim=-1)
            energy_per_l.append(energy_l)
        energy = torch.stack(energy_per_l, dim=-1)
        total_energy = torch.sum(energy, dim=-1, keepdim=True) + 1e-12
        cum_ratio = torch.cumsum(energy, dim=-1) / total_energy
        exceed = (cum_ratio >= (1 - self.epsilon)).float()
        L_eff_per_sample = torch.argmax(exceed, dim=-1)
        L_eff = int(L_eff_per_sample.max().item())
        return max(self.min_l, min(L_eff, self.l_max))

    def __call__(self, sh_coeffs: torch.Tensor) -> int:
        return self.compute_L_eff(sh_coeffs)
