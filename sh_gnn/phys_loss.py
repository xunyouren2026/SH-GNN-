"""Physical constraints loss: power spectrum matching, non-negativity, smoothness."""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


class PhysConstraintLoss(nn.Module):
    """
    Loss that enforces physical properties on spherical harmonic coefficients.
    """

    def __init__(
        self,
        l_max: int,
        theory_Cl: Optional[torch.Tensor] = None,
        lambda_phys: float = 1.0,
        lambda_nonneg: float = 10.0,
        lambda_smooth: float = 0.01,
        f_sky: float = 0.8,
        n_pix: int = 50331648,
    ):
        super().__init__()
        self.l_max = l_max
        self.lambda_phys = lambda_phys
        self.lambda_nonneg = lambda_nonneg
        self.lambda_smooth = lambda_smooth

        if theory_Cl is not None:
            self.register_buffer("theory_Cl", theory_Cl)
        else:
            ells = torch.arange(l_max + 1, dtype=torch.float32)
            As = 2.1e-9
            Cl = torch.zeros(l_max + 1)
            for l in range(2, l_max + 1):
                Cl[l] = As * (2 * math.pi) / (l * (l + 1)) * \
                    math.exp(-(l / 2500) ** 2)
            self.register_buffer("theory_Cl", Cl)

        weights = torch.zeros(l_max + 1)
        for l in range(l_max + 1):
            weights[l] = (2 * l + 1) * f_sky * n_pix / 2.0
        weights /= weights.sum()
        self.register_buffer("weights", weights)

    def angular_power_spectrum(self, alms: torch.Tensor) -> torch.Tensor:
        batch = alms.shape[0]
        device = alms.device
        dtype = alms.dtype
        if alms.is_complex():
            alms_abs2 = (alms.real ** 2 + alms.imag ** 2).float()
        else:
            alms_abs2 = (alms ** 2).float()
        Cl = torch.zeros(batch, self.l_max + 1, device=device, dtype=dtype)
        for l in range(self.l_max + 1):
            m_start = l * l
            m_end = (l + 1) * (l + 1)
            Cl[:, l] = torch.mean(alms_abs2[:, m_start:m_end], dim=1)
        return Cl

    def forward(self, pred_alms: torch.Tensor, L_eff: Optional[int] = None) -> torch.Tensor:
        if L_eff is None:
            L_eff = self.l_max
        else:
            L_eff = min(L_eff, self.l_max)

        pred_Cl = self.angular_power_spectrum(pred_alms)[:, :L_eff + 1]
        theory_Cl = self.theory_Cl[:L_eff + 1].to(pred_Cl.device)
        weights = self.weights[:L_eff + 1].to(pred_Cl.device)

        loss_matching = (weights.unsqueeze(
            0) * (pred_Cl - theory_Cl.unsqueeze(0)) ** 2).mean()

        loss_nonneg = torch.relu(-pred_Cl).mean()

        if L_eff >= 3:
            diff = pred_Cl[:, :-2] - 2 * pred_Cl[:, 1:-1] + pred_Cl[:, 2:]
            loss_smooth = (diff ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=pred_alms.device)

        total = self.lambda_phys * loss_matching + self.lambda_nonneg * \
            loss_nonneg + self.lambda_smooth * loss_smooth
        return total
