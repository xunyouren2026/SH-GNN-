"""Main SH-GNN model: encoder, equivariant layers, decoder, with dynamic sparsity and physics loss."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .config import Config
from .dynamic_sparse import DynamicSparseScheduler
from .phys_loss import PhysConstraintLoss
from .sh_conv import SHEquivariantConv


class SHGNN(nn.Module):
    def __init__(self, cfg: Optional[Config] = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = Config()
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        self.cfg = cfg

        self.l_max = cfg.l_max
        self.num_sh = (self.l_max + 1) ** 2
        self.epsilon = cfg.epsilon

        self.encoder = nn.Sequential(
            nn.Linear(cfg.in_features, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(
                SHEquivariantConv(
                    in_channels=cfg.hidden_dim,
                    out_channels=cfg.hidden_dim,
                    l_max=self.l_max,
                    radial_dim=cfg.radial_dim,
                )
            )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, self.num_sh + cfg.out_features),
        )

        self.sparse_scheduler = DynamicSparseScheduler(
            l_max=self.l_max, epsilon=self.epsilon, min_l=cfg.min_l
        )
        self.phys_loss_fn = PhysConstraintLoss(
            l_max=self.l_max,
            theory_Cl=cfg.theory_cl if hasattr(
                cfg, "theory_cl") and cfg.theory_cl else None,
            lambda_phys=cfg.lambda_phys,
            lambda_nonneg=cfg.lambda_nonneg,
            lambda_smooth=cfg.lambda_smooth,
            f_sky=cfg.f_sky,
            n_pix=cfg.n_pix,
        )
        self.current_L_eff = self.l_max

    def forward(self, data: Any, return_phys_loss: bool = True, compute_L_eff: bool = True):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if compute_L_eff and hasattr(data, "signals") and data.signals is not None:
            if data.signals.dim() == 2:
                global_signal = data.signals.mean(dim=0, keepdim=True)
            else:
                global_signal = data.signals
            # If signals are all zeros (no real SH data), fall back to l_max
            if global_signal.abs().sum() < 1e-10:
                self.current_L_eff = self.l_max
            else:
                self.current_L_eff = self.sparse_scheduler.compute_L_eff(
                    global_signal)
        else:
            self.current_L_eff = self.l_max

        h = self.encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr, self.current_L_eff)
        out = self.decoder(h)
        alms = out[:, : self.num_sh]
        task_out = out[:, self.num_sh:]

        global_alms = alms.mean(dim=0, keepdim=True)
        if return_phys_loss:
            phys_loss = self.phys_loss_fn(global_alms, self.current_L_eff)
        else:
            phys_loss = None
        return task_out, alms, phys_loss

    def predict(self, data: Any) -> torch.Tensor:
        task_out, _, _ = self.forward(data, return_phys_loss=False)
        return task_out

    def get_L_eff(self) -> int:
        return self.current_L_eff
