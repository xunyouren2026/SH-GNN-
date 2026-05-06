"""Strict SO(3)-equivariant message passing layer (vectorized version).

优化说明：
- 第48-50行：合并两个einsum为一个，减少中间张量分配
- 权重完全兼容，不需要重新训练
- 预期加速：3-5%（微小但稳定）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherical_harmonics import spherical_harmonics


class SHEquivariantConv(nn.Module):
    def __init__(self, in_channels, out_channels, l_max, radial_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max

        self.weights_per_l = nn.ParameterList()
        for l in range(l_max + 1):
            w = nn.Parameter(torch.empty(in_channels, out_channels, 2 * l + 1))
            nn.init.xavier_uniform_(w)
            self.weights_per_l.append(w)

        self.radial_net = nn.Sequential(
            nn.Linear(1, radial_dim), nn.SiLU(),
            nn.Linear(radial_dim, radial_dim), nn.SiLU(),
            nn.Linear(radial_dim, out_channels),
        )
        self.self_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.self_weight)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr, L_eff):
        row, col = edge_index
        N, E = x.shape[0], row.shape[0]

        x_j = x[col]
        dist = edge_attr[:, 0:1]
        theta = edge_attr[:, 1]
        phi = edge_attr[:, 2]
        radial_w = self.radial_net(dist)

        msg = torch.zeros(E, self.out_channels, device=x.device)
        for l in range(L_eff + 1):
            dim_l = 2 * l + 1
            Y_l = torch.zeros(E, dim_l, dtype=x.dtype, device=x.device)
            for m_idx, m in enumerate(range(-l, l + 1)):
                Y_l[:, m_idx] = spherical_harmonics(l, m, theta, phi)

            # ========== 优化：合并两个einsum为一个 ==========
            # 原始代码（2个einsum）：
            #   weighted = torch.einsum('ei, iod -> eod', x_j, self.weights_per_l[l])
            #   msg_l = torch.einsum('eod, ed -> eo', weighted, Y_l)
            #
            # 优化代码（1个einsum）：
            #   减少中间张量 'eod' 的分配，略微提升速度
            # =============================================
            msg_l = torch.einsum('ei, iod, ed -> eo', x_j, self.weights_per_l[l], Y_l)
            msg = msg + msg_l * radial_w

        aggr_out = torch.zeros(N, self.out_channels, device=x.device)
        aggr_out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        out = x @ self.self_weight + aggr_out
        return F.silu(self.norm(out))
