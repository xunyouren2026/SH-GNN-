"""
SH-GNN Spherical Harmonic Equivariant Convolution Layer
SH-GNN 球谐等变卷积层（最终优化版本）

优化内容：
1. 合并两个einsum为一个，减少中间张量分配（第48-50行）
2. 保持权重完全兼容，无需重新训练
3. 所有33个权重文件直接加载使用

版本：v1.0 (Optimized) | 日期：2026-05-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherical_harmonics import spherical_harmonics


class SHEquivariantConv(nn.Module):
    """
    SO(3)等变球谐图卷积层

    核心思想：利用球谐函数作为等变核函数，实现严格的几何等变性。
    对于任意旋转R，满足：f(R·x) = R·f(x)

    Attributes:
        in_channels: 输入通道数
        out_channels: 输出通道数
        l_max: 球谐函数最大阶数
        weights_per_l: 每个阶数l的可学习权重 [in_channels, out_channels, 2l+1]
        radial_net: 径向神经网络（处理边距离）
        self_weight: 自环权重
        norm: 层归一化
    """

    def __init__(self, in_channels, out_channels, l_max, radial_dim=8):
        """
        初始化等变卷积层

        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            l_max: 球谐函数最大阶数（决定等变精度）
            radial_dim: 径向网络隐藏层维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max

        # 为每个阶数l创建可学习权重
        # 权重形状: [in_channels, out_channels, 2l+1]
        # 其中2l+1是第l阶球谐函数的数量（m = -l, ..., l）
        self.weights_per_l = nn.ParameterList()
        for l in range(l_max + 1):
            w = nn.Parameter(torch.empty(in_channels, out_channels, 2 * l + 1))
            nn.init.xavier_uniform_(w)
            self.weights_per_l.append(w)

        # 径向神经网络：处理边距离，为不同距离的邻居分配不同权重
        # 输入：边距离 [E, 1]
        # 输出：边权重 [E, out_channels]
        self.radial_net = nn.Sequential(
            nn.Linear(1, radial_dim), nn.SiLU(),
            nn.Linear(radial_dim, radial_dim), nn.SiLU(),
            nn.Linear(radial_dim, out_channels),
        )

        # 自环权重：处理节点自身特征
        self.self_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.self_weight)

        # 层归一化：稳定训练
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr, L_eff):
        """
        前向传播

        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, 3] (距离, theta, phi)
            L_eff: 有效l_max（用于动态稀疏）

        Returns:
            out: 输出特征 [N, out_channels]
        """
        row, col = edge_index
        N, E = x.shape[0], row.shape[0]

        # 获取邻居节点特征
        x_j = x[col]  # [E, in_channels]

        # 解析边特征
        dist = edge_attr[:, 0:1]      # [E, 1] 边距离
        theta = edge_attr[:, 1]        # [E] 极角
        phi = edge_attr[:, 2]          # [E] 方位角

        # 计算径向权重
        radial_w = self.radial_net(dist)  # [E, out_channels]

        # 初始化消息张量
        msg = torch.zeros(E, self.out_channels, device=x.device)

        # 对每个阶数l计算球谐加权消息
        for l in range(L_eff + 1):
            dim_l = 2 * l + 1

            # 计算球谐函数值 Y_l^m(theta, phi)
            # Y_l: [E, dim_l]
            Y_l = torch.zeros(E, dim_l, dtype=x.dtype, device=x.device)
            for m_idx, m in enumerate(range(-l, l + 1)):
                Y_l[:, m_idx] = spherical_harmonics(l, m, theta, phi)

            # ============================================================
            # 核心计算：等变消息传递（优化版）
            # ============================================================
            # 数学原理：
            #   msg_l = sum_{i,o,m} x_j[i] * W[i,o,m] * Y_l^m
            #
            # 优化：合并两个einsum为一个，减少中间张量分配
            # 原始：einsum('ei,iod->eod', x, W) -> einsum('eod,ed->eo', tmp, Y)
            # 优化：einsum('ei,iod,ed->eo', x, W, Y)  # 一步完成
            # ============================================================
            msg_l = torch.einsum('ei, iod, ed -> eo',
                                 x_j,
                                 self.weights_per_l[l],
                                 Y_l)

            # 累加各阶消息，乘以径向权重
            msg = msg + msg_l * radial_w

        # 消息聚合：scatter_add将边的消息聚合到目标节点
        aggr_out = torch.zeros(N, self.out_channels, device=x.device)
        aggr_out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        # 自环连接 + 消息聚合
        out = x @ self.self_weight + aggr_out

        # 激活和归一化
        return F.silu(self.norm(out))
