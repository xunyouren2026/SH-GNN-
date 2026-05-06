"""
Quickstart: Train a small SH-GNN on simulated data and run inference.
"""
import torch
from sh_gnn import SHGNN, Config
from torch_geometric.data import Data

# Create dummy data
N, E = 100, 400
x = torch.randn(N, 16)
edge_index = torch.randint(0, N, (2, E))
edge_attr = torch.rand(E, 3)
edge_attr[:, 0] *= 2.0
edge_attr[:, 1] *= 3.14159
edge_attr[:, 2] *= 2 * 3.14159
y = torch.randn(N, 8)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Create model
cfg = Config()
cfg.in_features = 16
cfg.out_features = 8
model = SHGNN(cfg)

# Forward pass
out, alms, phys_loss = model(data)
print(f"Output shape: {out.shape}, Physics loss: {phys_loss.item():.4f}")
