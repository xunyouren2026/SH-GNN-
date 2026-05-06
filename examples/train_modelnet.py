"""
Train SH-GNN on ModelNet40 (requires internet to download dataset).
"""
import torch
from sh_gnn import SHGNN, Config, Trainer, ModelNet40Spherical
from torch_geometric.loader import DataLoader

cfg = Config()
cfg.dataset = "modelnet40"
cfg.l_max = 6
cfg.batch_size = 16
cfg.num_epochs = 50

train_dataset = ModelNet40Spherical(
    root=cfg.data_root, train=True, l_max=cfg.l_max)
val_dataset = ModelNet40Spherical(
    root=cfg.data_root, train=False, l_max=cfg.l_max)

train_loader = DataLoader(
    train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

model = SHGNN(cfg).to(cfg.device)
trainer = Trainer(model, train_loader, val_loader, cfg)
trainer.run()
