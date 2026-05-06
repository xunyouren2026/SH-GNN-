"""
Final comprehensive comparison of all models.
Includes: PointNet, DGCNN, SH-GNN (all scales), with accuracy and speed.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, knn_graph

from sh_gnn import SHGNN, Config
from shape_data import generate_dataset, CLASS_NAMES, NUM_CLASSES, NUM_POINTS, K_NEIGHBORS

print("=" * 80)
print("  最终完整对比：所有模型 + 所有规模 + 准确率 + 速度")
print("=" * 80)

DEVICE = "cpu"
NUM_WARMUP = 5
NUM_RUNS = 30

# 生成测试数据
print("\n准备测试数据...")
test_data = generate_dataset(100, num_points=NUM_POINTS, k=K_NEIGHBORS, augment=False, noise_level=0.05)
train_data = generate_dataset(800, num_points=NUM_POINTS, k=K_NEIGHBORS, augment=True, noise_level=0.05)

# ============================================================
# PointNet 基线
# ============================================================
class PointNet(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = data.x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        return self.fc4(x)


# ============================================================
# DGCNN 基线
# ============================================================
class EdgeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, edge_index, batch):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        msg = self.fc(torch.cat([x_i, x_j - x_i], dim=1))
        num_nodes = x.size(0)
        out_dim = msg.size(1)
        out = torch.full((num_nodes, out_dim), float('-inf'), device=x.device)
        out = out.scatter_reduce(0, row.unsqueeze(1).expand(-1, out_dim),
                                  msg, reduce="amax", include_self=True)
        out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
        if self.residual is not None:
            return out + self.residual(x)
        return out + x[:, :out_dim]


class SimpleDGCNN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=10, k=16):
        super().__init__()
        self.k = k
        self.ec1 = EdgeConvBlock(in_dim, hidden)
        self.ec2 = EdgeConvBlock(hidden, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.ec1(x, edge_index, data.batch)
        x = self.ec2(x, edge_index, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.fc(x)


# ============================================================
# 训练和评估函数
# ============================================================
def train_and_evaluate(model, name, train_loader, test_loader, epochs=30, is_shgnn=False):
    print(f"\n  训练 {name}...")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            if is_shgnn:
                task_out, _, _ = model(batch, return_phys_loss=False)
                pooled = global_mean_pool(task_out, batch.batch)
                loss = criterion(pooled, batch.y)
            else:
                out = model(batch)
                loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                if is_shgnn:
                    task_out, _, _ = model(batch, return_phys_loss=False)
                    out = global_mean_pool(task_out, batch.batch)
                else:
                    out = model(batch)
                pred = out.argmax(1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        val_acc = 100.0 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 推理速度测试
    model.eval()
    test_sample = test_data[0]
    batch_test = Batch.from_data_list([test_sample]).to(DEVICE)
    for _ in range(5):
        with torch.no_grad():
            model(batch_test)
    times = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            model(batch_test)
        times.append((time.time() - t0) * 1000)
    infer_ms = np.median(times)

    params = sum(p.numel() for p in model.parameters())
    print(f"    ✓ {name}: Acc={best_val_acc:.1f}%, Params={params:,}, Infer={infer_ms:.1f}ms")

    return {
        "name": name,
        "accuracy": round(best_val_acc, 2),
        "params": params,
        "infer_ms": round(infer_ms, 2),
    }


# ============================================================
# 主对比实验
# ============================================================
results = []

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- PointNet ---
print("\n" + "=" * 60)
print("  [1/6] PointNet 基线")
print("=" * 60)
pointnet = PointNet(in_dim=3, hidden=64, out_dim=NUM_CLASSES)
results.append(train_and_evaluate(pointnet, "PointNet", train_loader, test_loader))

# --- DGCNN ---
print("\n" + "=" * 60)
print("  [2/6] DGCNN 基线")
print("=" * 60)
dgcnn = SimpleDGCNN(in_dim=3, hidden=64, out_dim=NUM_CLASSES, k=K_NEIGHBORS)
results.append(train_and_evaluate(dgcnn, "DGCNN", train_loader, test_loader))

# --- SH-GNN 各规模 ---
shgnn_configs = [
    ("SH-GNN Tiny", {"l_max": 3, "num_layers": 2, "hidden_dim": 32}),
    ("SH-GNN Small", {"l_max": 6, "num_layers": 3, "hidden_dim": 64}),
    ("SH-GNN Medium", {"l_max": 10, "num_layers": 3, "hidden_dim": 128}),
    ("SH-GNN Large", {"l_max": 16, "num_layers": 4, "hidden_dim": 128}),
]

for idx, (name, config) in enumerate(shgnn_configs, 3):
    print("\n" + "=" * 60)
    print(f"  [{idx}/6] {name}")
    print("=" * 60)

    cfg = Config()
    for k, v in config.items():
        setattr(cfg, k, v)
    cfg.in_features = 3
    cfg.out_features = NUM_CLASSES

    try:
        model = SHGNN(cfg).to(DEVICE)
        results.append(train_and_evaluate(model, name, train_loader, test_loader, is_shgnn=True))
    except Exception as e:
        print(f"    ✗ Error: {e}")
        results.append({"name": name, "error": str(e)})

# ============================================================
# 最终总结
# ============================================================
print("\n" + "=" * 80)
print("  最终对比总结")
print("=" * 80)

print(f"\n  {'模型':<20} {'准确率':>10} {'参数量':>12} {'推理(ms)':>10} {'效率':>10}")
print(f"  {'-' * 70}")

for r in results:
    if "error" not in r:
        eff = r['accuracy'] / r['infer_ms'] if r['infer_ms'] > 0 else 0
        print(f"  {r['name']:<20} {r['accuracy']:>9.1f}% {r['params']:>12,} {r['infer_ms']:>10.2f} {eff:>9.2f}")

# 找出最佳
valid_results = [r for r in results if "error" not in r]
best_acc = max(valid_results, key=lambda x: x['accuracy'])
fastest = min(valid_results, key=lambda x: x['infer_ms'])
best_eff = max(valid_results, key=lambda x: x['accuracy'] / x['infer_ms'])

print(f"\n  🏆 最佳准确率: {best_acc['name']} ({best_acc['accuracy']:.1f}%)")
print(f"  ⚡ 最快推理: {fastest['name']} ({fastest['infer_ms']:.1f}ms)")
print(f"  📊 最佳效率(准确率/时间): {best_eff['name']} ({best_eff['accuracy']/best_eff['infer_ms']:.2f})")

# 保存结果
with open("final_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  结果保存: final_comparison_results.json")
print("=" * 80)
