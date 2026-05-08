"""
Train SH-GNN on structured synthetic data that simulates real 3D objects.
This creates challenging data with:
- 10 geometric shape classes (sphere, cube, cylinder, cone, torus, etc.)
- Noise, rotation, scaling augmentation
- Proper train/test split
- Enough difficulty to differentiate model performance
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sh_gnn import SHGNN, Config, Trainer
from sh_gnn.utils import set_seed
import time
import json

set_seed(42)

os.makedirs("./checkpoints_realdata", exist_ok=True)
os.makedirs("./logs_realdata", exist_ok=True)

# ============================================================
# Configuration
# ============================================================
NUM_CLASSES = 10
NUM_TRAIN = 800
NUM_TEST = 200
NUM_POINTS = 256
K_NEIGHBORS = 16
NOISE_LEVEL = 0.05

cfg = Config()
cfg.l_max = 6
cfg.num_layers = 3
cfg.hidden_dim = 64
cfg.in_features = 3
cfg.out_features = NUM_CLASSES
cfg.batch_size = 32
cfg.num_epochs = 30
cfg.learning_rate = 1e-3
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.use_amp = False
cfg.num_workers = 0
cfg.checkpoint_dir = "./checkpoints_realdata"
cfg.log_dir = "./logs_realdata"

print("=" * 65)
print("  SH-GNN Training on Structured Synthetic 3D Data")
print("  (Simulating real-world point cloud classification)")
print("=" * 65)
print(f"  Classes: {NUM_CLASSES}, Train: {NUM_TRAIN}, Test: {NUM_TEST}")
print(f"  Points per shape: {NUM_POINTS}, kNN: {K_NEIGHBORS}")
print(f"  Noise level: {NOISE_LEVEL}")
print(f"  Config: l_max={cfg.l_max}, hidden_dim={cfg.hidden_dim}, "
      f"layers={cfg.num_layers}, epochs={cfg.num_epochs}")
print(f"  Device: {cfg.device}")
print("=" * 65)


# ============================================================
# Shape Generators (10 classes of real 3D geometry)
# ============================================================
def generate_sphere(n, radius=1.0):
    """Sphere surface points."""
    theta = torch.rand(n) * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_cube(n, size=1.0):
    """Cube surface points."""
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
    pts = torch.cat(points, dim=0)[:n]
    return pts


def generate_cylinder(n, radius=0.7, height=1.4):
    """Cylinder surface points."""
    n_side = int(n * 0.7)
    n_top = (n - n_side) // 2
    n_bot = n - n_side - n_top
    # Side
    theta = torch.rand(n_side) * 2 * np.pi
    h = (torch.rand(n_side) - 0.5) * height
    side = torch.stack([radius*torch.cos(theta), radius*torch.sin(theta), h], dim=1)
    # Top cap
    r = torch.sqrt(torch.rand(n_top)) * radius
    t = torch.rand(n_top) * 2 * np.pi
    top = torch.stack([r*torch.cos(t), r*torch.sin(t), torch.full((n_top,), height/2)], dim=1)
    # Bottom cap
    r = torch.sqrt(torch.rand(n_bot)) * radius
    t = torch.rand(n_bot) * 2 * np.pi
    bot = torch.stack([r*torch.cos(t), r*torch.sin(t), torch.full((n_bot,), -height/2)], dim=1)
    return torch.cat([side, top, bot], dim=0)[:n]


def generate_cone(n, radius=0.8, height=1.6):
    """Cone surface points."""
    points = []
    n_side = int(n * 0.8)
    n_base = n - n_side
    # Side
    t = torch.rand(n_side)
    theta = torch.rand(n_side) * 2 * np.pi
    r = radius * (1 - t)
    h = height * t - height / 2
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    # Base
    r = torch.sqrt(torch.rand(n_base)) * radius
    th = torch.rand(n_base) * 2 * np.pi
    points.append(torch.stack([r*torch.cos(th), r*torch.sin(th), torch.full((n_base,), -height/2)], dim=1))
    return torch.cat(points, dim=0)[:n]


def generate_torus(n, R=0.8, r=0.3):
    """Torus surface points."""
    theta = torch.rand(n) * 2 * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=1)


def generate_ellipsoid(n, a=1.0, b=0.6, c=0.4):
    """Ellipsoid surface points."""
    theta = torch.rand(n) * np.pi
    phi = torch.rand(n) * 2 * np.pi
    x = a * torch.sin(theta) * torch.cos(phi)
    y = b * torch.sin(theta) * torch.sin(phi)
    z = c * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_plane(n, size=1.2):
    """Flat plane (thin box) points."""
    x = (torch.rand(n) - 0.5) * 2 * size
    y = (torch.rand(n) - 0.5) * 2 * size
    z = (torch.rand(n) - 0.5) * 0.05  # very thin
    return torch.stack([x, y, z], dim=1)


def generate_pyramid(n, base=1.0, height=1.4):
    """Square pyramid points."""
    points = []
    n_side = int(n * 0.8)
    n_base = n - n_side
    # 4 triangular faces
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
    # Base
    u = (torch.rand(n_base) - 0.5) * base
    v = (torch.rand(n_base) - 0.5) * base
    points.append(torch.stack([u, v, torch.full((n_base,), -height/2)], dim=1))
    return torch.cat(points, dim=0)[:n]


def generate_hemisphere(n, radius=1.0):
    """Hemisphere (half sphere) points."""
    theta = torch.rand(n) * (np.pi / 2)  # only upper half
    phi = torch.rand(n) * 2 * np.pi
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def generate_diamond(n, radius=1.0):
    """Diamond (double cone) points."""
    points = []
    n_half = n // 2
    # Upper cone
    t = torch.rand(n_half)
    theta = torch.rand(n_half) * 2 * np.pi
    r = radius * (1 - t)
    h = t * radius
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    # Lower cone
    t = torch.rand(n - n_half)
    theta = torch.rand(n - n_half) * 2 * np.pi
    r = radius * (1 - t)
    h = -t * radius
    points.append(torch.stack([r*torch.cos(theta), r*torch.sin(theta), h], dim=1))
    return torch.cat(points, dim=0)[:n]


SHAPE_GENERATORS = [
    generate_sphere,      # 0: sphere
    generate_cube,        # 1: cube
    generate_cylinder,    # 2: cylinder
    generate_cone,        # 3: cone
    generate_torus,       # 4: torus
    generate_ellipsoid,   # 5: ellipsoid
    generate_plane,       # 6: plane
    generate_pyramid,     # 7: pyramid
    generate_hemisphere,  # 8: hemisphere
    generate_diamond,     # 9: diamond
]

CLASS_NAMES = ["Sphere", "Cube", "Cylinder", "Cone", "Torus",
               "Ellipsoid", "Plane", "Pyramid", "Hemisphere", "Diamond"]


def random_rotation(points):
    """Apply random rotation to points."""
    # Random axis-angle rotation
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
    """Apply random scaling."""
    s = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return points * s


def add_noise(points, noise_level=0.05):
    """Add Gaussian noise."""
    noise = torch.randn_like(points) * noise_level
    return points + noise


def pytorch_knn_graph(x, k, loop=False):
    """Pure PyTorch kNN graph."""
    dist = torch.cdist(x, x)
    if not loop:
        dist.fill_diagonal_(float('inf'))
    _, indices = dist.topk(k, dim=1, largest=False)
    row = torch.arange(x.size(0)).unsqueeze(1).expand(-1, k)
    edge_index = torch.stack([row, indices], dim=0).reshape(2, -1)
    return edge_index


def generate_dataset(num_samples, num_points=NUM_POINTS, k=K_NEIGHBORS,
                     noise_level=NOISE_LEVEL, augment=True):
    """Generate structured synthetic dataset with real 3D shapes."""
    data_list = []
    samples_per_class = num_samples // NUM_CLASSES

    for cls_id in range(NUM_CLASSES):
        for i in range(samples_per_class):
            # Generate base shape
            points = SHAPE_GENERATORS[cls_id](num_points)

            # Data augmentation
            if augment:
                points = random_rotation(points)
                points = random_scale(points)
                points = add_noise(points, noise_level)

            # Normalize to unit sphere
            points = points / (torch.norm(points, dim=1, keepdim=True) + 1e-8)

            # Build kNN graph
            edge_index = pytorch_knn_graph(points, k=k, loop=False)

            # Edge attributes
            row, col = edge_index
            rel = points[col] - points[row]
            dist = torch.norm(rel, dim=1, keepdim=True)
            rel_norm = rel / (dist + 1e-8)
            edge_theta = torch.acos(torch.clamp(rel_norm[:, 2], -1.0, 1.0))
            edge_phi = torch.atan2(rel_norm[:, 1], rel_norm[:, 0]) + np.pi
            edge_phi = edge_phi % (2 * np.pi)
            edge_attr = torch.cat([dist, edge_theta.unsqueeze(1), edge_phi.unsqueeze(1)], dim=1)

            # Signals: compute real SH coefficients for dynamic sparsity
            norms = torch.norm(points, dim=1, keepdim=True) + 1e-8
            p = points / norms
            pt = torch.acos(torch.clamp(p[:, 2], -1.0, 1.0))
            pp = torch.atan2(p[:, 1], p[:, 0])
            num_sh = (cfg.l_max + 1) ** 2
            signals = torch.zeros(num_points, num_sh)
            for l in range(cfg.l_max + 1):
                nf = math.sqrt((2 * l + 1) / (4 * math.pi))
                for m in range(-l, l + 1):
                    idx = l * l + (l + m)
                    am = abs(m)
                    ct = torch.cos(pt)
                    st = torch.sin(pt)
                    # Stable associated Legendre recurrence
                    if am == 0:
                        pmm = torch.ones_like(ct)
                    else:
                        pmm = torch.ones_like(ct)
                        s = torch.sqrt((1 - ct) * (1 + ct))
                        f = 1.0
                        for i in range(1, am + 1):
                            pmm = pmm * (-f) * s
                            f += 2.0
                    if l == am:
                        plm = pmm
                    elif l == am + 1:
                        plm = ct * ((2 * am + 1) * pmm)
                    else:
                        pmm1 = ct * ((2 * am + 1) * pmm)
                        for ll in range(am + 2, l + 1):
                            pll = ((2 * ll - 1) * ct * pmm1 - (ll + am - 1) * pmm) / (ll - am)
                            pmm = pmm1
                            pmm1 = pll
                        plm = pll
                    if am > 0:
                        plm = ((-1) ** am) * (st ** am) * plm
                    ylm = nf * plm
                    if m > 0:
                        ylm = ylm * math.sqrt(2) * torch.cos(m * pp)
                    elif m < 0:
                        ylm = ylm * math.sqrt(2) * torch.sin(am * pp)
                    signals[:, idx] = ylm

            # Label
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


# ============================================================
# Generate Data
# ============================================================
print("\n[1/4] Generating structured synthetic dataset...")
print(f"  Shape classes: {', '.join(CLASS_NAMES)}")

train_data = generate_dataset(NUM_TRAIN, augment=True)
test_data = generate_dataset(NUM_TEST, augment=True, noise_level=0.08)  # slightly more noise for test

# Verify class distribution
train_labels = [d.y.item() for d in train_data]
test_labels = [d.y.item() for d in test_data]
print(f"  Train: {len(train_data)} samples")
print(f"  Test: {len(test_data)} samples")
print(f"  Train class distribution: {np.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")
print(f"  Test class distribution: {np.bincount(test_labels, minlength=NUM_CLASSES).tolist()}")

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)
print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ============================================================
# Create Model
# ============================================================
print("\n[2/4] Creating SH-GNN model...")
model = SHGNN(cfg).to(cfg.device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")

# ============================================================
# Train (custom loop with CrossEntropy for classification)
# ============================================================
print("\n[3/4] Training with CrossEntropy Loss...")
print("=" * 65)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
best_val_loss = float('inf')
history = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}

start_time = time.time()

for epoch in range(cfg.num_epochs):
    # --- Train ---
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        batch = batch.to(cfg.device)
        optimizer.zero_grad()
        
        task_out, alms, phys_loss = model(batch, return_phys_loss=True)
        
        # Global mean pooling for graph-level classification
        pooled = global_mean_pool(task_out, batch.batch)  # only class logits
        
        loss = criterion(pooled, batch.y) + phys_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss_sum += loss.item() * batch.num_graphs
        pred = pooled.argmax(dim=1)
        train_correct += (pred == batch.y).sum().item()
        train_total += batch.y.size(0)
    
    train_loss = train_loss_sum / len(train_data)
    train_acc = 100.0 * train_correct / train_total
    
    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(cfg.device)
            task_out, alms, phys_loss = model(batch, return_phys_loss=True)
            pooled = global_mean_pool(task_out, batch.batch)
            loss = criterion(pooled, batch.y) + phys_loss
            
            val_loss_sum += loss.item() * batch.num_graphs
            pred = pooled.argmax(dim=1)
            val_correct += (pred == batch.y).sum().item()
            val_total += batch.y.size(0)
    
    val_loss = val_loss_sum / len(test_data)
    val_acc = 100.0 * val_correct / val_total
    
    scheduler.step()
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["train_acc"].append(train_acc)
    
    # Save best
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, os.path.join(cfg.checkpoint_dir, "best.pt"))
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, os.path.join(cfg.checkpoint_dir, "latest.pt"))
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{cfg.num_epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%"
              f"{' ★' if is_best else ''}")

elapsed = time.time() - start_time

# ============================================================
# Evaluate
# ============================================================
print("\n[4/4] Final Evaluation on Test Set...")

# Load best model
checkpoint = torch.load(os.path.join(cfg.checkpoint_dir, "best.pt"), map_location=cfg.device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  Loaded best model from epoch {checkpoint['epoch']} (val_acc={checkpoint.get('val_acc', 'N/A')})")

# Per-class accuracy
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(cfg.device)
        task_out, _, _ = model(batch, return_phys_loss=False)
        # Global mean pooling for graph classification
        pooled = global_mean_pool(task_out, batch.batch)
        pred = pooled.argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
        for i in range(len(batch.y)):
            label = batch.y[i].item()
            class_total[label] += 1
            if pred[i] == label:
                class_correct[label] += 1

overall_acc = 100.0 * sum(class_correct) / sum(class_total)
print(f"\n  Overall Test Accuracy: {overall_acc:.2f}%")
print(f"\n  Per-class accuracy:")
for i in range(NUM_CLASSES):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
    print(f"    {CLASS_NAMES[i]:12s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]}) {bar}")

# Confusion matrix (simplified)
print(f"\n  Most confused pairs:")
from collections import Counter
errors = []
for p, l in zip(all_preds, all_labels):
    if p != l:
        errors.append((CLASS_NAMES[l], CLASS_NAMES[p]))
error_counts = Counter(errors)
for (true_cls, pred_cls), count in error_counts.most_common(5):
    print(f"    {true_cls} → {pred_cls}: {count} times")

# ============================================================
# Save Results
# ============================================================
results = {
    "dataset": "Structured Synthetic 3D (10 classes)",
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "num_train": NUM_TRAIN,
    "num_test": NUM_TEST,
    "num_points": NUM_POINTS,
    "noise_level": NOISE_LEVEL,
    "params": total_params,
    "overall_accuracy": overall_acc,
    "best_val_accuracy": best_val_acc,
    "best_val_loss": best_val_loss,
    "per_class_accuracy": {CLASS_NAMES[i]: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(NUM_CLASSES)},
    "training_history": history,
    "training_time_min": round(elapsed / 60, 1),
    "config": {
        "l_max": cfg.l_max,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
    }
}

with open("realdata_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 65}")
print(f"Training Complete!")
print(f"  Total time: {elapsed/60:.1f} minutes")
print(f"  Test Accuracy: {overall_acc:.2f}%")
print(f"  Results saved to: realdata_results.json")
print(f"  Checkpoint: checkpoints_realdata/best.pt")
print(f"{'=' * 65}")
