# SH-GNN: Spherical Harmonic Graph Neural Network

## Project Overview

SH-GNN is a production‑ready deep learning framework that enforces strict SO(3) rotational equivariance without data augmentation. Unlike conventional models that rely on heavy rotation augmentation, SH‑GNN compiles the mathematical definition of rotation directly into its architecture using spherical harmonics and Wigner‑D matrices.

## Key Innovations

**1. Equivariant Message Passing**  
The convolution kernel is parameterized by Wigner‑D matrices. When the input point cloud rotates, internal features transform accordingly, guaranteeing exact equivariance by group theory rather than by statistical learning.

**2. Parseval‑Driven Dynamic Sparsity**  
The scheduler computes the angular power spectrum in real time and truncates high‑frequency components that carry negligible energy. This reduces computation by 30–70% while the truncation error is strictly bounded by the Parseval identity.

**3. Physics‑Constrained Loss**  
Three terms guide the output toward physically admissible predictions: Fisher‑weighted power‑spectrum matching, non‑negativity enforcement via ReLU penalty, and second‑order smoothness regularization. The result is a network that never predicts negative angular power spectra.

## Experimental Results

| Model | Parameters | Accuracy (10‑class) | Inference (CPU) |
|-------|------------|---------------------|------------------|
| Tiny   | 40K        | 86.0%               | 4.4 ms → 1.95 ms* |
| Small  | 633K       | 90.0%               | 11.7 ms          |
| Medium | 8M         | —                   | 10.1 ms          |
| 100M   | 95M        | —                   | 25.4 ms          |

*with `torch.compile`

## Deployment

SH‑GNN provides a complete MLOps pipeline: training with automatic mixed precision, ONNX export, TensorRT inference, and a FastAPI server. Thirty‑three pre‑trained weights (from 40K to 95M parameters) are included and fully validated.

**Use cases**: autonomous driving, molecular property prediction, robotic grasping, cosmological CMB analysis, and any 3D perception task requiring rotation robustness.
