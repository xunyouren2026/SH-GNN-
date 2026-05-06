"""SH-GNN: Spherical Harmonic Graph Neural Network with strict SO(3) equivariance."""

from .config import Config
from .sh_gnn import SHGNN
from .datasets import (
    SphericalGraphDataset,
    ModelNet40Spherical,
    QM9Spherical,
)
from .trainer import Trainer
from .export import export_to_onnx, export_to_tensorrt

__version__ = "1.0.0"
__all__ = [
    "Config",
    "SHGNN",
    "SphericalGraphDataset",
    "ModelNet40Spherical",
    "QM9Spherical",
    "Trainer",
    "export_to_onnx",
    "export_to_tensorrt",
]
