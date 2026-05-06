"""Global configuration for SH-GNN."""
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Config:
    # Model architecture
    in_features: int = 16
    hidden_dim: int = 32
    out_features: int = 8
    l_max: int = 3
    num_layers: int = 2
    dropout: float = 0.1
    radial_dim: int = 8

    # Dynamic sparsity
    epsilon: float = 1e-3
    min_l: int = 2

    # Physical constraint loss
    lambda_phys: float = 1.0
    lambda_nonneg: float = 10.0
    lambda_smooth: float = 0.01
    f_sky: float = 0.8
    n_pix: int = 50331648

    # Training
    batch_size: int = 2
    learning_rate: float = 1e-3
    num_epochs: int = 10
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = False
    device: str = "cpu"

    # Data
    dataset: str = "modelnet40"
    data_root: str = "./data"
    num_workers: int = 0

    # Export / Serving
    onnx_opset: int = 14
    tensorrt_precision: str = "fp16"
    serve_host: str = "0.0.0.0"
    serve_port: int = 8000

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Theory C_l (optional)
    theory_cl: Optional[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, falling back to CPU.")


cfg = Config()
