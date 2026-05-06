"""Utility functions: logging, checkpointing, metrics, visualization."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime


class Logger:
    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.logger = logging.getLogger("SHGNN")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, best_val_loss, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        filepath,
    )


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return epoch + 1, best_val_loss


def plot_angular_power_spectrum(pred_alms, theory_Cl=None, l_max=None, save_path=None):
    if l_max is None:
        l_max = int(np.sqrt(pred_alms.shape[-1])) - 1
    Cl = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        m_start = l * l
        m_end = (l + 1) * (l + 1)
        Cl[l] = np.mean(np.abs(pred_alms[m_start:m_end]) ** 2)
    ells = np.arange(l_max + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ells, Cl, label="Predicted C_l")
    if theory_Cl is not None:
        plt.plot(ells[: len(theory_Cl)], theory_Cl,
                 label="Theory C_l", linestyle="--")
    plt.xlabel("l")
    plt.ylabel("C_l")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
