"""Training engine with mixed precision, checkpointing, and logging."""

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Callable
import os
import time
import numpy as np
from tqdm import tqdm
from .config import Config
from .sh_gnn import SHGNN
from .utils import AverageMeter, save_checkpoint, load_checkpoint, Logger


class Trainer:
    def __init__(
        self,
        model: SHGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        task_loss_fn: Optional[Callable] = None,
        writer: Optional[SummaryWriter] = None,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.accumulation_steps = accumulation_steps

        if task_loss_fn is None:
            self.task_loss_fn = torch.nn.MSELoss()
        else:
            self.task_loss_fn = task_loss_fn

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_epochs, eta_min=1e-6
        )
        self.scaler = GradScaler(enabled=cfg.use_amp)
        self.writer = writer if writer else SummaryWriter(log_dir=cfg.log_dir)
        self.logger = Logger(log_dir=cfg.log_dir)

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
        if os.path.exists(checkpoint_path):
            self.start_epoch, self.best_val_loss = load_checkpoint(
                self.model, self.optimizer, checkpoint_path
            )
            self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses = AverageMeter()
        task_losses = AverageMeter()
        phys_losses = AverageMeter()
        l_effs = []

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for i, data in enumerate(pbar):
            data = data.to(self.device)
            with autocast(device_type=self.cfg.device, enabled=self.cfg.use_amp):
                task_out, alms, phys_loss = self.model(
                    data, return_phys_loss=True)
                task_loss = self.task_loss_fn(task_out, data.y)
                total_loss = (task_loss + phys_loss) / self.accumulation_steps

            self.scaler.scale(total_loss).backward()
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            losses.update(total_loss.item() *
                          self.accumulation_steps, data.num_graphs)
            task_losses.update(task_loss.item(), data.num_graphs)
            phys_losses.update(phys_loss.item(), data.num_graphs)
            l_effs.append(self.model.get_L_eff())

            pbar.set_postfix(
                total=losses.avg,
                task=task_losses.avg,
                phys=phys_losses.avg,
                L_eff=np.mean(l_effs[-50:]) if l_effs else 0,
            )

        mean_l_eff = np.mean(l_effs) if l_effs else 0
        if self.writer:
            self.writer.add_scalar("train/total_loss", losses.avg, epoch)
            self.writer.add_scalar("train/task_loss", task_losses.avg, epoch)
            self.writer.add_scalar("train/phys_loss", phys_losses.avg, epoch)
            self.writer.add_scalar("train/L_eff", mean_l_eff, epoch)

        return {
            "total_loss": losses.avg,
            "task_loss": task_losses.avg,
            "phys_loss": phys_losses.avg,
            "L_eff": mean_l_eff,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        task_loss_sum = 0.0
        phys_loss_sum = 0.0
        num_batches = 0
        for data in tqdm(self.val_loader, desc="Valid"):
            data = data.to(self.device)
            task_out, alms, phys_loss = self.model(data, return_phys_loss=True)
            task_loss = self.task_loss_fn(task_out, data.y)
            loss = task_loss + phys_loss
            total_loss += loss.item()
            task_loss_sum += task_loss.item()
            phys_loss_sum += phys_loss.item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        avg_task = task_loss_sum / max(num_batches, 1)
        avg_phys = phys_loss_sum / max(num_batches, 1)
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
            self.writer.add_scalar("val/task_loss", avg_task, epoch)
            self.writer.add_scalar("val/phys_loss", avg_phys, epoch)
        return avg_loss

    def run(self):
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.best_val_loss,
                os.path.join(self.cfg.checkpoint_dir, "latest.pt"),
            )
            if is_best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_val_loss,
                    os.path.join(self.cfg.checkpoint_dir, "best.pt"),
                )

            self.logger.info(
                f"Epoch {epoch}: train_total={train_metrics['total_loss']:.4f}, "
                f"val_loss={val_loss:.4f}, L_eff={train_metrics['L_eff']:.1f}"
            )
        self.writer.close()
        self.logger.info("Training finished.")
