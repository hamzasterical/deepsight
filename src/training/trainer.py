import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import CombinedLoss
from src.training.metrics import batch_to_numpy, compute_metrics
from src.training.scheduler import create_scheduler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        self.phase1_epochs = train_cfg.get("phase1_epochs", 10)
        self.phase2_epochs = train_cfg.get("phase2_epochs", 30)
        self.lr = train_cfg.get("learning_rate", 0.0001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.dice_weight = train_cfg.get("dice_loss_weight", 0.5)
        self.early_stop_patience = train_cfg.get("early_stopping_patience", 7)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = create_scheduler(self.optimizer)
        self.criterion = CombinedLoss(dice_weight=self.dice_weight)

        self.best_val_auc = 0.0
        self.early_stop_counter = 0
        self.current_epoch = 0
        self.checkpoints_saved = []

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_label_loss = 0.0
        total_mask_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            rgb = batch["rgb"].to(self.device)
            noise = batch["noise"].to(self.device)
            labels = batch["label"].to(self.device).float().view(-1, 1)
            masks = batch["mask"].to(self.device).float()

            self.optimizer.zero_grad()
            pred_labels, pred_masks = self.model(rgb, noise)
            loss = self.criterion(pred_labels, labels, pred_masks, masks)
            loss.backward()
            self.optimizer.step()

            losses = self.criterion.separate(pred_labels, labels, pred_masks, masks)
            total_loss += losses["total_loss"]
            total_label_loss += losses["label_loss"]
            total_mask_loss += losses["mask_loss"]
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "label_loss": total_label_loss / num_batches,
            "mask_loss": total_mask_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_labels = []
        all_preds = []
        all_masks = []
        all_gt_masks = []
        all_forgery_types = []

        for batch in dataloader:
            rgb = batch["rgb"].to(self.device)
            noise = batch["noise"].to(self.device)
            labels = batch["label"].numpy()
            forgery_types = batch.get("forgery_type", [None] * len(labels))

            pred_labels, pred_masks = self.model(rgb, noise)
            pred_probs = torch.sigmoid(pred_labels)

            all_labels.extend(labels)
            all_preds.extend(batch_to_numpy(pred_probs).flatten())
            all_masks.extend(batch_to_numpy(pred_masks))
            all_gt_masks.extend(batch_to_numpy(masks))
            all_forgery_types.extend(forgery_types)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_masks = np.array(all_masks)
        all_gt_masks = np.array(all_gt_masks)

        metrics = compute_metrics(all_labels, all_preds, all_masks, all_gt_masks, all_forgery_types)
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def save_checkpoint(self, metrics: Dict, is_best: bool = False) -> str:
        epoch = self.current_epoch
        filename = f"epoch_{epoch:03d}_auc_{metrics.get('overall', {}).get('auc_roc', 0):.4f}.pth"
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved: %s", filepath)

        self.checkpoints_saved.append(filepath)
        if len(self.checkpoints_saved) > 3:
            oldest = self.checkpoints_saved.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug("Removed old checkpoint: %s", oldest)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info("Best model updated: %s", best_path)

        return str(filepath)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        phase1: bool = False,
    ) -> Dict:
        num_epochs = self.phase1_epochs if phase1 else self.phase2_epochs
        logger.info(
            "Starting %s training for %d epochs", "Phase 1" if phase1 else "Phase 2", num_epochs
        )

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            val_auc = val_metrics.get("overall", {}).get("auc_roc", 0)
            self.scheduler.step(val_auc)

            elapsed = time.time() - start
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_auc=%.4f | val_f1=%.4f | val_iou=%.4f | lr=%.6f | %.2fs",
                epoch, num_epochs,
                train_metrics["loss"],
                val_auc,
                val_metrics.get("overall", {}).get("f1", 0),
                val_metrics.get("overall", {}).get("iou", 0),
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.early_stop_counter = 0
                self.save_checkpoint(val_metrics, is_best=True)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter % 3 == 0:
                    self.save_checkpoint(val_metrics)

            if self.early_stop_counter >= self.early_stop_patience:
                logger.info("Early stopping triggered after %d epochs", epoch)
                break

        return {"best_val_auc": self.best_val_auc}

    def load_checkpoint(self, filepath: str) -> int:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_auc = checkpoint.get("metrics", {}).get("overall", {}).get("auc_roc", 0)
        logger.info("Loaded checkpoint: %s (epoch %d)", filepath, self.current_epoch)
        return self.current_epoch
