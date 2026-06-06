import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int = 1):
    scheduler_type = cfg["training"].get("scheduler", "plateau")
    total_epochs   = cfg["training"]["phase2_epochs"]
    warmup_epochs  = cfg["training"].get("warmup_epochs", 5)
    min_lr         = 1e-7

    if scheduler_type == "cosine":
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        scheduler._is_plateau = False
        return scheduler

    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=min_lr,
        )
        scheduler._is_plateau = True
        return scheduler
