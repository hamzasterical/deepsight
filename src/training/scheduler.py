import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = "max",
    factor: float = 0.5,
    patience: int = 3,
    threshold: float = 1e-4,
    min_lr: float = 1e-7,
) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        min_lr=min_lr,
        verbose=True,
    )
