import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Classification-only BCE loss with label smoothing and pos_weight support."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets_smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce_fn(logits.squeeze(1), targets_smooth)
