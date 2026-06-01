import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        pred_label: torch.Tensor,
        label: torch.Tensor,
        pred_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        label_loss = self.bce(pred_label, label)
        mask_loss = self.dice(pred_mask, mask)
        total = label_loss + self.dice_weight * mask_loss
        return total

    def separate(
        self,
        pred_label: torch.Tensor,
        label: torch.Tensor,
        pred_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        label_loss = self.bce(pred_label, label)
        mask_loss = self.dice(pred_mask, mask)
        return {
            "label_loss": label_loss.item(),
            "mask_loss": mask_loss.item(),
            "total_loss": (label_loss + self.dice_weight * mask_loss).item(),
        }
