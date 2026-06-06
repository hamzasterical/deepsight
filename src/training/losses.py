import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        dice_weight: float = 0.5,
        bce_mask_weight: float = 0.3,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_mask_weight = bce_mask_weight
        self.label_smoothing = label_smoothing
        self.dice = DiceLoss(smooth=1.0)

        self.register_buffer("pw", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        label_logit: torch.Tensor,
        mask_sigmoid: torch.Tensor,
        label_target: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> torch.Tensor:

        label_target_smooth = label_target * (1.0 - self.label_smoothing) + \
                              0.5 * self.label_smoothing
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=self.pw)
        loss_label = bce_fn(
            label_logit.squeeze(1),
            label_target_smooth,
        )

        loss_dice = self.dice(mask_sigmoid, mask_target)

        p = mask_sigmoid.clamp(1e-6, 1.0 - 1e-6)
        loss_bce_mask = F.binary_cross_entropy(p, mask_target)

        return loss_label + self.dice_weight * loss_dice + self.bce_mask_weight * loss_bce_mask
