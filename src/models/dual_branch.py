from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.classification_head import ClassificationHead
from src.models.fusion import FeatureFusion
from src.models.noise_branch import NoiseBranch
from src.models.rgb_branch import RGBBranch
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DualBranchModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained_rgb: bool = True,
        pretrained_noise: bool = False,
        feature_dim: int = 1280,
        hidden_dim: int = 768,
        dropout: float = 0.2,
        freeze_bn: bool = True,
    ):
        super().__init__()

        self.rgb_branch = RGBBranch(
            backbone=backbone,
            pretrained=pretrained_rgb,
            return_features=True,
            freeze_bn=freeze_bn,
        )
        self.noise_branch = NoiseBranch(
            backbone=backbone,
            pretrained=pretrained_noise,
            return_features=True,
            freeze_bn=freeze_bn,
        )
        self.fusion = FeatureFusion(
            rgb_dim=feature_dim,
            noise_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.classification_head = ClassificationHead(
            in_features=hidden_dim,
            dropout=dropout,
        )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        logger.debug(
            "DualBranchModel(backbone=%s, feature_dim=%d, hidden_dim=%d, dropout=%.2f)",
            backbone, feature_dim, hidden_dim, dropout,
        )

    def forward(self, rgb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        rgb_out = self.rgb_branch(rgb)

        noise_out = self.noise_branch(noise)

        fused = self.fusion(rgb_out, noise_out)

        fused_pooled = fused.mean(dim=[2, 3])
        label = self.classification_head(fused_pooled)

        return label

    def predict(self, rgb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            label = self.forward(rgb, noise)
            prob = torch.sigmoid(label)
        return prob

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
