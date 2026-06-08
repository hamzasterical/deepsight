from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

NOISE_BRANCH_NAME: str = "efficientnet_b0"
NOISE_FEATURE_DIM: int = 1280
NOISE_IN_CHANNELS: int = 33


class NoiseBranch(nn.Module):
    def __init__(
        self,
        backbone: str = NOISE_BRANCH_NAME,
        pretrained: bool = False,
        num_classes: int = 0,
        in_channels: int = NOISE_IN_CHANNELS,
        freeze_bn: bool = True,
        return_features: bool = False,
    ):
        super().__init__()
        self.return_features = return_features
        self._feature_maps: Dict[str, torch.Tensor] = {}

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required. Install with: pip install timm"
            ) from e

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            features_only=True,
        )
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        if freeze_bn:
            self._freeze_batchnorm()

    @staticmethod
    def _make_conv_stem(
        in_channels: int,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def _freeze_batchnorm(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        logger.debug("Frozen all BatchNorm layers in noise branch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        self._feature_maps = {str(i): f for i, f in enumerate(features)}
        out = features[-1]
        out = self.conv_head(out)
        out = self.bn2(out)
        if not self.return_features:
            out = out.mean(dim=[2, 3])
        return out

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        return self._feature_maps

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.forward(x)
        return self._feature_maps

    def unfreeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.debug("Unfrozen all noise branch parameters")

    def unfreeze_batchnorm(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
        logger.debug("Unfrozen all BatchNorm layers in noise branch")

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    @property
    def feature_dim(self) -> int:
        return NOISE_FEATURE_DIM

    @property
    def in_channels(self) -> int:
        return NOISE_IN_CHANNELS
