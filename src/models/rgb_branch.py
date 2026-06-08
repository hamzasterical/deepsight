from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

RGB_BRANCH_NAME: str = "efficientnet_b0"
RGB_FEATURE_DIM: int = 1280


class RGBBranch(nn.Module):
    def __init__(
        self,
        backbone: str = RGB_BRANCH_NAME,
        pretrained: bool = True,
        num_classes: int = 0,
        in_channels: int = 3,
        freeze_stem: bool = False,
        freeze_bn: bool = True,
        freeze_layers: Optional[List[int]] = None,
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

        if freeze_stem:
            self._freeze_stem()

        if freeze_bn:
            self._freeze_batchnorm()

        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)

    def _freeze_stem(self) -> None:
        if hasattr(self.backbone, "conv_stem"):
            for p in self.backbone.conv_stem.parameters():
                p.requires_grad = False
            logger.debug("Frozen conv_stem")
        if hasattr(self.backbone, "bn1"):
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
            logger.debug("Frozen bn1")

    def _freeze_batchnorm(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        logger.debug("Frozen all BatchNorm layers")

    def _freeze_layers(self, layers: List[int]) -> None:
        frozen = 0
        for i, (name, child) in enumerate(self.backbone.named_children()):
            if i in layers or any(str(l) in name for l in layers):
                for p in child.parameters():
                    p.requires_grad = False
                frozen += 1
        logger.debug("Frozen %d child modules by indices %s", frozen, layers)

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
        logger.debug("Unfrozen all parameters")

    def unfreeze_batchnorm(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
        logger.debug("Unfrozen all BatchNorm layers")

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    @property
    def feature_dim(self) -> int:
        return RGB_FEATURE_DIM

    def load_pretrained(self, strict: bool = True) -> None:
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required to load pretrained weights")

        state_dict = timm.create_model(
            RGB_BRANCH_NAME, pretrained=True, num_classes=0
        ).state_dict()
        self.backbone.load_state_dict(state_dict, strict=strict)
        logger.debug("Loaded pretrained EfficientNet-B0 weights")
