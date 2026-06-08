from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.skip_channels = skip_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        conv_in = out_channels + skip_channels
        layers = [
            nn.Conv2d(
                conv_in, out_channels, kernel_size=3, padding=1,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is None and self.skip_channels > 0:
            skip = torch.zeros(
                x.size(0), self.skip_channels, *x.shape[-2:],
                device=x.device, dtype=x.dtype,
            )
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode="bilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        fused_channels: int = 768,
        decoder_channels: Optional[List[int]] = None,
        skip_channels: Optional[List[int]] = None,
        output_channels: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        if skip_channels is None:
            skip_channels = [112, 40, 24, 16]

        if len(decoder_channels) < len(skip_channels) + 1:
            raise ValueError(
                f"decoder_channels length ({len(decoder_channels)}) must be at least "
                f"skip_channels length ({len(skip_channels)}) + 1"
            )

        self.fused_channels = fused_channels
        self.decoder_channels = decoder_channels
        self.skip_channels = skip_channels

        self.blocks = nn.ModuleList()
        in_ch = fused_channels

        for i, out_ch in enumerate(decoder_channels):
            sk_ch = skip_channels[i] if i < len(skip_channels) else 0
            self.blocks.append(DecoderBlock(in_ch, out_ch, sk_ch, use_batchnorm))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=1)

        logger.debug(
            "SegmentationHead(fused=%d, decoder=%s, skips=%s, output=%d, bn=%s)",
            fused_channels,
            decoder_channels,
            skip_channels,
            output_channels,
            use_batchnorm,
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = fused_features
        for i, block in enumerate(self.blocks):
            skip = (
                skip_features[i]
                if (skip_features is not None and i < len(skip_features))
                else None
            )
            x = block(x, skip)

        x = self.final_conv(x)
        return torch.sigmoid(x)
