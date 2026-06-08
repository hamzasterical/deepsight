from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureFusion(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 1280,
        noise_dim: int = 1280,
        hidden_dim: int = 768,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.rgb_dim = rgb_dim
        self.noise_dim = noise_dim
        self.fused_dim = rgb_dim + noise_dim
        self.hidden_dim = hidden_dim

        layers: list = []
        layers.append(
            nn.Conv2d(self.fused_dim, hidden_dim, kernel_size=1, bias=False)
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.fusion = nn.Sequential(*layers)

        logger.debug(
            "FeatureFusion(%d + %d = %d -> %d, dropout=%s, bn=%s)",
            rgb_dim, noise_dim, self.fused_dim, hidden_dim, dropout, use_batch_norm,
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        noise_features: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_features.dim() == 2:
            rgb_features = rgb_features.unsqueeze(-1).unsqueeze(-1)
        if noise_features.dim() == 2:
            noise_features = noise_features.unsqueeze(-1).unsqueeze(-1)

        if rgb_features.shape[1] != self.rgb_dim:
            raise ValueError(
                f"Expected rgb_features dim={self.rgb_dim}, got {rgb_features.shape[1]}"
            )
        if noise_features.shape[1] != self.noise_dim:
            raise ValueError(
                f"Expected noise_features dim={self.noise_dim}, got {noise_features.shape[1]}"
            )

        combined = torch.cat([rgb_features, noise_features], dim=1)
        fused = self.fusion(combined)
        return fused

    def forward_flat(
        self,
        rgb_features: torch.Tensor,
        noise_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.forward(rgb_features, noise_features)
        return fused.view(fused.size(0), -1)


class AdaptiveFusion(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 1280,
        noise_dim: int = 1280,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.rgb_dim = rgb_dim
        self.noise_dim = noise_dim
        self.fused_dim = rgb_dim + noise_dim

        self.rgb_gate = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.noise_gate = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        noise_features: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_features.dim() == 4:
            rgb_features = rgb_features.view(rgb_features.size(0), -1)
        if noise_features.dim() == 4:
            noise_features = noise_features.view(noise_features.size(0), -1)

        rgb_weight = self.rgb_gate(rgb_features)
        noise_weight = self.noise_gate(noise_features)

        gated_rgb = rgb_features * rgb_weight
        gated_noise = noise_features * noise_weight

        combined = torch.cat([gated_rgb, gated_noise], dim=1)
        return self.project(combined)
