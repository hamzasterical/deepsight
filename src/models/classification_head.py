from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_HEAD_IN_FEATURES: int = 768


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int = CLASSIFICATION_HEAD_IN_FEATURES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, 1)

        logger.debug(
            "ClassificationHead(in_features=%d, dropout=%.2f)",
            in_features,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
