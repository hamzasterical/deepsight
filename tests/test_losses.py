import pytest
import torch

from src.training.losses import CombinedLoss


class TestCombinedLoss:
    def test_forward_returns_scalar(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(4, 1)
        label = torch.randint(0, 2, (4,)).float()
        loss = criterion(pred_label, label)
        assert loss.ndim == 0

    def test_forward_finite(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(4, 1)
        label = torch.randint(0, 2, (4,)).float()
        loss = criterion(pred_label, label)
        assert torch.isfinite(loss)

    def test_differentiable(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(2, 1, requires_grad=True)
        label = torch.randint(0, 2, (2,)).float()
        loss = criterion(pred_label, label)
        loss.backward()
        assert pred_label.grad is not None
