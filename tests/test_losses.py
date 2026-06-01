import pytest
import torch

from src.training.losses import CombinedLoss, DiceLoss


class TestDiceLoss:
    def test_perfect_prediction(self):
        criterion = DiceLoss()
        pred = torch.ones(2, 1, 10, 10)
        target = torch.ones(2, 1, 10, 10)
        loss = criterion(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=0.01)

    def test_no_overlap(self):
        criterion = DiceLoss(smooth=0.0)
        pred = torch.zeros(2, 1, 10, 10)
        target = torch.ones(2, 1, 10, 10)
        loss = criterion(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=0.01)

    def test_half_overlap(self):
        criterion = DiceLoss(smooth=0.0)
        pred = torch.zeros(1, 1, 10, 10)
        pred[:, :, :5, :] = 1.0
        target = torch.ones(1, 1, 10, 10)
        loss = criterion(pred, target)
        assert 0.3 < loss.item() < 0.7

    def test_smooth_prevents_division_by_zero(self):
        criterion = DiceLoss(smooth=1.0)
        pred = torch.zeros(1, 1, 10, 10)
        target = torch.zeros(1, 1, 10, 10)
        loss = criterion(pred, target)
        assert torch.isfinite(loss)

    def test_differentiable(self):
        criterion = DiceLoss()
        pred = torch.randn(2, 1, 10, 10, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 10, 10)).float()
        loss = criterion(torch.sigmoid(pred), target)
        loss.backward()
        assert pred.grad is not None


class TestCombinedLoss:
    def test_forward_returns_scalar(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(4, 1)
        label = torch.randint(0, 2, (4, 1)).float()
        pred_mask = torch.sigmoid(torch.randn(4, 1, 10, 10))
        mask = torch.randint(0, 2, (4, 1, 10, 10)).float()
        loss = criterion(pred_label, label, pred_mask, mask)
        assert loss.ndim == 0

    def test_forward_finite(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(4, 1)
        label = torch.randint(0, 2, (4, 1)).float()
        pred_mask = torch.sigmoid(torch.randn(4, 1, 10, 10))
        mask = torch.randint(0, 2, (4, 1, 10, 10)).float()
        loss = criterion(pred_label, label, pred_mask, mask)
        assert torch.isfinite(loss)

    def test_separate_returns_dict(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(4, 1)
        label = torch.randint(0, 2, (4, 1)).float()
        pred_mask = torch.sigmoid(torch.randn(4, 1, 10, 10))
        mask = torch.randint(0, 2, (4, 1, 10, 10)).float()
        result = criterion.separate(pred_label, label, pred_mask, mask)
        assert "label_loss" in result
        assert "mask_loss" in result
        assert "total_loss" in result

    def test_custom_dice_weight(self):
        criterion = CombinedLoss(dice_weight=1.0)
        pred_label = torch.randn(2, 1)
        label = torch.randint(0, 2, (2, 1)).float()
        pred_mask = torch.sigmoid(torch.randn(2, 1, 10, 10))
        mask = torch.randint(0, 2, (2, 1, 10, 10)).float()
        result = criterion.separate(pred_label, label, pred_mask, mask)
        expected = result["label_loss"] + 1.0 * result["mask_loss"]
        assert abs(result["total_loss"] - expected) < 1e-6

    def test_differentiable(self):
        criterion = CombinedLoss()
        pred_label = torch.randn(2, 1, requires_grad=True)
        label = torch.randint(0, 2, (2, 1)).float()
        pred_mask = torch.sigmoid(torch.randn(2, 1, 10, 10, requires_grad=True))
        mask = torch.randint(0, 2, (2, 1, 10, 10)).float()
        loss = criterion(pred_label, label, pred_mask, mask)
        loss.backward()
        assert pred_label.grad is not None
        assert pred_mask.grad is not None
