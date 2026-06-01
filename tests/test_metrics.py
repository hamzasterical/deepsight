import numpy as np
import pytest

from src.training.metrics import compute_iou, compute_metrics


class TestComputeIoU:
    def test_perfect_overlap(self):
        pred = np.ones((100, 100), dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        assert compute_iou(pred, gt) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        assert compute_iou(pred, gt) == pytest.approx(0.0)

    def test_half_overlap(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[:, :50] = 1.0
        gt = np.ones((100, 100), dtype=np.float32)
        iou = compute_iou(pred, gt)
        assert 0.3 < iou < 0.6

    def test_both_empty_returns_one(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        gt = np.zeros((100, 100), dtype=np.float32)
        assert compute_iou(pred, gt) == pytest.approx(1.0)

    def test_custom_threshold(self):
        pred = np.full((100, 100), 0.6, dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        assert compute_iou(pred, gt, threshold=0.5) == pytest.approx(1.0)

    def test_custom_threshold_excludes(self):
        pred = np.full((100, 100), 0.4, dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        assert compute_iou(pred, gt, threshold=0.5) == pytest.approx(0.0)


class TestComputeMetrics:
    def test_overall_metrics_present(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.9, 0.8])
        masks = np.random.rand(4, 10, 10)
        gt_masks = np.random.randint(0, 2, (4, 10, 10)).astype(np.float32)
        results = compute_metrics(labels, preds, masks, gt_masks)
        assert "overall" in results
        assert "auc_roc" in results["overall"]
        assert "f1" in results["overall"]
        assert "iou" in results["overall"]

    def test_forgery_type_breakdown(self):
        labels = np.array([0, 1, 0, 1])
        preds = np.array([0.1, 0.9, 0.2, 0.8])
        masks = np.random.rand(4, 10, 10)
        gt_masks = np.random.randint(0, 2, (4, 10, 10)).astype(np.float32)
        ftypes = ["Splicing", "Splicing", "Copy-Move", "Copy-Move"]
        results = compute_metrics(labels, preds, masks, gt_masks, ftypes)
        assert "splicing" in results
        assert "copy_move" in results

    def test_single_class_does_not_crash(self):
        labels = np.array([1, 1, 1])
        preds = np.array([0.9, 0.8, 0.95])
        masks = np.random.rand(3, 10, 10)
        gt_masks = np.random.randint(0, 2, (3, 10, 10)).astype(np.float32)
        results = compute_metrics(labels, preds, masks, gt_masks)
        assert "overall" in results
