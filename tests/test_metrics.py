import numpy as np
import pytest

from src.training.metrics import compute_metrics


class TestComputeMetrics:
    def test_overall_metrics_present(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.9, 0.8])
        results = compute_metrics(labels, preds)
        assert "overall" in results
        assert "auc_roc" in results["overall"]
        assert "f1" in results["overall"]
        assert "accuracy" in results["overall"]

    def test_forgery_type_breakdown(self):
        labels = np.array([0, 1, 0, 1])
        preds = np.array([0.1, 0.9, 0.2, 0.8])
        ftypes = ["Splicing", "Splicing", "Copy-Move", "Copy-Move"]
        results = compute_metrics(labels, preds, ftypes)
        assert "splicing" in results
        assert "copy_move" in results

    def test_single_class_does_not_crash(self):
        labels = np.array([1, 1, 1])
        preds = np.array([0.9, 0.8, 0.95])
        results = compute_metrics(labels, preds)
        assert "overall" in results
