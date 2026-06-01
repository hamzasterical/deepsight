import pytest
import torch

from src.models.dual_branch import DualBranchModel


class TestDualBranchModelInit:
    def test_default_init(self):
        model = DualBranchModel()
        assert model.rgb_branch is not None
        assert model.noise_branch is not None
        assert model.fusion is not None
        assert model.classification_head is not None
        assert model.segmentation_head is not None
        assert model.feature_dim == 1280
        assert model.hidden_dim == 512

    def test_custom_config(self):
        model = DualBranchModel(feature_dim=640, hidden_dim=256, dropout=0.1)
        assert model.hidden_dim == 256

    def test_has_device_property(self):
        model = DualBranchModel()
        assert hasattr(model, "device")
        assert isinstance(model.device, torch.device)

    def test_predict_method_exists(self):
        model = DualBranchModel()
        assert hasattr(model, "predict")


class TestDualBranchModelForward:
    def test_forward_shape(self):
        model = DualBranchModel()
        model.eval()
        rgb = torch.randn(1, 3, 224, 224)
        noise = torch.randn(1, 33, 224, 224)
        label, mask = model(rgb, noise)
        assert label.shape == (1, 1)
        assert mask.shape == (1, 1, 224, 224)

    def test_forward_batch(self):
        model = DualBranchModel()
        model.eval()
        rgb = torch.randn(4, 3, 224, 224)
        noise = torch.randn(4, 33, 224, 224)
        label, mask = model(rgb, noise)
        assert label.shape == (4, 1)
        assert mask.shape == (4, 1, 224, 224)

    def test_forward_dtype(self):
        model = DualBranchModel()
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        noise = torch.randn(2, 33, 224, 224)
        label, mask = model(rgb, noise)
        assert label.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_mask_output_is_probability(self):
        model = DualBranchModel()
        model.eval()
        rgb = torch.randn(1, 3, 224, 224)
        noise = torch.randn(1, 33, 224, 224)
        _, mask = model(rgb, noise)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestDualBranchModelPredict:
    def test_predict_returns_sigmoided_logits(self):
        model = DualBranchModel()
        rgb = torch.randn(1, 3, 224, 224)
        noise = torch.randn(1, 33, 224, 224)
        prob, mask = model.predict(rgb, noise)
        assert prob.shape == (1, 1)
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0
        assert mask.shape == (1, 1, 224, 224)

    def test_predict_uses_no_grad(self):
        model = DualBranchModel()
        rgb = torch.randn(1, 3, 224, 224, requires_grad=True)
        noise = torch.randn(1, 33, 224, 224)
        prob, mask = model.predict(rgb, noise)
        assert rgb.grad is None


class TestDualBranchModelEdgeCases:
    def test_train_eval_mode(self):
        model = DualBranchModel()
        model.train()
        assert model.training is True
        model.eval()
        assert model.training is False

    def test_parameters_exist(self):
        model = DualBranchModel()
        params = list(model.parameters())
        assert len(params) > 0
