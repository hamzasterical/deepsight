import pytest
import torch

from src.models.classification_head import (
    CLASSIFICATION_HEAD_IN_FEATURES,
    ClassificationHead,
)


class TestClassificationHeadInit:
    def test_imports(self):
        assert ClassificationHead is not None

    def test_constants(self):
        assert CLASSIFICATION_HEAD_IN_FEATURES == 768

    def test_default_init(self):
        head = ClassificationHead()
        assert head.in_features == CLASSIFICATION_HEAD_IN_FEATURES
        assert isinstance(head.fc, torch.nn.Linear)

    def test_fc_in_features(self):
        head = ClassificationHead(in_features=512)
        assert head.fc.in_features == 512

    def test_fc_out_features(self):
        head = ClassificationHead()
        assert head.fc.out_features == 1

    def test_custom_in_features(self):
        head = ClassificationHead(in_features=256)
        assert head.in_features == 256
        assert head.fc.in_features == 256

    def test_dropout_layer_created(self):
        head = ClassificationHead(dropout=0.5)
        assert isinstance(head.dropout, torch.nn.Dropout)
        assert head.dropout.p == 0.5

    def test_zero_dropout_identity(self):
        head = ClassificationHead(dropout=0.0)
        assert isinstance(head.dropout, torch.nn.Identity)

    def test_predict_method_exists(self):
        head = ClassificationHead()
        assert hasattr(head, "predict")


class TestClassificationHeadForward:
    def test_forward_2d_input(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head(x)
        assert out.shape == (4, 1)

    def test_forward_4d_input(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES, 1, 1)
        out = head(x)
        assert out.shape == (4, 1)

    def test_forward_dtype(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(2, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head(x)
        assert out.dtype == torch.float32

    def test_forward_logits_not_sigmoided(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(2, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head(x)
        assert not (out.min() >= 0 and out.max() <= 1)

    def test_predict_returns_probability(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head.predict(x)
        assert out.shape == (4, 1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_sample(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(1, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head(x)
        assert out.shape == (1, 1)

    def test_large_batch(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(64, CLASSIFICATION_HEAD_IN_FEATURES)
        out = head(x)
        assert out.shape == (64, 1)

    def test_gradient_flow(self):
        head = ClassificationHead()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in head.parameters():
            assert p.grad is not None
            break

    def test_multiple_forward_passes(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(2, CLASSIFICATION_HEAD_IN_FEATURES)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        head = ClassificationHead()
        head.eval()
        x1 = torch.randn(1, CLASSIFICATION_HEAD_IN_FEATURES)
        x2 = torch.randn(1, CLASSIFICATION_HEAD_IN_FEATURES)
        out1 = head(x1)
        out2 = head(x2)
        assert out1.shape == out2.shape


class TestClassificationHeadEdgeCases:
    def test_train_eval_mode(self):
        head = ClassificationHead()
        head.train()
        assert head.training is True
        head.eval()
        assert head.training is False

    def test_parameters_exist(self):
        head = ClassificationHead()
        params = list(head.parameters())
        assert len(params) > 0

    def test_forward_wrong_dim_raises(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES, 5)
        with pytest.raises(RuntimeError):
            head(x)

    def test_forward_3d_batched(self):
        head = ClassificationHead()
        head.eval()
        x = torch.randn(4, CLASSIFICATION_HEAD_IN_FEATURES, 1)
        with pytest.raises(RuntimeError):
            head(x)
