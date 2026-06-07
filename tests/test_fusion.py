import pytest
import torch

from src.models.fusion import AdaptiveFusion, FeatureFusion


# ==============================================================
# FIXTURES
# ==============================================================


@pytest.fixture
def rgb_feats_2d():
    return torch.randn(4, 1280)


@pytest.fixture
def noise_feats_2d():
    return torch.randn(4, 1280)


@pytest.fixture
def rgb_feats_4d():
    return torch.randn(4, 1280, 1, 1)


@pytest.fixture
def noise_feats_4d():
    return torch.randn(4, 1280, 1, 1)


# ==============================================================
# FEATURE FUSION — INIT
# ==============================================================


class TestFeatureFusionInit:
    def test_default_init(self):
        fusion = FeatureFusion()
        assert fusion.rgb_dim == 1280
        assert fusion.noise_dim == 1280
        assert fusion.fused_dim == 2560
        assert fusion.hidden_dim == 768

    def test_custom_dims(self):
        fusion = FeatureFusion(rgb_dim=512, noise_dim=512, hidden_dim=256)
        assert fusion.rgb_dim == 512
        assert fusion.noise_dim == 512
        assert fusion.fused_dim == 1024
        assert fusion.hidden_dim == 256

    def test_with_dropout(self):
        fusion = FeatureFusion(dropout=0.3)
        assert fusion is not None

    def test_without_batchnorm(self):
        fusion = FeatureFusion(use_batch_norm=False)
        assert fusion is not None

    def test_fusion_is_sequential(self):
        fusion = FeatureFusion()
        assert isinstance(fusion.fusion, torch.nn.Sequential)

    def test_fusion_layers_count_with_bn(self):
        fusion = FeatureFusion(use_batch_norm=True)
        assert len(fusion.fusion) == 3

    def test_fusion_layers_count_without_bn(self):
        fusion = FeatureFusion(use_batch_norm=False)
        assert len(fusion.fusion) == 2

    def test_fusion_layers_with_dropout(self):
        fusion = FeatureFusion(dropout=0.3, use_batch_norm=True)
        assert len(fusion.fusion) == 4


# ==============================================================
# FEATURE FUSION — FORWARD
# ==============================================================


class TestFeatureFusionForward:
    def test_forward_2d_inputs(self, rgb_feats_2d, noise_feats_2d):
        fusion = FeatureFusion()
        out = fusion(rgb_feats_2d, noise_feats_2d)
        assert out.shape == (4, 768, 1, 1)

    def test_forward_4d_inputs(self, rgb_feats_4d, noise_feats_4d):
        fusion = FeatureFusion()
        out = fusion(rgb_feats_4d, noise_feats_4d)
        assert out.shape == (4, 768, 1, 1)

    def test_forward_mixed_dims(self, rgb_feats_2d, noise_feats_4d):
        fusion = FeatureFusion()
        out = fusion(rgb_feats_2d, noise_feats_4d)
        assert out.shape == (4, 768, 1, 1)

    def test_forward_output_dtype(self, rgb_feats_2d, noise_feats_2d):
        fusion = FeatureFusion()
        out = fusion(rgb_feats_2d, noise_feats_2d)
        assert out.dtype == torch.float32

    def test_forward_flat(self, rgb_feats_2d, noise_feats_2d):
        fusion = FeatureFusion()
        out = fusion.forward_flat(rgb_feats_2d, noise_feats_2d)
        assert out.shape == (4, 768)

    def test_single_sample(self):
        fusion = FeatureFusion()
        fusion.eval()
        rgb = torch.randn(1, 1280)
        noise = torch.randn(1, 1280)
        out = fusion(rgb, noise)
        assert out.shape == (1, 768, 1, 1)

    def test_gradient_flow(self, rgb_feats_2d, noise_feats_2d):
        fusion = FeatureFusion()
        rgb_feats_2d.requires_grad_(True)
        noise_feats_2d.requires_grad_(True)
        out = fusion(rgb_feats_2d, noise_feats_2d)
        loss = out.sum()
        loss.backward()
        assert rgb_feats_2d.grad is not None
        assert noise_feats_2d.grad is not None

    def test_wrong_rgb_dim_raises(self, noise_feats_2d):
        fusion = FeatureFusion()
        bad_rgb = torch.randn(4, 512)
        with pytest.raises(ValueError):
            fusion(bad_rgb, noise_feats_2d)

    def test_wrong_noise_dim_raises(self, rgb_feats_2d):
        fusion = FeatureFusion()
        bad_noise = torch.randn(4, 512)
        with pytest.raises(ValueError):
            fusion(rgb_feats_2d, bad_noise)

    def test_batch_mismatch_raises(self):
        fusion = FeatureFusion()
        rgb = torch.randn(4, 1280)
        noise = torch.randn(8, 1280)
        with pytest.raises(RuntimeError):
            fusion(rgb, noise)


# ==============================================================
# ADAPTIVE FUSION
# ==============================================================


class TestAdaptiveFusionInit:
    def test_default_init(self):
        fusion = AdaptiveFusion()
        assert fusion.rgb_dim == 1280
        assert fusion.noise_dim == 1280
        assert fusion.fused_dim == 2560

    def test_custom_dims(self):
        fusion = AdaptiveFusion(rgb_dim=512, noise_dim=512, hidden_dim=256)
        assert fusion.rgb_dim == 512
        assert fusion.noise_dim == 512
        assert fusion.fused_dim == 1024


class TestAdaptiveFusionForward:
    def test_forward_2d(self, rgb_feats_2d, noise_feats_2d):
        fusion = AdaptiveFusion()
        out = fusion(rgb_feats_2d, noise_feats_2d)
        assert out.shape == (4, 512)

    def test_forward_4d(self, rgb_feats_4d, noise_feats_4d):
        fusion = AdaptiveFusion()
        out = fusion(rgb_feats_4d, noise_feats_4d)
        assert out.shape == (4, 512)

    def test_forward_output_dtype(self, rgb_feats_2d, noise_feats_2d):
        fusion = AdaptiveFusion()
        out = fusion(rgb_feats_2d, noise_feats_2d)
        assert out.dtype == torch.float32

    def test_gradient_flow(self, rgb_feats_2d, noise_feats_2d):
        fusion = AdaptiveFusion()
        rgb_feats_2d.requires_grad_(True)
        noise_feats_2d.requires_grad_(True)
        out = fusion(rgb_feats_2d, noise_feats_2d)
        loss = out.sum()
        loss.backward()
        assert rgb_feats_2d.grad is not None
        assert noise_feats_2d.grad is not None

    def test_gate_outputs_vary_with_input(self):
        fusion = AdaptiveFusion()
        fusion.eval()
        rgb_a = torch.ones(1, 1280)
        noise_a = torch.ones(1, 1280)
        rgb_b = torch.zeros(1, 1280)
        noise_b = torch.ones(1, 1280)
        out_a = fusion(rgb_a, noise_a)
        out_b = fusion(rgb_b, noise_b)
        assert not torch.allclose(out_a, out_b)

    def test_single_sample(self):
        fusion = AdaptiveFusion()
        fusion.eval()
        rgb = torch.randn(1, 1280)
        noise = torch.randn(1, 1280)
        out = fusion(rgb, noise)
        assert out.shape == (1, 512)


# ==============================================================
# EDGE CASES
# ==============================================================


class TestFusionEdgeCases:
    def test_fusion_with_zero_dropout(self):
        fusion = FeatureFusion(dropout=0.0)
        assert len(fusion.fusion) == 3

    def test_fusion_batchnorm_affects_training(self):
        fusion = FeatureFusion(use_batch_norm=True)
        fusion.train()
        for m in fusion.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                assert m.training is True

    def test_fusion_eval_mode(self):
        fusion = FeatureFusion()
        fusion.eval()
        assert not fusion.training

    def test_adaptive_fusion_eval_mode(self):
        fusion = AdaptiveFusion()
        fusion.eval()
        assert not fusion.training

    def test_multiple_forward_passes(self, rgb_feats_2d, noise_feats_2d):
        fusion = FeatureFusion()
        out1 = fusion(rgb_feats_2d, noise_feats_2d)
        out2 = fusion(rgb_feats_2d, noise_feats_2d)
        assert torch.allclose(out1, out2)

    def test_large_batch(self):
        fusion = FeatureFusion()
        rgb = torch.randn(64, 1280)
        noise = torch.randn(64, 1280)
        out = fusion(rgb, noise)
        assert out.shape == (64, 768, 1, 1)
