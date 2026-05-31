import pytest

from src.models.noise_branch import NOISE_FEATURE_DIM, NOISE_IN_CHANNELS, NoiseBranch
from src.models.rgb_branch import RGB_FEATURE_DIM, RGBBranch


class TestRGBBranchInit:
    def test_imports(self):
        from src.models.rgb_branch import RGBBranch
        assert RGBBranch is not None

    def test_constants(self):
        assert RGB_FEATURE_DIM == 1280

    def test_init_defaults(self):
        branch = RGBBranch(pretrained=False)
        assert branch.feature_dim == RGB_FEATURE_DIM
        assert branch.return_features is False

    def test_init_return_features(self):
        branch = RGBBranch(pretrained=False, return_features=True)
        assert branch.return_features is True

    def test_init_freeze_stem(self):
        branch = RGBBranch(pretrained=False, freeze_stem=True)

    def test_init_freeze_bn(self):
        branch = RGBBranch(pretrained=False, freeze_bn=True)

    def test_init_freeze_layers(self):
        branch = RGBBranch(pretrained=False, freeze_layers=[0])

    def test_init_custom_in_channels(self):
        branch = RGBBranch(pretrained=False, in_channels=1)

    def test_init_pretrained_false_no_error(self):
        branch = RGBBranch(pretrained=False)
        assert branch is not None

    def test_has_backbone_attribute(self):
        branch = RGBBranch(pretrained=False)
        assert hasattr(branch, "backbone")

    def test_backbone_is_module(self):
        branch = RGBBranch(pretrained=False)
        import torch.nn as nn
        assert isinstance(branch.backbone, nn.Module)

    def test_unfreeze_all(self):
        branch = RGBBranch(pretrained=False)
        branch.unfreeze_all()
        for p in branch.backbone.parameters():
            assert p.requires_grad is True

    def test_unfreeze_batchnorm(self):
        branch = RGBBranch(pretrained=False, freeze_bn=True)
        branch.unfreeze_batchnorm()
        import torch.nn as nn
        for m in branch.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert m.training is True

    def test_load_pretrained_method_exists(self):
        branch = RGBBranch(pretrained=False)
        assert hasattr(branch, "load_pretrained")

    def test_get_feature_maps_empty_initially(self):
        branch = RGBBranch(pretrained=False)
        assert branch.get_feature_maps() == {}


class TestRGBBranchForward:
    def test_forward_shape(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = branch(x)
        assert out.shape == (1, RGB_FEATURE_DIM)

    def test_forward_batch(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = branch(x)
        assert out.shape == (4, RGB_FEATURE_DIM)

    def test_forward_float32(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = branch(x)
        assert out.dtype == torch.float32

    def test_forward_return_features(self):
        import torch
        branch = RGBBranch(pretrained=False, return_features=True)
        x = torch.randn(1, 3, 224, 224)
        out = branch(x)
        assert out.dim() == 4

    def test_intermediate_features(self):
        import torch
        branch = RGBBranch(pretrained=False, return_features=True)
        x = torch.randn(1, 3, 224, 224)
        branch(x)
        features = branch.get_feature_maps()
        assert len(features) > 0

    def test_forward_small_input_shape(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(1, 3, 128, 128)
        out = branch(x)
        assert out.shape == (1, RGB_FEATURE_DIM)

    def test_gradients_flow(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = branch(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestNoiseBranchInit:
    def test_imports(self):
        from src.models.noise_branch import NoiseBranch
        assert NoiseBranch is not None

    def test_constants(self):
        assert NOISE_IN_CHANNELS == 33
        assert NOISE_FEATURE_DIM == 1280

    def test_init_defaults(self):
        branch = NoiseBranch()
        assert branch.feature_dim == NOISE_FEATURE_DIM
        assert branch.in_channels == NOISE_IN_CHANNELS
        assert branch.return_features is False

    def test_init_return_features(self):
        branch = NoiseBranch(return_features=True)
        assert branch.return_features is True

    def test_init_freeze_bn(self):
        branch = NoiseBranch(freeze_bn=True)

    def test_init_pretrained_false_by_default(self):
        branch = NoiseBranch()
        assert branch is not None

    def test_has_backbone_attribute(self):
        branch = NoiseBranch()
        assert hasattr(branch, "backbone")

    def test_unfreeze_all(self):
        branch = NoiseBranch()
        branch.unfreeze_all()
        import torch.nn as nn
        for p in branch.backbone.parameters():
            assert p.requires_grad is True

    def test_unfreeze_batchnorm(self):
        branch = NoiseBranch(freeze_bn=True)
        branch.unfreeze_batchnorm()
        import torch.nn as nn
        for m in branch.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert m.training is True

    def test_get_feature_maps_empty_initially(self):
        branch = NoiseBranch()
        assert branch.get_feature_maps() == {}

    def test_make_conv_stem_creates_sequential(self):
        import torch.nn as nn
        stem = NoiseBranch._make_conv_stem(33, 32)
        assert isinstance(stem, nn.Sequential)
        assert len(stem) == 3


class TestNoiseBranchForward:
    def test_forward_shape(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(1, NOISE_IN_CHANNELS, 224, 224)
        out = branch(x)
        assert out.shape == (1, NOISE_FEATURE_DIM)

    def test_forward_batch(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(4, NOISE_IN_CHANNELS, 224, 224)
        out = branch(x)
        assert out.shape == (4, NOISE_FEATURE_DIM)

    def test_forward_float32(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(2, NOISE_IN_CHANNELS, 224, 224)
        out = branch(x)
        assert out.dtype == torch.float32

    def test_forward_return_features(self):
        import torch
        branch = NoiseBranch(return_features=True)
        x = torch.randn(1, NOISE_IN_CHANNELS, 224, 224)
        out = branch(x)
        assert out.dim() == 4

    def test_intermediate_features(self):
        import torch
        branch = NoiseBranch(return_features=True)
        x = torch.randn(1, NOISE_IN_CHANNELS, 224, 224)
        branch(x)
        features = branch.get_feature_maps()
        assert len(features) > 0

    def test_gradients_flow(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(1, NOISE_IN_CHANNELS, 224, 224, requires_grad=True)
        out = branch(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_wrong_input_channels_errors(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(Exception):
            branch(x)


class TestRGBBranchEdgeCases:
    def test_init_with_all_freezes(self):
        branch = RGBBranch(
            pretrained=False,
            freeze_stem=True,
            freeze_bn=True,
            freeze_layers=[0, 1],
        )
        import torch
        x = torch.randn(1, 3, 224, 224)
        out = branch(x)
        assert out.shape == (1, RGB_FEATURE_DIM)

    def test_train_mode_toggle(self):
        branch = RGBBranch(pretrained=False)
        branch.train(True)
        assert branch.training is True
        branch.train(False)
        assert branch.training is False

    def test_backbone_parameters_exist(self):
        branch = RGBBranch(pretrained=False)
        params = list(branch.parameters())
        assert len(params) > 0

    def test_multiple_forward_passes(self):
        import torch
        branch = RGBBranch(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out1 = branch(x)
        out2 = branch(x)
        import torch
        assert torch.allclose(out1, out2)


class TestNoiseBranchEdgeCases:
    def test_train_mode_toggle(self):
        branch = NoiseBranch()
        branch.train(True)
        assert branch.training is True
        branch.train(False)
        assert branch.training is False

    def test_backbone_parameters_exist(self):
        branch = NoiseBranch()
        params = list(branch.parameters())
        assert len(params) > 0

    def test_multiple_forward_passes(self):
        import torch
        branch = NoiseBranch()
        x = torch.randn(1, NOISE_IN_CHANNELS, 224, 224)
        out1 = branch(x)
        out2 = branch(x)
        import torch
        assert torch.allclose(out1, out2)
