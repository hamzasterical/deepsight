import numpy as np
import pytest
import torch

from src.preprocessing.srm_filters import (
    SRM_FILTER_COUNT,
    SRM_KERNEL_SIZE,
    SRM_PADDING,
    SRMFilterLayer,
    extract_srm_noise,
    extract_srm_noise_batch,
    get_srm_filters,
    get_srm_weight_tensor,
)


# ==============================================================
# FILTER GENERATION
# ==============================================================


class TestGetSRMFilters:
    def test_returns_correct_number_of_filters(self):
        filters = get_srm_filters()
        assert filters.shape[0] == SRM_FILTER_COUNT

    def test_returns_5x5_kernels(self):
        filters = get_srm_filters()
        assert filters.shape[1:] == (SRM_KERNEL_SIZE, SRM_KERNEL_SIZE)

    def test_returns_float32(self):
        filters = get_srm_filters()
        assert filters.dtype == np.float32

    def test_each_filter_is_normalized(self):
        filters = get_srm_filters()
        for i in range(SRM_FILTER_COUNT):
            norm = np.sqrt((filters[i] ** 2).sum())
            assert abs(norm - 1.0) < 1e-4, f"Filter {i} L2 norm {norm} != 1"

    def test_filters_are_not_all_zero(self):
        filters = get_srm_filters()
        for i in range(SRM_FILTER_COUNT):
            assert not np.allclose(filters[i], 0), f"Filter {i} is all zeros"

    def test_filters_have_near_zero_sum(self):
        filters = get_srm_filters()
        high_sum_count = 0
        for i in range(SRM_FILTER_COUNT):
            total = filters[i].sum()
            if abs(total) >= 0.5:
                high_sum_count += 1
        assert high_sum_count <= 8, f"{high_sum_count} filters have sum >= 0.5"

    def test_filters_diverse(self):
        filters = get_srm_filters()
        pairs = [(0, 1), (5, 10), (15, 20)]
        for i, j in pairs:
            corr = np.corrcoef(filters[i].ravel(), filters[j].ravel())[0, 1]
            assert abs(corr) < 0.99, f"Filters {i} and {j} are too similar (corr={corr})"


# ==============================================================
# WEIGHT TENSOR
# ==============================================================


class TestGetSRMWeightTensor:
    def test_returns_tensor(self):
        weight = get_srm_weight_tensor()
        assert isinstance(weight, torch.Tensor)

    def test_correct_shape(self):
        weight = get_srm_weight_tensor()
        assert weight.shape == (SRM_FILTER_COUNT, 3, SRM_KERNEL_SIZE, SRM_KERNEL_SIZE)

    def test_all_channels_identical(self):
        weight = get_srm_weight_tensor()
        for i in range(SRM_FILTER_COUNT):
            assert torch.allclose(weight[i, 0], weight[i, 1])
            assert torch.allclose(weight[i, 1], weight[i, 2])

    def test_requires_grad_false(self):
        weight = get_srm_weight_tensor()
        assert weight.requires_grad is False

    def test_dtype_is_float32(self):
        weight = get_srm_weight_tensor()
        assert weight.dtype == torch.float32


# ==============================================================
# SRM FILTER LAYER
# ==============================================================


class TestSRMFilterLayer:
    def test_initialization(self):
        layer = SRMFilterLayer()
        assert isinstance(layer.conv, torch.nn.Conv2d)
        assert layer.conv.in_channels == 3
        assert layer.conv.out_channels == SRM_FILTER_COUNT
        assert layer.conv.kernel_size == (SRM_KERNEL_SIZE, SRM_KERNEL_SIZE)
        assert layer.conv.padding == (SRM_PADDING, SRM_PADDING)
        assert layer.conv.bias is None

    def test_weights_are_frozen(self):
        layer = SRMFilterLayer()
        for p in layer.parameters():
            assert p.requires_grad is False

    def test_weights_are_srm_filters(self):
        layer = SRMFilterLayer()
        expected = get_srm_weight_tensor()
        assert torch.allclose(layer.conv.weight, expected)

    def test_eval_mode(self):
        layer = SRMFilterLayer()
        layer.eval()
        assert not layer.training

    def test_forward_output_shape(self):
        layer = SRMFilterLayer()
        x = torch.randn(1, 3, 224, 224)
        out = layer(x)
        assert out.shape == (1, SRM_FILTER_COUNT, 224, 224)

    def test_forward_output_dtype(self):
        layer = SRMFilterLayer()
        x = torch.randn(2, 3, 64, 64)
        out = layer(x)
        assert out.dtype == torch.float32

    def test_batch_independence(self):
        layer = SRMFilterLayer()
        x1 = torch.randn(1, 3, 32, 32)
        x2 = torch.randn(1, 3, 32, 32)
        x = torch.cat([x1, x2], dim=0)
        out = layer(x)
        assert out.shape[0] == 2
        assert not torch.allclose(out[0], out[1])

    def test_forward_uses_no_grad(self):
        layer = SRMFilterLayer()
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)
            out = layer(x)
        assert out.requires_grad is False

    def test_multiple_channels_preserved(self):
        layer = SRMFilterLayer()
        x = torch.randn(1, 3, 128, 128)
        out = layer(x)
        assert out.shape[1] == SRM_FILTER_COUNT
        zero = torch.zeros_like(out[0, 0])
        for c in range(SRM_FILTER_COUNT):
            assert not torch.allclose(out[0, c], zero)

    def test_identity_image_has_low_high_freq(self):
        layer = SRMFilterLayer()
        x = torch.ones(1, 3, 64, 64)
        out = layer(x)
        max_vals = out.abs().max(dim=3)[0].max(dim=2)[0]
        for c in range(SRM_FILTER_COUNT):
            if max_vals[0, c] > 0.01:
                break
        else:
            pytest.fail("All filter outputs are near zero for constant input")

    def test_noise_input_produces_response(self):
        layer = SRMFilterLayer()
        x = torch.randn(1, 3, 64, 64) * 20
        out = layer(x)
        assert out.abs().mean().item() > 0.1


# ==============================================================
# EXTRACT SRM NOISE
# ==============================================================


class TestExtractSRMNoise:
    def test_extract_from_rgb_image(self):
        image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        noise = extract_srm_noise(image)
        assert noise.shape == (100, 100, SRM_FILTER_COUNT)
        assert noise.dtype == np.float32

    def test_extract_from_float_image(self):
        image = np.random.rand(64, 64, 3).astype(np.float32) * 255
        noise = extract_srm_noise(image)
        assert noise.shape == (64, 64, SRM_FILTER_COUNT)

    def test_constant_image_output(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        noise = extract_srm_noise(image)
        assert noise.shape == (32, 32, SRM_FILTER_COUNT)

    def test_raises_on_grayscale(self):
        image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_srm_noise(image)

    def test_raises_on_wrong_dims(self):
        image = np.random.randint(0, 256, size=(100, 100, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_srm_noise(image)

    def test_noise_variance_changes_with_input(self):
        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        noisy = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        noise_flat = extract_srm_noise(flat)
        noise_noisy = extract_srm_noise(noisy)
        assert noise_noisy.std() > noise_flat.std()

    def test_output_is_not_all_zero(self):
        image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        noise = extract_srm_noise(image)
        assert noise.std() > 0


# ==============================================================
# EXTRACT SRM NOISE BATCH
# ==============================================================


class TestExtractSRMNoiseBatch:
    def test_batch_output_shape(self):
        images = np.random.randint(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)
        noise = extract_srm_noise_batch(images)
        assert noise.shape == (4, 64, 64, SRM_FILTER_COUNT)

    def test_batch_uint8_conversion(self):
        images = np.random.randint(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        noise = extract_srm_noise_batch(images)
        assert noise.dtype == np.float32

    def test_batch_consistency_with_single(self):
        image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        single = extract_srm_noise(image)
        batch = extract_srm_noise_batch(image[np.newaxis, ...])
        assert np.allclose(single, batch[0], atol=1e-5)

    def test_batch_raises_on_wrong_ndim(self):
        image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_srm_noise_batch(image)

    def test_batch_different_sizes(self):
        batch_2 = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
        batch_8 = np.random.randint(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
        n2 = extract_srm_noise_batch(batch_2)
        n8 = extract_srm_noise_batch(batch_8)
        assert n2.shape[0] == 2
        assert n8.shape[0] == 8


# ==============================================================
# EDGE CASES
# ==============================================================


class TestEdgeCases:
    def test_minimal_image_size(self):
        image = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
        noise = extract_srm_noise(image)
        assert noise.shape == (5, 5, SRM_FILTER_COUNT)

    def test_srm_layer_with_batch_of_one(self):
        layer = SRMFilterLayer()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (1, SRM_FILTER_COUNT, 224, 224)

    def test_all_channels_same_filter_applied(self):
        layer = SRMFilterLayer()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = layer(x)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not torch.allclose(out[0, i], out[0, j])

    def test_layer_is_sequential_compatible(self):
        layer = SRMFilterLayer()
        seq = torch.nn.Sequential(layer)
        x = torch.randn(1, 3, 32, 32)
        out = seq(x)
        assert out.shape == (1, SRM_FILTER_COUNT, 32, 32)

    def test_extract_noise_is_deterministic(self):
        image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        n1 = extract_srm_noise(image)
        n2 = extract_srm_noise(image)
        assert np.allclose(n1, n2)
