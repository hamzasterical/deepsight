import base64

import cv2
import numpy as np
import pytest

from src.inference.postprocess import (
    compute_forged_area_percentage,
    encode_image_base64,
    generate_heatmap_overlay,
    generate_red_alpha_overlay,
    postprocess,
    upscale_mask,
)


@pytest.fixture
def orig_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(400, 600, 3), dtype=np.uint8)


@pytest.fixture
def small_mask() -> np.ndarray:
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[50:100, 50:100] = 1.0
    return mask


@pytest.fixture
def full_mask() -> np.ndarray:
    mask = np.ones((224, 224), dtype=np.float32)
    return mask


@pytest.fixture
def half_mask() -> np.ndarray:
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[:, :112] = 1.0
    return mask


# ==============================================================
# UPSCALE MASK
# ==============================================================


class TestUpscaleMask:
    def test_upscale_to_larger(self, small_mask):
        result = upscale_mask(small_mask, orig_width=800, orig_height=600)
        assert result.shape == (600, 800)
        assert result.dtype == np.float32

    def test_upscale_to_smaller(self, small_mask):
        result = upscale_mask(small_mask, orig_width=100, orig_height=100)
        assert result.shape == (100, 100)

    def test_upscale_same_size(self):
        mask = np.random.rand(224, 224).astype(np.float32)
        result = upscale_mask(mask, orig_width=224, orig_height=224)
        assert result.shape == (224, 224)
        assert np.allclose(result, mask)

    def test_upscale_3d_input(self, small_mask):
        mask_3d = small_mask[np.newaxis, ...]
        result = upscale_mask(mask_3d, orig_width=600, orig_height=400)
        assert result.shape == (400, 600)

    def test_values_preserved_in_range(self, small_mask):
        result = upscale_mask(small_mask, orig_width=600, orig_height=400)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_upscale_with_different_interpolation(self, small_mask):
        result = upscale_mask(small_mask, orig_width=600, orig_height=400, interpolation=cv2.INTER_CUBIC)
        assert result.shape == (400, 600)


# ==============================================================
# COMPUTE FORGED AREA PERCENTAGE
# ==============================================================


class TestComputeForgedAreaPercentage:
    def test_no_forged_pixels(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        assert compute_forged_area_percentage(mask) == 0.0

    def test_all_forged_pixels(self):
        mask = np.ones((100, 100), dtype=np.float32)
        assert compute_forged_area_percentage(mask) == 100.0

    def test_half_forged(self, half_mask):
        pct = compute_forged_area_percentage(half_mask)
        assert pct == 50.0

    def test_custom_threshold(self):
        mask = np.full((100, 100), 0.3, dtype=np.float32)
        pct = compute_forged_area_percentage(mask, threshold=0.25)
        assert pct == 100.0

    def test_no_pixels_returns_zero(self):
        mask = np.zeros((0, 100), dtype=np.float32)
        assert compute_forged_area_percentage(mask) == 0.0

    def test_quarter_forged(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[:100, :100] = 1.0
        pct = compute_forged_area_percentage(mask)
        assert pct == 25.0

    def test_edge_values(self):
        mask = np.full((100, 100), 0.5, dtype=np.float32)
        pct = compute_forged_area_percentage(mask, threshold=0.5)
        assert pct == 0.0

    def test_result_is_rounded(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[0, 0] = 1.0
        mask[0, 1] = 1.0
        mask[0, 2] = 1.0
        pct = compute_forged_area_percentage(mask)
        assert isinstance(pct, float)


# ==============================================================
# GENERATE HEATMAP OVERLAY
# ==============================================================


class TestGenerateHeatmapOverlay:
    def test_output_shape(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_heatmap_overlay(orig_image, mask_upscaled)
        assert overlay.shape == orig_image.shape

    def test_output_dtype(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_heatmap_overlay(orig_image, mask_upscaled)
        assert overlay.dtype == np.uint8

    def test_output_values_in_range(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_heatmap_overlay(orig_image, mask_upscaled)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_default_alpha(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_heatmap_overlay(orig_image, mask_upscaled)
        assert overlay is not None

    def test_custom_alpha(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_heatmap_overlay(orig_image, mask_upscaled, alpha=0.7)
        assert overlay.shape == orig_image.shape

    def test_zero_mask_no_modification(self, orig_image):
        zero_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]), dtype=np.float32)
        overlay = generate_heatmap_overlay(orig_image, zero_mask, alpha=0.0)
        assert np.allclose(overlay, orig_image)


# ==============================================================
# GENERATE RED ALPHA OVERLAY
# ==============================================================


class TestGenerateRedAlphaOverlay:
    def test_output_shape(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_red_alpha_overlay(orig_image, mask_upscaled)
        assert overlay.shape == orig_image.shape

    def test_output_dtype(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_red_alpha_overlay(orig_image, mask_upscaled)
        assert overlay.dtype == np.uint8

    def test_output_values_in_range(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_red_alpha_overlay(orig_image, mask_upscaled)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_custom_color(self, orig_image, small_mask):
        mask_upscaled = upscale_mask(small_mask, orig_image.shape[1], orig_image.shape[0])
        overlay = generate_red_alpha_overlay(orig_image, mask_upscaled, color=(0, 255, 0))
        assert overlay.shape == orig_image.shape

    def test_zero_mask_no_modification(self, orig_image):
        zero_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]), dtype=np.float32)
        overlay = generate_red_alpha_overlay(orig_image, zero_mask, alpha=0.0)
        assert np.allclose(overlay, orig_image)


# ==============================================================
# ENCODE IMAGE BASE64
# ==============================================================


class TestEncodeImageBase64:
    def test_returns_string(self, orig_image):
        encoded = encode_image_base64(orig_image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_valid_base64(self, orig_image):
        encoded = encode_image_base64(orig_image)
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    def test_jpg_format(self, orig_image):
        encoded = encode_image_base64(orig_image, ext=".jpg", quality=90)
        assert isinstance(encoded, str)

    def test_png_format(self, orig_image):
        encoded = encode_image_base64(orig_image, ext=".png")
        assert isinstance(encoded, str)

    def test_different_quality(self, orig_image):
        low_q = encode_image_base64(orig_image, ext=".jpg", quality=10)
        high_q = encode_image_base64(orig_image, ext=".jpg", quality=95)
        assert isinstance(low_q, str)
        assert isinstance(high_q, str)

    def test_roundtrip_matches_shape(self, orig_image):
        encoded = encode_image_base64(orig_image, ext=".png")
        decoded_bytes = base64.b64decode(encoded)
        decoded_array = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded_array.shape == orig_image.shape


# ==============================================================
# POSTPROCESS (combined)
# ==============================================================


class TestPostprocess:
    def test_returns_dict(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=95.5,
            verdict="FORGED",
            forgery_type="Splicing",
        )
        assert isinstance(result, dict)

    def test_keys_present(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=95.5,
            verdict="FORGED",
            forgery_type="Splicing",
        )
        expected_keys = {"verdict", "confidence", "forgery_type", "forged_area_percentage", "heatmap_base64"}
        assert set(result.keys()) == expected_keys

    def test_verdict_passthrough(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=95.5,
            verdict="AUTHENTIC",
            forgery_type="Unknown",
        )
        assert result["verdict"] == "AUTHENTIC"
        assert result["confidence"] == 95.5
        assert result["forgery_type"] == "Unknown"

    def test_forged_area_percentage(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=80.0,
            verdict="FORGED",
        )
        assert isinstance(result["forged_area_percentage"], float)

    def test_heatmap_base64_is_valid(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=80.0,
            verdict="FORGED",
        )
        decoded = base64.b64decode(result["heatmap_base64"])
        assert len(decoded) > 0

    def test_custom_mask_threshold(self, orig_image, small_mask):
        result = postprocess(
            mask=small_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=80.0,
            verdict="FORGED",
            mask_threshold=0.9,
        )
        assert isinstance(result["forged_area_percentage"], float)

    def test_full_mask_gives_100_percent(self, orig_image, full_mask):
        result = postprocess(
            mask=full_mask,
            original_image=orig_image,
            original_size=(orig_image.shape[1], orig_image.shape[0]),
            confidence=99.0,
            verdict="FORGED",
        )
        assert result["forged_area_percentage"] == 100.0
