import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.ela import (
    DEFAULT_ELA_AMPLIFY,
    DEFAULT_ELA_QUALITY,
    ELATransform,
    compute_ela,
    compute_ela_from_bytes,
    compute_ela_from_file,
    compute_ela_from_pil,
    ela_to_3channel,
    estimate_ela_usefulness,
)


@pytest.fixture
def synthetic_image() -> np.ndarray:
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[30:70, 30:70, :] = 200
    arr[40:60, 40:60, 0] = 255
    return arr


@pytest.fixture
def jpeg_bytes(synthetic_image) -> bytes:
    img = Image.fromarray(synthetic_image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def png_bytes(synthetic_image) -> bytes:
    img = Image.fromarray(synthetic_image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tmp_jpeg(synthetic_image, tmp_path) -> str:
    path = tmp_path / "test.jpg"
    Image.fromarray(synthetic_image).save(path, format="JPEG", quality=95)
    return str(path)


# ==============================================================
# COMPUTE ELA
# ==============================================================


class TestComputeELA:
    def test_output_shape_and_dtype(self, synthetic_image):
        ela = compute_ela(synthetic_image)
        assert ela.shape == synthetic_image.shape
        assert ela.dtype == np.uint8

    def test_ela_values_non_negative(self, synthetic_image):
        ela = compute_ela(synthetic_image)
        assert ela.min() >= 0

    def test_ela_values_bounded(self, synthetic_image):
        ela = compute_ela(synthetic_image)
        assert ela.max() <= 255

    def test_default_quality(self, synthetic_image):
        ela = compute_ela(synthetic_image, quality=75)
        assert ela.shape == synthetic_image.shape

    def test_custom_quality(self, synthetic_image):
        ela_low = compute_ela(synthetic_image, quality=50)
        ela_high = compute_ela(synthetic_image, quality=95)
        assert ela_low.mean() >= ela_high.mean()

    def test_custom_amplify(self, synthetic_image):
        ela_low = compute_ela(synthetic_image, amplify=5)
        ela_high = compute_ela(synthetic_image, amplify=30)
        assert ela_low.mean() <= ela_high.mean()

    def test_amplify_zero(self, synthetic_image):
        ela = compute_ela(synthetic_image, amplify=0)
        assert ela.max() == 0

    def test_raises_on_grayscale(self):
        gray = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_ela(gray)

    def test_raises_on_wrong_dims(self):
        img = np.random.randint(0, 256, size=(100, 100, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_ela(img)

    def test_float_input_converted(self, synthetic_image):
        float_img = synthetic_image.astype(np.float32)
        ela = compute_ela(float_img)
        assert ela.dtype == np.uint8

    def test_uniform_image_low_ela(self):
        uniform = np.full((50, 50, 3), 128, dtype=np.uint8)
        ela = compute_ela(uniform)
        assert ela.mean() < 10


# ==============================================================
# COMPUTE ELA FROM PIL
# ==============================================================


class TestComputeELAFromPIL:
    def test_from_pil(self, synthetic_image):
        pil_img = Image.fromarray(synthetic_image)
        ela = compute_ela_from_pil(pil_img)
        assert ela.shape == synthetic_image.shape
        assert ela.dtype == np.uint8

    def test_from_grayscale_pil(self):
        pil_img = Image.new("L", (50, 50), color=128)
        ela = compute_ela_from_pil(pil_img)
        assert ela.shape == (50, 50, 3)

    def test_from_rgba_pil(self):
        pil_img = Image.new("RGBA", (50, 50), color=(128, 64, 200, 255))
        ela = compute_ela_from_pil(pil_img)
        assert ela.shape == (50, 50, 3)


# ==============================================================
# COMPUTE ELA FROM BYTES
# ==============================================================


class TestComputeELAFromBytes:
    def test_from_jpeg_bytes(self, jpeg_bytes):
        ela = compute_ela_from_bytes(jpeg_bytes)
        assert ela.shape == (100, 100, 3)
        assert ela.dtype == np.uint8

    def test_from_png_bytes(self, png_bytes):
        ela = compute_ela_from_bytes(png_bytes)
        assert ela.shape == (100, 100, 3)

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError):
            compute_ela_from_bytes(b"\x00\x01\x02" * 100)


# ==============================================================
# COMPUTE ELA FROM FILE
# ==============================================================


class TestComputeELAFromFile:
    def test_from_file(self, tmp_jpeg):
        ela = compute_ela_from_file(tmp_jpeg)
        assert ela.shape == (100, 100, 3)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            compute_ela_from_file("/nonexistent/image.jpg")


# ==============================================================
# ELA TO 3 CHANNEL
# ==============================================================


class TestELATo3Channel:
    def test_grayscale_to_3channel(self):
        ela = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        result = ela_to_3channel(ela)
        assert result.shape == (50, 50, 3)

    def test_3channel_passthrough(self):
        ela = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        result = ela_to_3channel(ela)
        assert result.shape == (50, 50, 3)
        assert np.array_equal(result, ela)

    def test_1channel_to_3channel(self):
        ela = np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8)
        result = ela_to_3channel(ela)
        assert result.shape == (50, 50, 3)

    def test_channels_identical(self):
        ela = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        result = ela_to_3channel(ela)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])

    def test_raises_on_4d(self):
        ela = np.random.randint(0, 256, size=(10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            ela_to_3channel(ela)


# ==============================================================
# ESTIMATE ELA USEFULNESS
# ==============================================================


class TestEstimateELAUsefulness:
    def test_useful_for_random_image(self, synthetic_image):
        info = estimate_ela_usefulness(synthetic_image)
        assert "is_useful" in info
        assert "mean_ela_diff" in info

    def test_constant_image_not_useful(self):
        constant = np.full((50, 50, 3), 128, dtype=np.uint8)
        info = estimate_ela_usefulness(constant)
        assert info["is_useful"] is False


# ==============================================================
# ELA TRANSFORM
# ==============================================================


class TestELATransform:
    def test_call_returns_3channel(self, synthetic_image):
        transform = ELATransform()
        result = transform(synthetic_image)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_call_returns_grayscale(self, synthetic_image):
        transform = ELATransform(to_3channel=False)
        result = transform(synthetic_image)
        assert result.shape == (100, 100, 3)

    def test_default_params(self):
        transform = ELATransform()
        assert transform.quality == DEFAULT_ELA_QUALITY
        assert transform.amplify == DEFAULT_ELA_AMPLIFY
        assert transform.to_3channel is True

    def test_custom_params(self):
        transform = ELATransform(quality=50, amplify=10, to_3channel=False)
        assert transform.quality == 50
        assert transform.amplify == 10
        assert transform.to_3channel is False

    def test_from_pil(self, synthetic_image):
        transform = ELATransform()
        pil_img = Image.fromarray(synthetic_image)
        result = transform.from_pil(pil_img)
        assert result.shape == (100, 100, 3)

    def test_from_bytes(self, jpeg_bytes):
        transform = ELATransform()
        result = transform.from_bytes(jpeg_bytes)
        assert result.shape == (100, 100, 3)

    def test_from_file(self, tmp_jpeg):
        transform = ELATransform()
        result = transform.from_file(tmp_jpeg)
        assert result.shape == (100, 100, 3)


# ==============================================================
# EDGE CASES
# ==============================================================


class TestEdgeCases:
    def test_small_image(self):
        small = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        ela = compute_ela(small)
        assert ela.shape == (10, 10, 3)

    def test_large_image(self):
        large = np.random.randint(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
        ela = compute_ela(large)
        assert ela.shape == (1000, 1000, 3)

    def test_jpeg_artifacts_visible(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255
        ela = compute_ela(img, quality=75, amplify=15)
        assert ela.mean() > 0

    def test_ela_on_png_has_lower_response(self):
        png_img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        pil_img = Image.fromarray(png_img)
        buf_png = io.BytesIO()
        pil_img.save(buf_png, format="PNG")
        buf_png.seek(0)

        buf_jpg = io.BytesIO()
        pil_img.save(buf_jpg, format="JPEG", quality=95)
        buf_jpg.seek(0)

        from PIL import Image as PILImage
        png_reloaded = np.array(PILImage.open(buf_png))
        jpg_reloaded = np.array(PILImage.open(buf_jpg))

        ela_png = compute_ela(png_reloaded)
        ela_jpg = compute_ela(jpg_reloaded)
        assert ela_jpg.mean() > 0

    def test_ela_deterministic(self, synthetic_image):
        ela1 = compute_ela(synthetic_image)
        ela2 = compute_ela(synthetic_image)
        assert np.array_equal(ela1, ela2)

    def test_transform_from_bytes_raises_on_corrupt(self):
        transform = ELATransform()
        with pytest.raises(Exception):
            transform.from_bytes(b"\x00" * 100)
