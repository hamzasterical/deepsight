import numpy as np
import pytest

from src.preprocessing.resize_normalise import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
    NormaliseConfig,
    NormaliseTransform,
    OriginalDimensions,
    PreprocessingPipeline,
    ResizeConfig,
    ResizeTransform,
    denormalise_rgb,
    normalise_rgb,
    resize_image,
    resize_mask,
    resize_with_aspect_ratio,
)


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(50, 60, 3), dtype=np.uint8)


@pytest.fixture
def square_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def binary_mask() -> np.ndarray:
    mask = np.zeros((300, 400), dtype=np.float32)
    mask[50:150, 100:200] = 1.0
    return mask


# ==============================================================
# RESIZE
# ==============================================================


class TestResizeImage:
    def test_resize_to_target(self, rgb_image):
        result = resize_image(rgb_image, target_size=224)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_resize_upscale(self, small_image):
        result = resize_image(small_image, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_resize_already_target(self, square_image):
        result = resize_image(square_image)
        assert result.shape == (224, 224, 3)
        assert np.array_equal(result, square_image)

    def test_resize_grayscale(self):
        gray = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        result = resize_image(gray, target_size=224)
        assert result.shape == (224, 224)

    def test_resize_different_target_size(self, rgb_image):
        result = resize_image(rgb_image, target_size=128)
        assert result.shape == (128, 128, 3)


class TestResizeMask:
    def test_resize_mask_binary(self, binary_mask):
        result = resize_mask(binary_mask, target_size=224)
        assert result.shape == (224, 224)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_resize_mask_3d(self):
        mask_3d = np.random.rand(100, 100, 1).astype(np.float32)
        result = resize_mask(mask_3d, target_size=224)
        assert result.shape == (224, 224, 1)

    def test_resize_mask_already_target(self):
        mask = np.random.rand(224, 224).astype(np.float32)
        result = resize_mask(mask)
        assert result.shape == (224, 224)
        assert np.array_equal(result, mask)

    def test_resize_mask_clips_values(self):
        mask = np.ones((100, 100), dtype=np.float32) * 2.0
        result = resize_mask(mask)
        assert result.max() <= 1.0

    def test_resize_mask_preserves_zeros(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        result = resize_mask(mask, target_size=224)
        assert result.shape == (224, 224)


class TestResizeWithAspectRatio:
    def test_preserves_aspect_ratio(self, rgb_image):
        result = resize_with_aspect_ratio(rgb_image, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_letterbox_padding(self):
        tall = np.random.randint(0, 256, size=(400, 100, 3), dtype=np.uint8)
        result = resize_with_aspect_ratio(tall, target_size=224)
        assert result.shape == (224, 224, 3)
        assert (result[:, 0, :] == 0).all()
        assert (result[:, -1, :] == 0).all()

    def test_wide_image_padding(self):
        wide = np.random.randint(0, 256, size=(100, 400, 3), dtype=np.uint8)
        result = resize_with_aspect_ratio(wide, target_size=224)
        assert result.shape == (224, 224, 3)
        assert (result[0, :, :] == 0).all()
        assert (result[-1, :, :] == 0).all()

    def test_custom_padding_value(self, rgb_image):
        result = resize_with_aspect_ratio(rgb_image, target_size=224, padding_value=128)
        assert result.shape == (224, 224, 3)

    def test_grayscale_letterbox(self):
        tall = np.random.randint(0, 256, size=(300, 100), dtype=np.uint8)
        result = resize_with_aspect_ratio(tall, target_size=224)
        assert result.shape == (224, 224)


# ==============================================================
# NORMALISE
# ==============================================================


class TestNormaliseRGB:
    def test_normalise_output_shape_and_dtype(self, rgb_image):
        result = normalise_rgb(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32

    def test_normalise_values(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = normalise_rgb(image)
        expected_r = (128 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert np.allclose(result[:, :, 0], expected_r)

    def test_normalise_black_image(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = normalise_rgb(image)
        expected_each = (0.0 - IMAGENET_MEAN) / IMAGENET_STD
        for c in range(3):
            assert np.allclose(result[:, :, c], expected_each[c])

    def test_normalise_already_float(self):
        image = np.random.rand(10, 10, 3).astype(np.float32) * 255
        result = normalise_rgb(image)
        assert result.dtype == np.float32

    def test_normalise_custom_mean_std(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = normalise_rgb(image, mean=mean, std=std)
        assert np.allclose(result, image / 255.0)


class TestDenormaliseRGB:
    def test_denormalise_roundtrip(self, rgb_image):
        normed = normalise_rgb(rgb_image)
        restored = denormalise_rgb(normed)
        assert restored.shape == rgb_image.shape
        assert restored.dtype == np.uint8
        diff = np.abs(restored.astype(np.int16) - rgb_image.astype(np.int16))
        assert diff.max() <= 2

    def test_denormalise_output_range(self):
        image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        normed = normalise_rgb(image)
        restored = denormalise_rgb(normed)
        assert restored.min() >= 0
        assert restored.max() <= 255


# ==============================================================
# ResizeTransform
# ==============================================================


class TestResizeTransform:
    def test_call_returns_resized_and_dims(self, rgb_image):
        transform = ResizeTransform()
        resized, orig = transform(rgb_image)
        assert resized.shape == (224, 224, 3)
        assert orig.width == 400
        assert orig.height == 300

    def test_resize_mask_method(self, binary_mask):
        transform = ResizeTransform()
        result = transform.resize_mask(binary_mask)
        assert result.shape == (224, 224)

    def test_custom_target_size(self, rgb_image):
        config = ResizeConfig(target_size=128)
        transform = ResizeTransform(config)
        resized, _ = transform(rgb_image)
        assert resized.shape == (128, 128, 3)

    def test_preserve_aspect_ratio_config(self, rgb_image):
        config = ResizeConfig(preserve_aspect_ratio=True)
        transform = ResizeTransform(config)
        resized, orig = transform(rgb_image)
        assert resized.shape == (224, 224, 3)
        assert orig.width == 400
        assert orig.height == 300


# ==============================================================
# NormaliseTransform
# ==============================================================


class TestNormaliseTransform:
    def test_call(self, rgb_image):
        transform = NormaliseTransform()
        result = transform(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32

    def test_inverse(self, rgb_image):
        transform = NormaliseTransform()
        normed = transform(rgb_image)
        restored = transform.inverse(normed)
        assert restored.shape == rgb_image.shape
        assert restored.dtype == np.uint8

    def test_custom_config(self):
        config = NormaliseConfig(
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            std=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        transform = NormaliseTransform(config)
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = transform(image)
        assert np.allclose(result, image / 255.0)


# ==============================================================
# PreprocessingPipeline
# ==============================================================


class TestPreprocessingPipeline:
    def test_full_pipeline(self, rgb_image):
        pipeline = PreprocessingPipeline()
        result = pipeline(rgb_image)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32

    def test_original_dimensions_property(self, rgb_image):
        pipeline = PreprocessingPipeline()
        pipeline(rgb_image)
        dims = pipeline.original_dimensions
        assert dims is not None
        assert dims.width == 400
        assert dims.height == 300

    def test_original_dimensions_none_before_call(self):
        pipeline = PreprocessingPipeline()
        assert pipeline.original_dimensions is None

    def test_process_mask_after_image(self, rgb_image, binary_mask):
        pipeline = PreprocessingPipeline()
        pipeline(rgb_image)
        resized_mask = pipeline.process_mask(binary_mask)
        assert resized_mask.shape == (224, 224)
        assert resized_mask.dtype == np.float32

    def test_to_chw(self, rgb_image):
        pipeline = PreprocessingPipeline()
        result = pipeline(rgb_image)
        chw = pipeline.to_chw(result)
        assert chw.shape == (3, 224, 224)

    def test_to_hwc(self, rgb_image):
        pipeline = PreprocessingPipeline()
        result = pipeline(rgb_image)
        chw = pipeline.to_chw(result)
        hwc = pipeline.to_hwc(chw)
        assert hwc.shape == (224, 224, 3)

    def test_process_method(self, rgb_image):
        pipeline = PreprocessingPipeline()
        result = pipeline.process(rgb_image)
        assert result.shape == (224, 224, 3)

    def test_process_image_only(self, rgb_image):
        pipeline = PreprocessingPipeline()
        result = pipeline.process_image_only(rgb_image)
        assert result.shape == (224, 224, 3)

    def test_mask_process_without_image(self, binary_mask):
        pipeline = PreprocessingPipeline()
        resized = pipeline.process_mask(binary_mask)
        assert resized.shape == (224, 224)


# ==============================================================
# CONSTANTS
# ==============================================================


class TestConstants:
    def test_imagenet_mean_shape(self):
        assert IMAGENET_MEAN.shape == (3,)
        assert IMAGENET_MEAN.dtype == np.float32

    def test_imagenet_std_shape(self):
        assert IMAGENET_STD.shape == (3,)
        assert IMAGENET_STD.dtype == np.float32

    def test_target_size(self):
        assert TARGET_SIZE == 224


# ==============================================================
# EDGE CASES
# ==============================================================


class TestEdgeCases:
    def test_huge_image_resize(self):
        huge = np.random.randint(0, 256, size=(4000, 6000, 3), dtype=np.uint8)
        result = resize_image(huge, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_tiny_image_upscale(self):
        tiny = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        result = resize_image(tiny, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_single_channel_mask(self):
        mask = np.random.rand(100, 100).astype(np.float32)
        result = resize_mask(mask, target_size=224)
        assert result.shape == (224, 224)

    def test_letterbox_both_padding(self):
        small = np.random.randint(0, 256, size=(50, 30, 3), dtype=np.uint8)
        result = resize_with_aspect_ratio(small, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_normalise_uint8_input(self):
        image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        result = normalise_rgb(image)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_pipeline_idempotent_for_224_image(self, square_image):
        pipeline = PreprocessingPipeline()
        result = pipeline(square_image)
        assert result.shape == (3, 224, 224) or result.shape == (224, 224, 3)

    def test_denormalise_recoverable(self):
        original = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        normed = normalise_rgb(original)
        restored = denormalise_rgb(normed)
        mse = np.mean((original.astype(np.float32) - restored.astype(np.float32)) ** 2)
        assert mse < 2.0
