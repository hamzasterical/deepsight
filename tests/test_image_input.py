import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_input import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    ImageLoader,
    ImageTooSmallError,
    UnsupportedFormatError,
    detect_format_from_bytes,
    has_exif,
    is_lossless_format,
    is_lossy_format,
    is_supported_format,
    strip_exif,
)


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    return Image.new("RGB", (100, 80), color=(128, 64, 200))


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    return Image.new("L", (100, 80), color=128)


@pytest.fixture
def tmp_jpeg(sample_rgb_image, tmp_path: Path) -> Path:
    path = tmp_path / "test.jpg"
    sample_rgb_image.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def tmp_png(sample_rgb_image, tmp_path: Path) -> Path:
    path = tmp_path / "test.png"
    sample_rgb_image.save(path, format="PNG")
    return path


@pytest.fixture
def tmp_webp(sample_rgb_image, tmp_path: Path) -> Path:
    path = tmp_path / "test.webp"
    sample_rgb_image.save(path, format="WEBP")
    return path


@pytest.fixture
def loader() -> ImageLoader:
    return ImageLoader(min_file_size_kb=0, min_dimension=8)


# ==============================================================
# SUPPORTED EXTENSIONS / FORMAT DETECTION
# ==============================================================


class TestSupportedExtensions:
    def test_common_extensions_are_supported(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} should be supported"

    def test_heic_extensions_are_supported(self):
        assert ".heic" in SUPPORTED_EXTENSIONS
        assert ".heif" in SUPPORTED_EXTENSIONS


class TestIsSupportedFormat:
    def test_jpeg_supported(self, tmp_jpeg):
        assert is_supported_format(tmp_jpeg) is True

    def test_png_supported(self, tmp_png):
        assert is_supported_format(tmp_png) is True

    def test_unsupported_extension_returns_false(self, tmp_path):
        path = tmp_path / "test.bmp"
        path.write_bytes(b"fake")
        assert is_supported_format(path) is False

    def test_nonexistent_file_by_extension(self):
        assert is_supported_format("photo.tiff") is False


class TestDetectFormatFromBytes:
    def test_detect_jpeg(self):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        assert detect_format_from_bytes(buf.getvalue()) == "JPEG"

    def test_detect_png(self):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        assert detect_format_from_bytes(buf.getvalue()) == "PNG"

    def test_detect_webp(self):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="WEBP")
        assert detect_format_from_bytes(buf.getvalue()) == "WEBP"

    def test_unknown_format_returns_none(self):
        assert detect_format_from_bytes(b"\x00\x00\x00\x00 rubbish") is None


class TestIsLossyLossless:
    def test_jpeg_is_lossy(self, tmp_jpeg):
        assert is_lossy_format(tmp_jpeg) is True
        assert is_lossless_format(tmp_jpeg) is False

    def test_png_is_lossless(self, tmp_png):
        assert is_lossless_format(tmp_png) is True
        assert is_lossy_format(tmp_png) is False

    def test_webp_is_lossless_by_extension(self, tmp_webp):
        assert is_lossless_format(tmp_webp) is True


# ==============================================================
# IMAGE LOADER — BASIC LOADING
# ==============================================================


class TestImageLoader:
    def test_load_jpeg_returns_rgb_array(self, loader, tmp_jpeg):
        arr = loader.load(tmp_jpeg)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (80, 100, 3)
        assert arr.dtype == np.uint8

    def test_load_png_returns_rgb_array(self, loader, tmp_png):
        arr = loader.load(tmp_png)
        assert arr.shape == (80, 100, 3)

    def test_load_webp_returns_rgb_array(self, loader, tmp_webp):
        arr = loader.load(tmp_webp)
        assert arr.shape == (80, 100, 3)

    def test_grayscale_is_converted_to_rgb(self, loader, tmp_path, sample_grayscale_image):
        path = tmp_path / "gray.png"
        sample_grayscale_image.save(path)
        arr = loader.load(path)
        assert arr.shape == (80, 100, 3)

    def test_load_nonexistent_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load(Path("does_not_exist.jpg"))

    def test_load_unsupported_format_raises(self, loader, tmp_path):
        path = tmp_path / "test.bmp"
        Image.new("RGB", (10, 10)).save(path, format="BMP")
        with pytest.raises(UnsupportedFormatError):
            loader.load(path)

    def test_load_empty_file_raises(self, loader, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        with pytest.raises(ImageLoadError):
            loader.load(path)

    def test_load_too_small_file_raises(self, tmp_path):
        loader = ImageLoader(min_file_size_kb=9999)
        path = tmp_path / "small.jpg"
        Image.new("RGB", (10, 10)).save(path, format="JPEG")
        with pytest.raises(ImageTooSmallError):
            loader.load(path)

    def test_load_too_small_dimensions_raises(self, tmp_path):
        loader = ImageLoader(min_file_size_kb=0, min_dimension=500)
        path = tmp_path / "tiny.jpg"
        Image.new("RGB", (10, 10)).save(path, format="JPEG")
        with pytest.raises(ImageTooSmallError):
            loader.load(path)


# ==============================================================
# LOAD FROM BYTES
# ==============================================================


class TestLoadFromBytes:
    def test_load_jpeg_bytes(self, loader, tmp_jpeg):
        data = tmp_jpeg.read_bytes()
        arr = loader.load_from_bytes(data, source_name=tmp_jpeg.name)
        assert arr.shape == (80, 100, 3)

    def test_load_png_bytes(self, loader, tmp_png):
        data = tmp_png.read_bytes()
        arr = loader.load_from_bytes(data)
        assert arr.shape == (80, 100, 3)

    def test_load_from_bytes_too_small_raises(self, loader):
        with pytest.raises(ImageTooSmallError):
            loader.load_from_bytes(b"\xff\xd8\xff\xe0")

    def test_corrupt_bytes_raises(self, loader):
        with pytest.raises(ImageLoadError):
            loader.load_from_bytes(b"\x00" * 1024, source_name="corrupt")


# ==============================================================
# METADATA
# ==============================================================


class TestMetadata:
    def test_get_metadata_basic(self, loader, tmp_jpeg):
        meta = loader.get_metadata(tmp_jpeg)
        assert meta["file_name"] == "test.jpg"
        assert meta["width"] == 100
        assert meta["height"] == 80
        assert meta["format"] == "jpg"

    def test_has_exif_false_on_simple_image(self, tmp_png):
        assert has_exif(tmp_png) is False

    def test_strip_exif_returns_jpeg_bytes(self, tmp_jpeg):
        buf = strip_exif(tmp_jpeg)
        assert isinstance(buf, bytes)
        assert len(buf) > 0
        assert buf[:2] == b"\xff\xd8"


class TestSummary:
    def test_summary_includes_key_info(self, loader, tmp_jpeg):
        s = loader.summary(tmp_jpeg)
        assert "test.jpg" in s
        assert "JPG" in s or "JPEG" in s
        assert "100x80" in s


# ==============================================================
# EDGE CASES
# ==============================================================


class TestEdgeCases:
    def test_loader_defaults_do_not_crash(self):
        loader = ImageLoader()
        assert loader.min_file_size_kb == 12.0
        assert loader.min_dimension == 32
        assert loader.convert_to_rgb is True
        assert loader.apply_orientation is True

    def test_disable_orientation_and_rgb(self, tmp_path):
        loader = ImageLoader(apply_orientation=False, convert_to_rgb=False)
        path = tmp_path / "gray.png"
        Image.new("L", (50, 50)).save(path)
        arr = loader.load(path)
        assert arr.ndim == 2

    def test_supported_extensions_set_contains_all(self):
        expected = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
        assert SUPPORTED_EXTENSIONS == expected
