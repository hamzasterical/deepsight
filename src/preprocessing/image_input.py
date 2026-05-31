import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
PIL_SAFE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOSSY_FORMATS = {".jpg", ".jpeg"}
LOSSLESS_FORMATS = {".png", ".webp"}


class ImageLoadError(Exception):
    pass


class UnsupportedFormatError(ImageLoadError):
    pass


class ImageTooSmallError(ImageLoadError):
    pass


def _has_pillow_heif() -> bool:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        return False


_HAS_HEIF_SUPPORT = _has_pillow_heif()


def _get_format_hint(file_path: Path) -> Optional[str]:
    ext = file_path.suffix.lower()
    fmt_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".heic": "HEIC",
        ".heif": "HEIF",
    }
    return fmt_map.get(ext)


def is_supported_format(file_path: Union[str, Path]) -> bool:
    ext = Path(file_path).suffix.lower()
    if ext in {".heic", ".heif"} and not _HAS_HEIF_SUPPORT:
        return False
    return ext in SUPPORTED_EXTENSIONS


def is_lossy_format(file_path: Union[str, Path]) -> bool:
    return Path(file_path).suffix.lower() in LOSSY_FORMATS


def is_lossless_format(file_path: Union[str, Path]) -> bool:
    return Path(file_path).suffix.lower() in LOSSLESS_FORMATS


def detect_format_from_bytes(data: bytes) -> Optional[str]:
    if data[:2] == b"\xff\xd8":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[:4] == b"ftyp":
        ftyp = data[4:12]
        if ftyp[:4] in (b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1"):
            return "HEIC"
    return None


def load_image_bytes(data: bytes, source_name: str = "bytes") -> np.ndarray:
    if len(data) < 256:
        raise ImageTooSmallError(
            f"Image data too small ({len(data)} bytes) from {source_name}"
        )
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        raise ImageLoadError(
            f"Failed to decode image from {source_name}: {e}"
        ) from e


def has_exif(file_path: Union[str, Path]) -> bool:
    try:
        img = Image.open(file_path)
        exif = img.getexif()
        return len(exif) > 0
    except Exception:
        return False


def strip_exif(file_path: Union[str, Path]) -> bytes:
    img = Image.open(file_path)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class ImageLoader:
    def __init__(
        self,
        min_file_size_kb: float = 12.0,
        min_dimension: int = 32,
        convert_to_rgb: bool = True,
        apply_orientation: bool = True,
    ):
        self.min_file_size_kb = min_file_size_kb
        self.min_dimension = min_dimension
        self.convert_to_rgb = convert_to_rgb
        self.apply_orientation = apply_orientation

    def load(self, file_path: Union[str, Path]) -> np.ndarray:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if not is_supported_format(path):
            raise UnsupportedFormatError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        file_size_kb = path.stat().st_size / 1024
        if file_size_kb < self.min_file_size_kb:
            raise ImageTooSmallError(
                f"File too small ({file_size_kb:.1f} KB < {self.min_file_size_kb} KB): {path}"
            )

        ext = path.suffix.lower()
        if ext in {".heic", ".heif"} and not _HAS_HEIF_SUPPORT:
            raise UnsupportedFormatError(
                "HEIC/HEIF support requires 'pillow-heif'. "
                "Install with: pip install pillow-heif"
            )

        try:
            img = Image.open(path)

            if self.apply_orientation:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)

            if self.convert_to_rgb:
                img = img.convert("RGB")

            arr = np.array(img)

            h, w = arr.shape[:2]
            if h < self.min_dimension or w < self.min_dimension:
                raise ImageTooSmallError(
                    f"Image dimensions ({w}x{h}) below minimum ({self.min_dimension}x{self.min_dimension}): {path}"
                )

            logger.debug(
                "Loaded %s — %s, size=%dx%d, %.1f KB",
                path.name, ext.upper(), w, h, file_size_kb
            )
            return arr

        except (ImageLoadError, UnsupportedFormatError, ImageTooSmallError):
            raise
        except Exception as e:
            raise ImageLoadError(f"Failed to load {path}: {e}") from e

    def load_from_bytes(
        self, data: bytes, source_name: str = "bytes"
    ) -> np.ndarray:
        if len(data) < 256:
            raise ImageTooSmallError(
                f"Image data too small ({len(data)} bytes) from {source_name}"
            )

        detected = detect_format_from_bytes(data)
        if detected == "HEIC" and not _HAS_HEIF_SUPPORT:
            raise UnsupportedFormatError(
                "HEIC/HEIF support requires 'pillow-heif'."
            )

        try:
            img = Image.open(io.BytesIO(data))

            if self.apply_orientation:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)

            if self.convert_to_rgb:
                img = img.convert("RGB")

            arr = np.array(img)

            h, w = arr.shape[:2]
            if h < self.min_dimension or w < self.min_dimension:
                raise ImageTooSmallError(
                    f"Image dimensions ({w}x{h}) below minimum "
                    f"({self.min_dimension}x{self.min_dimension})"
                )

            fmt_name = detected or "unknown"
            logger.debug("Loaded %s from bytes — %s, size=%dx%d", source_name, fmt_name, w, h)
            return arr

        except (ImageLoadError, UnsupportedFormatError, ImageTooSmallError):
            raise
        except Exception as e:
            raise ImageLoadError(
                f"Failed to decode image from {source_name}: {e}"
            ) from e

    def get_metadata(self, file_path: Union[str, Path]) -> dict:
        path = Path(file_path)
        info = {
            "file_name": path.name,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
            "format": path.suffix.lower().lstrip("."),
            "has_exif": False,
            "exif_fields": [],
        }

        try:
            img = Image.open(path)
            exif = img.getexif()
            if exif:
                info["has_exif"] = True
                info["exif_fields"] = [
                    _EXIF_TAGS.get(k, str(k)) for k, v in exif.items()
                ]
            info["width"], info["height"] = img.size
            info["mode"] = img.mode
        except Exception as e:
            logger.warning("Could not read metadata from %s: %s", path, e)

        return info

    def to_pil(self, file_path: Union[str, Path]) -> Image.Image:
        arr = self.load(file_path)
        return Image.fromarray(arr)

    def summary(self, file_path: Union[str, Path]) -> str:
        meta = self.get_metadata(file_path)
        parts = [
            f"File: {meta['file_name']}",
            f"Format: {meta['format'].upper()}",
            f"Size: {meta['width']}x{meta['height']}",
            f"File Size: {meta['file_size_kb']} KB",
        ]
        if meta["has_exif"]:
            parts.append(f"EXIF: {len(meta['exif_fields'])} fields")
        else:
            parts.append("EXIF: none (likely stripped)")
        return " | ".join(parts)


_EXIF_TAGS = {
    271: "Make",
    272: "Model",
    274: "Orientation",
    296: "ResolutionUnit",
    282: "XResolution",
    283: "YResolution",
    271: "Make",
    272: "Model",
    305: "Software",
    306: "DateTime",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    42034: "CameraOwnerName",
    42035: "BodySerialNumber",
    42036: "LensSpecification",
    42037: "LensMake",
    42038: "LensModel",
}
