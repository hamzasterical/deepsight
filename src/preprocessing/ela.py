import io
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ELA_QUALITY: int = 75
DEFAULT_ELA_AMPLIFY: float = 20.0


def _is_png_data(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def _is_jpeg_data(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8"


def compute_ela(
    image: np.ndarray,
    quality: int = DEFAULT_ELA_QUALITY,
    amplify: float = DEFAULT_ELA_AMPLIFY,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    recompressed = Image.open(buf)
    recompressed_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(image.astype(np.float32) - recompressed_arr)
    ela = diff * amplify
    ela = np.clip(ela, 0, 255).astype(np.uint8)

    return ela


def compute_ela_from_pil(
    pil_image: Image.Image,
    quality: int = DEFAULT_ELA_QUALITY,
    amplify: float = DEFAULT_ELA_AMPLIFY,
) -> np.ndarray:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    arr = np.array(pil_image, dtype=np.uint8)
    return compute_ela(arr, quality=quality, amplify=amplify)


def compute_ela_from_bytes(
    data: bytes,
    quality: int = DEFAULT_ELA_QUALITY,
    amplify: float = DEFAULT_ELA_AMPLIFY,
) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(data))
        return compute_ela_from_pil(pil_img, quality=quality, amplify=amplify)
    except Exception as e:
        raise ValueError(f"Failed to decode image bytes for ELA: {e}") from e


def compute_ela_from_file(
    file_path: Union[str, Path],
    quality: int = DEFAULT_ELA_QUALITY,
    amplify: float = DEFAULT_ELA_AMPLIFY,
) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    pil_img = Image.open(path)
    return compute_ela_from_pil(pil_img, quality=quality, amplify=amplify)


def ela_to_3channel(ela_map: np.ndarray) -> np.ndarray:
    if ela_map.ndim == 2:
        return np.stack([ela_map] * 3, axis=-1)
    if ela_map.ndim == 3 and ela_map.shape[2] == 1:
        return np.repeat(ela_map, 3, axis=-1)
    if ela_map.ndim == 3 and ela_map.shape[2] == 3:
        return ela_map
    raise ValueError(f"Cannot convert ELA map of shape {ela_map.shape} to 3-channel")


def estimate_ela_usefulness(image: np.ndarray) -> dict:
    info = {
        "is_useful": True,
        "reason": None,
    }

    if image.dtype == np.uint8:
        pixel_range = image.max() - image.min()
        if pixel_range < 10:
            info["is_useful"] = False
            info["reason"] = "near-constant image"
            return info

    temp_path = None
    try:
        pil_img = Image.fromarray(image)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
            pil_img.save(f, format="JPEG", quality=DEFAULT_ELA_QUALITY)

        re_img = Image.open(temp_path)
        re_arr = np.array(re_img, dtype=np.float32)
        diff = np.abs(image.astype(np.float32) - re_arr)

        mean_diff = diff.mean()
        info["mean_ela_diff"] = float(mean_diff)

        if mean_diff < 0.5:
            info["is_useful"] = False
            info["reason"] = (
                f"very low ELA response ({mean_diff:.2f}) — "
                "image may be lossless (PNG) or already low-quality JPEG"
            )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    return info


class ELATransform:
    def __init__(
        self,
        quality: int = DEFAULT_ELA_QUALITY,
        amplify: float = DEFAULT_ELA_AMPLIFY,
        to_3channel: bool = True,
    ):
        self.quality = quality
        self.amplify = amplify
        self.to_3channel = to_3channel

    def __call__(self, image: np.ndarray) -> np.ndarray:
        ela = compute_ela(image, quality=self.quality, amplify=self.amplify)

        if self.to_3channel:
            return ela_to_3channel(ela)

        return ela

    def from_pil(self, pil_image: Image.Image) -> np.ndarray:
        ela = compute_ela_from_pil(
            pil_image, quality=self.quality, amplify=self.amplify
        )
        if self.to_3channel:
            return ela_to_3channel(ela)
        return ela

    def from_bytes(self, data: bytes) -> np.ndarray:
        ela = compute_ela_from_bytes(
            data, quality=self.quality, amplify=self.amplify
        )
        if self.to_3channel:
            return ela_to_3channel(ela)
        return ela

    def from_file(self, file_path: Union[str, Path]) -> np.ndarray:
        ela = compute_ela_from_file(
            file_path, quality=self.quality, amplify=self.amplify
        )
        if self.to_3channel:
            return ela_to_3channel(ela)
        return ela
