import base64
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def upscale_mask(
    mask: np.ndarray,
    orig_width: int,
    orig_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.shape != (orig_height, orig_width):
        mask = cv2.resize(mask, (orig_width, orig_height), interpolation=interpolation)

    return mask


def compute_forged_area_percentage(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0

    forged_pixels = float(np.sum(mask > threshold))
    return round(forged_pixels / total_pixels * 100.0, 2)


def generate_heatmap_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    mask_uint8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_uint8, colormap)
    overlay = cv2.addWeighted(original, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def generate_red_alpha_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    overlay = original.copy().astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended = overlay * (1.0 - mask_3ch * alpha) + np.array(color, dtype=np.float32) * mask_3ch * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def encode_image_base64(
    image: np.ndarray,
    ext: str = ".jpg",
    quality: int = 95,
) -> str:
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise RuntimeError(f"Failed to encode image as {ext}")

    return base64.b64encode(buffer).decode("utf-8")


def postprocess(
    mask: np.ndarray,
    original_image: np.ndarray,
    original_size: Tuple[int, int],
    confidence: float,
    verdict: str,
    forgery_type: str = "Unknown",
    mask_threshold: float = 0.5,
    heatmap_alpha: float = 0.4,
) -> dict:
    orig_w, orig_h = original_size

    upscaled_mask = upscale_mask(mask, orig_w, orig_h)
    forged_pct = compute_forged_area_percentage(upscaled_mask, mask_threshold)
    heatmap = generate_heatmap_overlay(original_image, upscaled_mask, heatmap_alpha)
    heatmap_b64 = encode_image_base64(heatmap)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "forgery_type": forgery_type,
        "forged_area_percentage": forged_pct,
        "heatmap_base64": heatmap_b64,
    }
