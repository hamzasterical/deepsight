from typing import Optional, Tuple

import cv2
import numpy as np


def create_heatmap_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    mask_uint8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_uint8, colormap)

    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


def create_red_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    overlay = image.astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    color_arr = np.array(color, dtype=np.float32)
    blended = overlay * (1.0 - mask_3ch * alpha) + color_arr * mask_3ch * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_comparison_grid(
    original: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    title: Optional[str] = None,
) -> np.ndarray:
    mask_display = np.clip(mask * 255, 0, 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

    top = np.hstack([original, mask_bgr])
    bottom = np.hstack([overlay, np.zeros_like(original)])

    if mask.shape[:2] != original.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (original.shape[1], original.shape[0]))

    grid = np.vstack([top, bottom])
    return grid
