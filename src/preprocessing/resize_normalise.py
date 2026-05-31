from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE: int = 224


@dataclass
class OriginalDimensions:
    width: int
    height: int


@dataclass
class ResizeConfig:
    target_size: int = TARGET_SIZE
    image_interpolation: int = cv2.INTER_LINEAR
    image_down_interpolation: int = cv2.INTER_LANCZOS4
    mask_interpolation: int = cv2.INTER_LINEAR
    preserve_aspect_ratio: bool = False
    padding_value: int = 0


@dataclass
class NormaliseConfig:
    mean: np.ndarray = field(default_factory=lambda: IMAGENET_MEAN.copy())
    std: np.ndarray = field(default_factory=lambda: IMAGENET_STD.copy())
    scale: float = 255.0


DEFAULT_RESIZE_CONFIG = ResizeConfig()
DEFAULT_NORMALISE_CONFIG = NormaliseConfig()


def resize_image(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
    down_interpolation: Optional[int] = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        return image.copy()

    if h > target_size or w > target_size:
        interp = down_interpolation if down_interpolation is not None else interpolation
    else:
        interp = interpolation

    return cv2.resize(image, (target_size, target_size), interpolation=interp)


def resize_mask(
    mask: np.ndarray,
    target_size: int = TARGET_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    h, w = mask.shape[:2]
    if h == target_size and w == target_size:
        return mask.copy()

    if mask.ndim == 2:
        resized = cv2.resize(mask, (target_size, target_size), interpolation=interpolation)
        if interpolation != cv2.INTER_NEAREST:
            resized = np.clip(resized, 0, 1)
        return resized

    single_channel = mask.shape[2] == 1
    resized = cv2.resize(mask, (target_size, target_size), interpolation=interpolation)
    if single_channel and resized.ndim == 2:
        resized = resized[..., np.newaxis]
    if interpolation != cv2.INTER_NEAREST:
        resized = np.clip(resized, 0, 1)
    return resized


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
    padding_value: int = 0,
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if image.ndim == 3:
        c = image.shape[2]
        canvas = np.full((target_size, target_size, c), padding_value, dtype=image.dtype)
    else:
        canvas = np.full((target_size, target_size), padding_value, dtype=image.dtype)

    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def normalise_rgb(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    scale: float = 255.0,
) -> np.ndarray:
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    image = image / scale
    for c in range(3):
        image[..., c] = (image[..., c] - mean[c]) / std[c]
    return image


def denormalise_rgb(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    scale: float = 255.0,
) -> np.ndarray:
    result = image.copy()
    for c in range(3):
        result[..., c] = result[..., c] * std[c] + mean[c]
    result = np.clip(result * scale, 0, 255).astype(np.uint8)
    return result


class ResizeTransform:
    def __init__(self, config: Optional[ResizeConfig] = None):
        self.config = config or DEFAULT_RESIZE_CONFIG

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, OriginalDimensions]:
        orig = OriginalDimensions(width=image.shape[1], height=image.shape[0])

        if self.config.preserve_aspect_ratio:
            resized = resize_with_aspect_ratio(
                image,
                target_size=self.config.target_size,
                interpolation=self.config.image_interpolation,
                padding_value=self.config.padding_value,
            )
        else:
            resized = resize_image(
                image,
                target_size=self.config.target_size,
                interpolation=self.config.image_interpolation,
                down_interpolation=self.config.image_down_interpolation,
            )

        return resized, orig

    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        return resize_mask(
            mask,
            target_size=self.config.target_size,
            interpolation=self.config.mask_interpolation,
        )


class NormaliseTransform:
    def __init__(self, config: Optional[NormaliseConfig] = None):
        self.config = config or DEFAULT_NORMALISE_CONFIG

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return normalise_rgb(
            image,
            mean=self.config.mean,
            std=self.config.std,
            scale=self.config.scale,
        )

    def inverse(self, image: np.ndarray) -> np.ndarray:
        return denormalise_rgb(
            image,
            mean=self.config.mean,
            std=self.config.std,
            scale=self.config.scale,
        )


class PreprocessingPipeline:
    def __init__(
        self,
        resize_config: Optional[ResizeConfig] = None,
        normalise_config: Optional[NormaliseConfig] = None,
    ):
        self.resize = ResizeTransform(resize_config)
        self.normalise = NormaliseTransform(normalise_config)
        self._last_dims: Optional[OriginalDimensions] = None

    @property
    def original_dimensions(self) -> Optional[OriginalDimensions]:
        return self._last_dims

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized, self._last_dims = self.resize(image)
        normalised = self.normalise(resized)
        return normalised

    def process(self, image: np.ndarray) -> np.ndarray:
        return self(image)

    def process_image_only(self, image: np.ndarray) -> np.ndarray:
        resized, self._last_dims = self.resize(image)
        return self.normalise(resized)

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        if self._last_dims is None:
            logger.warning("No last image dimensions; resizing mask to target directly")
        return self.resize.resize_mask(mask)

    def to_chw(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return np.transpose(image, (2, 0, 1))
        return image

    def to_hwc(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] in (1, 3):
            return np.transpose(image, (1, 2, 0))
        return image
