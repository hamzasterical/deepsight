import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

SRM_FILTER_COUNT: int = 30
SRM_KERNEL_SIZE: int = 5
SRM_PADDING: int = 2


def _normalize_filter(f: np.ndarray) -> np.ndarray:
    norm = np.sqrt((f ** 2).sum())
    return f / norm if norm > 0 else f


def _pad_kernel_3x3_to_5x5(k: np.ndarray) -> np.ndarray:
    padded = np.zeros((5, 5), dtype=np.float64)
    padded[1:4, 1:4] = k
    return padded


def _build_srm_filters_3x3() -> list:
    f1 = np.array([[-1, 2, -1],
                   [2, -4, 2],
                   [-1, 2, -1]], dtype=np.float64)

    f2 = np.array([[-1, 2, -2],
                   [2, -4, 2],
                   [-1, 2, -1]], dtype=np.float64)

    f3 = np.array([[1, -2, 1],
                   [-2, 4, -2],
                   [1, -2, 1]], dtype=np.float64)

    f4 = np.array([[-1, 2, -1],
                   [2, -4, 2],
                   [-1, 2, -1]], dtype=np.float64) * -1

    return [_pad_kernel_3x3_to_5x5(f) for f in [f1, f2, f3, f4]]


def _build_srm_filters_5x5_edge() -> list:
    kernels = []

    sobel_h = np.array([[-1, -2, 0, 2, 1],
                        [-2, -3, 0, 3, 2],
                        [-3, -5, 0, 5, 3],
                        [-2, -3, 0, 3, 2],
                        [-1, -2, 0, 2, 1]], dtype=np.float64)
    kernels.append(sobel_h)

    sobel_v = np.array([[-1, -2, -3, -2, -1],
                        [-2, -3, -5, -3, -2],
                        [0, 0, 0, 0, 0],
                        [2, 3, 5, 3, 2],
                        [1, 2, 3, 2, 1]], dtype=np.float64)
    kernels.append(sobel_v)

    diag_1 = np.array([[0, 0, 1, 2, 1],
                       [0, 1, 2, 4, 2],
                       [-1, -2, 0, 2, 1],
                       [-2, -4, -2, -1, 0],
                       [-1, -2, -1, 0, 0]], dtype=np.float64)
    kernels.append(diag_1)

    diag_2 = np.array([[-1, -2, -1, 0, 0],
                       [-2, -4, -2, -1, 0],
                       [-1, -2, 0, 2, 1],
                       [0, -1, 2, 4, 2],
                       [0, 0, 1, 2, 1]], dtype=np.float64)
    kernels.append(diag_2)

    return kernels


def _build_srm_filters_5x5_laplacian() -> list:
    kernels = []

    lap_1 = np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]], dtype=np.float64)
    kernels.append(lap_1)

    lap_2 = np.array([[0, 1, 1, 1, 0],
                      [1, 0, -1, 0, 1],
                      [1, -1, -4, -1, 1],
                      [1, 0, -1, 0, 1],
                      [0, 1, 1, 1, 0]], dtype=np.float64)
    kernels.append(lap_2)

    lap_3 = np.array([[-1, 0, 0, 0, 1],
                      [0, -1, 0, 1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, -1, 0],
                      [1, 0, 0, 0, -1]], dtype=np.float64)
    kernels.append(lap_3)

    return kernels


def _build_srm_filters_5x5_noise() -> list:
    kernels = []

    n1 = np.array([[1, 0, 0, 0, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 1]], dtype=np.float64)
    kernels.append(n1)

    n2 = np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, -1]], dtype=np.float64)
    kernels.append(n2)

    n3 = np.array([[0, 0, 0, 0, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]], dtype=np.float64)
    kernels.append(n3)

    n4 = np.array([[0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0]], dtype=np.float64)
    kernels.append(n4)

    n5 = np.array([[0, 1, 0, -1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, -1, 0]], dtype=np.float64)
    kernels.append(n5)

    n6 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]], dtype=np.float64)
    kernels.append(n6)

    n7 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 1]], dtype=np.float64)
    kernels.append(n7)

    n8 = np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, -1]], dtype=np.float64)
    kernels.append(n8)

    n9 = np.array([[0, 0, 0, 0, 0],
                   [0, 1, 0, -1, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float64)
    kernels.append(n9)

    return kernels


def _build_srm_filters_5x5_quad() -> list:
    kernels = []

    q1 = np.array([[1, 0, 0, 0, 1],
                   [0, -1, 0, -1, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 0, -1, 0],
                   [1, 0, 0, 0, 1]], dtype=np.float64)
    kernels.append(q1)

    q2 = np.array([[1, 0, 0, 0, -1],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, -1, 0],
                   [-1, 0, 0, 0, 1]], dtype=np.float64)
    kernels.append(q2)

    q3 = np.array([[1, 0, -1, 0, 1],
                   [0, -1, 0, -1, 0],
                   [-1, 0, 4, 0, -1],
                   [0, -1, 0, -1, 0],
                   [1, 0, -1, 0, 1]], dtype=np.float64)
    kernels.append(q3)

    q4 = np.array([[-1, 0, 1, 0, -1],
                   [0, -1, 0, -1, 0],
                   [1, 0, 4, 0, 1],
                   [0, -1, 0, -1, 0],
                   [-1, 0, 1, 0, -1]], dtype=np.float64)
    kernels.append(q4)

    return kernels


def _build_srm_filters_5x5_cross() -> list:
    kernels = []

    c1 = np.array([[0, -1, 0, -1, 0],
                   [-1, 2, 0, 2, -1],
                   [0, 0, 0, 0, 0],
                   [-1, 2, 0, 2, -1],
                   [0, -1, 0, -1, 0]], dtype=np.float64)
    kernels.append(c1)

    c2 = np.array([[0, -1, 0, 1, 0],
                   [-1, 2, 0, -2, 1],
                   [0, 0, 0, 0, 0],
                   [1, -2, 0, 2, -1],
                   [0, 1, 0, -1, 0]], dtype=np.float64)
    kernels.append(c2)

    c3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]], dtype=np.float64)
    kernels.append(c3)

    return kernels


def _build_srm_filters_5x5_random() -> list:
    kernels = []

    r1 = np.array([[1, -1, 0, 1, -1],
                   [-1, 1, 0, -1, 1],
                   [0, 0, 0, 0, 0],
                   [1, -1, 0, 1, -1],
                   [-1, 1, 0, -1, 1]], dtype=np.float64)
    kernels.append(r1)

    r2 = np.array([[-1, 1, 0, -1, 1],
                   [1, -1, 0, 1, -1],
                   [0, 0, 0, 0, 0],
                   [-1, 1, 0, -1, 1],
                   [1, -1, 0, 1, -1]], dtype=np.float64)
    kernels.append(r2)

    r3 = np.array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 1, 0],
                   [0, 0, -4, 0, 0],
                   [0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float64)
    kernels.append(r3)

    return kernels


def get_srm_filters() -> np.ndarray:
    filters = []

    filters.extend(_build_srm_filters_3x3())
    filters.extend(_build_srm_filters_5x5_edge())
    filters.extend(_build_srm_filters_5x5_laplacian())
    filters.extend(_build_srm_filters_5x5_noise())
    filters.extend(_build_srm_filters_5x5_quad())
    filters.extend(_build_srm_filters_5x5_cross())
    filters.extend(_build_srm_filters_5x5_random())

    assert len(filters) == SRM_FILTER_COUNT, (
        f"Expected {SRM_FILTER_COUNT} SRM filters, got {len(filters)}"
    )

    normalized = np.stack([_normalize_filter(f) for f in filters], axis=0)
    return normalized.astype(np.float32)


def get_srm_weight_tensor() -> torch.Tensor:
    filters = get_srm_filters()
    weight = np.zeros((SRM_FILTER_COUNT, 3, SRM_KERNEL_SIZE, SRM_KERNEL_SIZE),
                      dtype=np.float32)
    for i in range(SRM_FILTER_COUNT):
        for c in range(3):
            weight[i, c] = filters[i]
    return torch.from_numpy(weight)


class SRMFilterLayer(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = SRM_FILTER_COUNT):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=SRM_KERNEL_SIZE,
            padding=SRM_PADDING,
            bias=False,
        )
        weight = get_srm_weight_tensor()
        self.conv.weight = nn.Parameter(weight, requires_grad=False)
        self.conv.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def extract_srm_noise(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        layer = SRMFilterLayer()
        layer.eval()
        noise = layer(tensor)

    noise_np = noise.squeeze(0).permute(1, 2, 0).numpy()
    return noise_np


def extract_srm_noise_batch(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected NHWC batch, got shape {images.ndim}")

    dtype_in = images.dtype
    if images.dtype == np.uint8:
        images = images.astype(np.float32)

    tensor = torch.from_numpy(images).permute(0, 3, 1, 2)

    with torch.no_grad():
        layer = SRMFilterLayer()
        layer.eval()
        noise = layer(tensor)

    noise_np = noise.permute(0, 2, 3, 1).numpy()
    return noise_np
