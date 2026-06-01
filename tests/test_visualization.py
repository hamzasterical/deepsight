import numpy as np
import pytest

from src.utils.visualization import create_comparison_grid, create_heatmap_overlay, create_red_overlay


class TestCreateHeatmapOverlay:
    def test_output_shape(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        assert overlay.shape == (100, 100, 3)

    def test_output_dtype(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        assert overlay.dtype == np.uint8

    def test_output_values_in_range(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_resizes_mask_if_needed(self):
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        assert overlay.shape == (200, 200, 3)


class TestCreateRedOverlay:
    def test_output_shape(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_red_overlay(image, mask)
        assert overlay.shape == (100, 100, 3)

    def test_output_dtype(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_red_overlay(image, mask)
        assert overlay.dtype == np.uint8

    def test_custom_color(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_red_overlay(image, mask, color=(0, 255, 0))
        assert overlay.shape == (100, 100, 3)


class TestCreateComparisonGrid:
    def test_output_shape(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        grid = create_comparison_grid(image, mask, overlay)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_output_dtype(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100).astype(np.float32)
        overlay = create_heatmap_overlay(image, mask)
        grid = create_comparison_grid(image, mask, overlay)
        assert grid.dtype == np.uint8
