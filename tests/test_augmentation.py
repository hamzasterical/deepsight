import pytest


class TestAugmentationImports:
    def test_train_transform_importable(self):
        from src.preprocessing.augmentation import TRAIN_TRANSFORM
        assert TRAIN_TRANSFORM is not None

    def test_val_transform_importable(self):
        from src.preprocessing.augmentation import VAL_TRANSFORM
        assert VAL_TRANSFORM is not None

    def test_get_train_transform(self):
        from src.preprocessing.augmentation import get_train_transform
        transform = get_train_transform()
        assert transform is not None

    def test_get_val_transform(self):
        from src.preprocessing.augmentation import get_val_transform
        transform = get_val_transform()
        assert transform is not None

    def test_train_transform_has_image_compression(self):
        from src.preprocessing.augmentation import TRAIN_TRANSFORM
        transforms = [t for t in TRAIN_TRANSFORM]
        has_compression = any(
            "ImageCompression" in type(t).__name__ for t in transforms
        )
        assert has_compression

    def test_train_transform_has_normalize(self):
        from src.preprocessing.augmentation import TRAIN_TRANSFORM
        transforms = [t for t in TRAIN_TRANSFORM]
        has_normalize = any(
            "Normalize" in type(t).__name__ for t in transforms
        )
        assert has_normalize

    def test_val_transform_has_resize(self):
        from src.preprocessing.augmentation import VAL_TRANSFORM
        transforms = [t for t in VAL_TRANSFORM]
        has_resize = any(
            "Resize" in type(t).__name__ for t in transforms
        )
        assert has_resize
