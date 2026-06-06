import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Combined pipelines kept for backward compatibility (used by tests / callers
# that only need an RGB tensor).
TRAIN_TRANSFORM = A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.ImageCompression(quality_range=(50, 95), p=0.5),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def get_train_transform():
    return TRAIN_TRANSFORM


def get_val_transform():
    return VAL_TRANSFORM


# Split pipelines so that the forensic feature channels (ELA / SRM noise) can be
# computed on the geometrically-augmented, fixed-size image. This guarantees the
# noise branch input is spatially aligned with the RGB branch input and the mask,
# and that every sample in a batch has identical spatial dimensions.
GEOMETRIC_TRAIN = A.Compose([
    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
    ),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

GEOMETRIC_VAL = A.Compose([
    A.Resize(224, 224),
])

PHOTOMETRIC_TRAIN = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.ImageCompression(quality_range=(50, 95), p=0.5),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])

NORMALIZE = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def get_geometric_transform(split: str = "train"):
    return GEOMETRIC_TRAIN if split == "train" else GEOMETRIC_VAL


def get_photometric_transform(split: str = "train"):
    return PHOTOMETRIC_TRAIN if split == "train" else None


def get_normalize_transform():
    return NORMALIZE
