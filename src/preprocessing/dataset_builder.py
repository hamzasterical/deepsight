from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.augmentation import (
    get_geometric_transform,
    get_normalize_transform,
    get_photometric_transform,
)
from src.preprocessing.ela import compute_ela, ela_to_3channel
from src.preprocessing.srm_filters import SRMFilterLayer, extract_srm_noise_batch
from src.utils.file_utils import ensure_dir, list_files, read_image_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dataset_metadata(
    config: dict,
) -> pd.DataFrame:
    raw_dir = Path(config["data"]["raw_dir"])
    splits_file = Path(config["data"]["splits_file"])
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    min_size_kb = config["preprocessing"].get("min_file_size_kb", 12)

    records = []

    casia_dir = raw_dir / "CASIA_v2"
    coverage_dir = raw_dir / "Coverage"
    korus_dir = raw_dir / "Korus"

    if casia_dir.exists():
        records.extend(_scan_casia(casia_dir, min_size_kb))
    if coverage_dir.exists():
        records.extend(_scan_coverage(coverage_dir, min_size_kb))
    if korus_dir.exists():
        records.extend(_scan_korus(korus_dir, min_size_kb))

    if not records:
        logger.warning("No dataset records found. Check raw data paths.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = _stratified_split(df, config["data"])
    df.to_csv(splits_file, index=False)
    logger.info(
        "Built metadata: %d images (%d train, %d val, %d test)",
        len(df),
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    return df


def _scan_casia(casia_dir: Path, min_size_kb: float) -> List[Dict]:
    records = []
    au_dir = casia_dir / "Au"
    tp_dir = casia_dir / "Tp"

    if au_dir.exists():
        for img_path in read_image_paths(au_dir, recursive=True, min_file_size_kb=min_size_kb):
            records.append({
                "image_path": str(img_path),
                "mask_path": "",
                "label": 0,
                "forgery_type": "Authentic",
                "dataset_source": "CASIA_v2",
            })

    if tp_dir.exists():
        for img_path in read_image_paths(tp_dir, recursive=True, min_file_size_kb=min_size_kb):
            mask_path = _find_casia_mask(img_path, tp_dir)
            ftype = _detect_casia_forgery_type(img_path)
            records.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path else "",
                "label": 1,
                "forgery_type": ftype,
                "dataset_source": "CASIA_v2",
            })

    return records


def _find_casia_mask(img_path: Path, tp_dir: Path) -> Optional[Path]:
    mask_name = img_path.stem + "_mask" + img_path.suffix
    mask_path = img_path.parent / "mask" / mask_name
    if mask_path.exists():
        return mask_path
    mask_path = tp_dir / "mask" / mask_name
    if mask_path.exists():
        return mask_path
    return None


def _detect_casia_forgery_type(img_path: Path) -> str:
    name = img_path.stem.lower()
    if "copy" in name or "move" in name:
        return "Copy-Move"
    return "Splicing"


def _scan_coverage(coverage_dir: Path, min_size_kb: float) -> List[Dict]:
    records = []
    image_dir = coverage_dir / "image"
    mask_dir = coverage_dir / "mask"

    if not image_dir.exists():
        return records

    mask_map = {}
    if mask_dir.exists():
        for mask_path in list_files(mask_dir, extensions=[".png", ".jpg", ".jpeg", ".tif", ".tiff"]):
            mask_map[mask_path.stem] = mask_path

    for img_path in read_image_paths(image_dir, recursive=True, min_file_size_kb=min_size_kb):
        mask_path = mask_map.get(img_path.stem)
        records.append({
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path else "",
            "label": 1,
            "forgery_type": "Copy-Move",
            "dataset_source": "Coverage",
        })

    return records


def _scan_korus(korus_dir: Path, min_size_kb: float) -> List[Dict]:
    records = []
    data_dir = korus_dir / "data-images"

    if not data_dir.exists():
        return records

    for img_path in read_image_paths(data_dir, recursive=True, min_file_size_kb=min_size_kb):
        name = img_path.stem.lower()
        if "tampered" in name or "forged" in name:
            records.append({
                "image_path": str(img_path),
                "mask_path": "",
                "label": 1,
                "forgery_type": "Retouching",
                "dataset_source": "Korus",
            })
        else:
            records.append({
                "image_path": str(img_path),
                "mask_path": "",
                "label": 0,
                "forgery_type": "Authentic",
                "dataset_source": "Korus",
            })

    return records


def _stratified_split(df: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    train_ratio = data_config.get("train_ratio", 0.8)
    val_ratio = data_config.get("val_ratio", 0.1)

    df = df.copy()
    df["split"] = "test"

    for ftype in df["forgery_type"].unique():
        type_idx = df[df["forgery_type"] == ftype].index.to_numpy(copy=True)
        np.random.shuffle(type_idx)
        n = len(type_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = type_idx[:n_train]
        val_idx = type_idx[n_train:n_train + n_val]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"

    return df


class ForgeryDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        transform=None,
        srm_layer: Optional[SRMFilterLayer] = None,
    ):
        self.data = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.split = split
        self.geometric = get_geometric_transform(split)
        self.photometric = get_photometric_transform(split)
        self.normalize = get_normalize_transform()
        self.srm_layer = srm_layer
        self._srm_cache = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        image = cv2.imread(row["image_path"])
        if image is None:
            return self.__getitem__((idx + 1) % len(self.data))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        mask_path = row["mask_path"]
        if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = mask.astype(np.float32) / 255.0

        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Geometric augmentation first: produces a fixed-size (224x224) image and
        # mask that stay spatially aligned. Forensic features are derived from the
        # SAME augmented image so the noise branch input matches the RGB input.
        geo = self.geometric(image=image, mask=mask)
        image_aug = geo["image"]
        mask_aug = geo["mask"]

        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_aug)).unsqueeze(0).float()

        ela = compute_ela(image_aug)
        ela_3ch = ela_to_3channel(ela)
        ela_tensor = torch.from_numpy(ela_3ch.transpose(2, 0, 1).astype(np.float32) / 255.0)
        if ela_tensor.shape[0] == 1:
            ela_tensor = ela_tensor.repeat(3, 1, 1)

        noise = extract_srm_noise_batch(image_aug[np.newaxis, ...], self.srm_layer)
        noise_tensor = torch.from_numpy(noise[0].transpose(2, 0, 1)).float()

        noise_input = torch.cat([noise_tensor, ela_tensor], dim=0)

        # Photometric augmentation (train only) + normalization for the RGB branch.
        rgb_img = image_aug
        if self.photometric is not None:
            rgb_img = self.photometric(image=image_aug)["image"]
        rgb = self.normalize(image=rgb_img)["image"].float()

        return {
            "rgb": rgb,
            "noise": noise_input,
            "label": torch.tensor([row["label"]], dtype=torch.float32),
            "mask": mask_tensor,
            "forgery_type": row["forgery_type"],
            "image_path": row["image_path"],
        }


def create_dataloaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    srm_layer: Optional[SRMFilterLayer] = None,
) -> Tuple:
    splits_file = Path(config["data"]["splits_file"])
    if not splits_file.exists() or splits_file.stat().st_size == 0:
        df = build_dataset_metadata(config)
    else:
        try:
            df = pd.read_csv(splits_file)
        except pd.errors.EmptyDataError:
            df = build_dataset_metadata(config)

    train_ds = ForgeryDataset(df, "train", srm_layer=srm_layer)
    val_ds = ForgeryDataset(df, "val", srm_layer=srm_layer)
    test_ds = ForgeryDataset(df, "test", srm_layer=srm_layer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
