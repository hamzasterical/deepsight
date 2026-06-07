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
    min_size_kb = config["preprocessing"].get("min_file_size_kb", 6.0)

    records = []

    casia_dir = raw_dir / "CASIA_v2"
    if casia_dir.exists():
        records.extend(_scan_casia(casia_dir, min_size_kb))

    before = len(records)
    records = _validate_records(records)
    after = len(records)
    if before - after > 0:
        logger.warning("Hard validation dropped %d/%d unreadable records", before - after, before)

    if not records:
        logger.warning("No dataset records found. Check raw data paths.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["forgery_type"] = (
        df["forgery_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    # Sanity-check: only expected types
    allowed = {"authentic", "splicing", "copy_move"}
    actual = set(df["forgery_type"].unique())
    unexpected = actual - allowed
    if unexpected:
        logger.warning("Unexpected forgery_type values (will be kept): %s", unexpected)

    df = _stratified_split(df, config["data"])
    df.to_csv(splits_file, index=False)

    logger.info(
        "Built metadata: %d images (%d train, %d val, %d test)",
        len(df),
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    logger.info("Label breakdown:  authentic=%d  forged=%d",
                 (df["label"] == 0).sum(), (df["label"] == 1).sum())
    logger.info("Forgery-type breakdown:\n%s", df["forgery_type"].value_counts().to_string())
    logger.info("Split × forgery-type breakdown:\n%s",
                df.groupby(["split", "forgery_type"]).size().to_string())

    return df


def _is_readable_image(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    img = cv2.imread(str(p))
    return img is not None and img.ndim == 3 and img.shape[2] == 3


def _is_readable_mask(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return m is not None and m.ndim == 2


def _validate_records(records: List[Dict]) -> List[Dict]:
    """Drop unreadable images. Forged records without masks are allowed
    (mask branch receives a zero target — classification still trains correctly)."""
    good = []
    for r in records:
        if not _is_readable_image(r["image_path"]):
            logger.debug("Dropping unreadable image: %s", r["image_path"])
            continue
        good.append(r)
    return good


# ── CASIA v2 ─────────────────────────────────────────────────────────────────

def _resolve_casia_subdirs(casia_dir: Path):
    """Return (au_dir, tp_dir) regardless of whether dirs are named
    Au/Tp (older download) or Au_jpg/Tp_jpg (Kaggle download)."""
    au_candidates = ["Au", "Au_jpg", "au", "au_jpg"]
    tp_candidates = ["Tp", "Tp_jpg", "tp", "tp_jpg"]

    au_dir = None
    for name in au_candidates:
        candidate = casia_dir / name
        if candidate.exists():
            au_dir = candidate
            break

    tp_dir = None
    for name in tp_candidates:
        candidate = casia_dir / name
        if candidate.exists():
            tp_dir = candidate
            break

    return au_dir, tp_dir


def _scan_casia(casia_dir: Path, min_size_kb: float) -> List[Dict]:
    records = []
    au_dir, tp_dir = _resolve_casia_subdirs(casia_dir)

    if au_dir:
        logger.info("CASIA authentic dir: %s", au_dir)
        for img_path in read_image_paths(au_dir, recursive=True, min_file_size_kb=min_size_kb):
            records.append({
                "image_path": str(img_path),
                "mask_path": "",
                "label": 0,
                "forgery_type": "authentic",
                "dataset_source": "CASIA_v2",
            })
    else:
        logger.warning("CASIA: no authentic (Au/Au_jpg) directory found under %s", casia_dir)

    if tp_dir:
        logger.info("CASIA tampered dir: %s", tp_dir)
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
    else:
        logger.warning("CASIA: no tampered (Tp/Tp_jpg) directory found under %s", casia_dir)

    logger.info("CASIA scan complete: %d total records", len(records))
    return records


def _find_casia_mask(img_path: Path, tp_dir: Path) -> Optional[Path]:
    candidates = []

    candidates.append(img_path.parent / "mask" / f"{img_path.stem}_mask{img_path.suffix}")
    candidates.append(tp_dir / "mask" / f"{img_path.stem}_mask{img_path.suffix}")

    gt_dir = tp_dir / "GT"
    candidates.append(gt_dir / f"{img_path.stem}.png")
    candidates.append(gt_dir / f"{img_path.stem}_gt.png")
    candidates.append(gt_dir / f"{img_path.stem}_mask.png")

    candidates.append(img_path.parent / "GT" / f"{img_path.stem}.png")
    candidates.append(img_path.parent / "GT" / f"{img_path.stem}_gt.png")

    for p in candidates:
        if p.exists():
            return p

    return None


def _detect_casia_forgery_type(img_path: Path) -> str:
    stem = img_path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        op = parts[1].upper()
        if op == "D":
            return "Copy-Move"
        if op == "S":
            return "Splicing"
    return "Splicing"


# ── Split ─────────────────────────────────────────────────────────────────────

def _stratified_split(df: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    train_ratio = data_config.get("train_ratio", 0.8)
    val_ratio = data_config.get("val_ratio", 0.1)

    rng = np.random.RandomState(data_config.get("seed", 42))

    df = df.copy()
    df["split"] = "test"

    df["strata"] = df["label"].astype(str) + "_" + df["forgery_type"].astype(str)

    for strata in df["strata"].unique():
        idx = df[df["strata"] == strata].index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"

    df = df.drop(columns=["strata"])
    return df


# ── Dataset ───────────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        transform=None,
        srm_layer: Optional[SRMFilterLayer] = None,
        ela_amplify: float = 20.0,
    ):
        self.data = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.split = split
        self.geometric = get_geometric_transform(split)
        self.photometric = get_photometric_transform(split)
        self.normalize = get_normalize_transform()
        self.srm_layer = srm_layer
        self.ela_amplify = ela_amplify
        self._srm_cache = {}
        self._logged_missing: set = set()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        max_attempts = len(self.data)
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.data)
            try:
                row = self.data.iloc[current_idx]
                sample = self._load_sample(row)
                if sample is not None:
                    return sample
            except Exception as e:
                img_path = str(row.get("image_path", current_idx)) if attempt == 0 else None
                if img_path and img_path not in self._logged_missing:
                    self._logged_missing.add(img_path)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Skipping unreadable sample (attempt {attempt}): {img_path} — {e}"
                    )
                continue

        raise RuntimeError(
            f"ForgeryDataset: could not find ANY valid sample after scanning "
            f"all {max_attempts} entries. Check that data/raw/ files exist on disk."
        )

    def _load_sample(self, row) -> dict | None:
        import cv2
        import numpy as np

        image_path = str(row["image_path"])
        label      = int(row["label"])
        forgery_type = str(row.get("forgery_type", "authentic"))

        img = cv2.imread(image_path)
        if img is None:
            return None
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        geo = self.geometric(image=img)
        image_aug = geo["image"]

        ela = compute_ela(image_aug, amplify=self.ela_amplify)
        ela_3ch = ela_to_3channel(ela)
        ela_tensor = torch.from_numpy(ela_3ch.transpose(2, 0, 1).astype(np.float32) / 255.0)
        if ela_tensor.shape[0] == 1:
            ela_tensor = ela_tensor.repeat(3, 1, 1)

        noise = extract_srm_noise_batch(image_aug[np.newaxis, ...], self.srm_layer)
        noise_tensor = torch.from_numpy(noise[0].transpose(2, 0, 1)).float()

        noise_input = torch.cat([noise_tensor, ela_tensor], dim=0)

        rgb_img = image_aug
        if self.photometric is not None:
            rgb_img = self.photometric(image=image_aug)["image"]
        rgb = self.normalize(image=rgb_img)["image"].float()

        return {
            "rgb": rgb,
            "noise": noise_input,
            "label": torch.tensor([label], dtype=torch.float32),
            "forgery_type": forgery_type,
            "image_path": image_path,
        }


def _safe_collate(batch):
    """Drop None entries that slipped through __getitem__."""
    import torch
    from torch.utils.data.dataloader import default_collate

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def create_dataloaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    srm_layer: Optional[SRMFilterLayer] = None,
) -> Tuple:
    df = build_dataset_metadata(config)

    ela_amplify = float(config.get("preprocessing", {}).get("ela_amplify", 20.0))
    train_ds = ForgeryDataset(df, "train", srm_layer=srm_layer, ela_amplify=ela_amplify)
    val_ds = ForgeryDataset(df, "val", srm_layer=srm_layer, ela_amplify=ela_amplify)
    test_ds = ForgeryDataset(df, "test", srm_layer=srm_layer, ela_amplify=ela_amplify)

    # ── Weighted sampler to handle class imbalance ────────────────────────────
    use_sampler = config.get("training", {}).get("use_weighted_sampler", True)

    if use_sampler:
        labels = [int(train_ds.data.iloc[i]["label"]) for i in range(len(train_ds.data))]
        class_counts = [labels.count(0), labels.count(1)]
        class_weights = [1.0 / max(c, 1) for c in class_counts]
        sample_weights = [class_weights[l] for l in labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            pin_memory=True, collate_fn=_safe_collate,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, collate_fn=_safe_collate,
        )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_safe_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_safe_collate,
    )

    return train_loader, val_loader, test_loader
