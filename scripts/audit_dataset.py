"""
DeepSight — Dataset Audit Script
Run this BEFORE training to find all missing/corrupt files referenced
in split_metadata.csv so they can be removed from the CSV or re-downloaded.

Usage:
    python scripts/audit_dataset.py              # fast: checks file existence
    python scripts/audit_dataset.py --deep       # slow: also verifies OpenCV decode
    python scripts/audit_dataset.py --fix        # removes bad rows from CSV in-place
    python scripts/audit_dataset.py --deep --fix # full verification + fix
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


CSV_PATH = Path("data/splits/split_metadata.csv")
RAW_DIR  = Path("data/raw")


def audit(fix: bool = False, deep: bool = False):
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run dataset_builder first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Auditing {total} rows in {CSV_PATH} ...")

    if deep:
        import cv2

    bad_rows = []
    missing_images = []
    corrupt_images = []
    missing_masks = []

    for idx, row in df.iterrows():
        img_path  = str(row["image_path"])
        mask_path = str(row.get("mask_path", ""))
        label     = int(row["label"])

        # Check image file exists
        if not os.path.isfile(img_path):
            missing_images.append(img_path)
            bad_rows.append(idx)
            continue

        # Deep check: verify OpenCV can decode
        if deep:
            img = cv2.imread(img_path)
            if img is None:
                corrupt_images.append(img_path)
                bad_rows.append(idx)
                continue

        # Check mask for forged images (non-fatal)
        if label == 1 and mask_path and mask_path != "nan":
            if not os.path.isfile(mask_path):
                missing_masks.append(mask_path)

    bad_rows = list(set(bad_rows))

    print(f"\nResults:")
    print(f"  Total rows            : {total}")
    print(f"  Missing image files   : {len(missing_images)}")
    print(f"  Corrupt image files   : {len(corrupt_images)}  (--deep only)")
    print(f"  Missing mask files    : {len(missing_masks)}  (non-fatal, zeros used)")
    print(f"  Total BAD rows        : {len(bad_rows)}")
    print(f"  Usable rows           : {total - len(bad_rows)}")

    if missing_images:
        print(f"\nFirst 20 missing images:")
        for p in missing_images[:20]:
            print(f"  MISSING  {p}")

    if corrupt_images:
        print(f"\nFirst 20 corrupt images:")
        for p in corrupt_images[:20]:
            print(f"  CORRUPT  {p}")

    if missing_masks:
        print(f"\nFirst 20 missing masks (non-fatal):")
        for p in missing_masks[:20]:
            print(f"  NO MASK  {p}")

    if fix and bad_rows:
        backup_path = CSV_PATH.with_suffix(".csv.bak")
        df.to_csv(backup_path, index=False)
        print(f"\nBackup saved to {backup_path}")

        df_clean = df.drop(index=bad_rows).reset_index(drop=True)
        df_clean.to_csv(CSV_PATH, index=False)
        print(f"Removed {len(bad_rows)} bad rows. Clean CSV saved to {CSV_PATH}")
        print(f"New totals:")
        for split in ["train", "val", "test"]:
            n = (df_clean["split"] == split).sum()
            print(f"  {split}: {n}")
    elif bad_rows:
        print(f"\nRun with --fix to remove {len(bad_rows)} bad rows from the CSV.")

    return len(bad_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove bad rows from CSV in-place (backs up original first)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Slow check: also verify each file decodes with OpenCV",
    )
    args = parser.parse_args()
    n_bad = audit(fix=args.fix, deep=args.deep)
    sys.exit(0 if n_bad == 0 else 1)
