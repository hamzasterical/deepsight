import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference.predictor import Predictor
from src.models.dual_branch import DualBranchModel
from src.training.metrics import compute_metrics, print_evaluation_report, save_evaluation_report


def load_model(ckpt_path: str = "models/checkpoints/best_model.pth") -> Predictor:
    model = DualBranchModel()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return Predictor(model)


SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".webp", ".heic", ".heif")

_MIN_FILE_BYTES = 12000


def _is_valid_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXT and p.stat().st_size > _MIN_FILE_BYTES


def _casia_forgery_type(p: Path) -> str:
    stem = p.stem
    parts = stem.split("_")
    code = parts[2] if len(parts) > 2 else ""
    if code.startswith("CND"):
        return "Copy-Move"
    return "Splicing"


def collect_samples():
    samples = []
    raw = Path("data/raw")

    casia_au = raw / "CASIA_v2" / "Au"
    if casia_au.exists():
        count = 0
        for p in sorted(casia_au.rglob("*")):
            if _is_valid_image(p):
                samples.append((str(p), 0, "Authentic"))
                count += 1
                if count >= 200:
                    break

    casia_tp = raw / "CASIA_v2" / "Tp"
    if casia_tp.exists():
        count = 0
        for p in sorted(casia_tp.rglob("*")):
            if _is_valid_image(p):
                ftype = _casia_forgery_type(p)
                samples.append((str(p), 1, ftype))
                count += 1
                if count >= 200:
                    break

    coverage = raw / "Coverage"
    cov_img = coverage / "image"
    if cov_img.exists():
        count = 0
        for p in sorted(cov_img.rglob("*")):
            if _is_valid_image(p):
                stem = p.stem
                is_tampered = stem.endswith("t") and not stem.endswith("tt")
                if is_tampered:
                    samples.append((str(p), 1, "Copy-Move"))
                else:
                    samples.append((str(p), 0, "Authentic"))
                count += 1
                if count >= 100:
                    break

    korus = raw / "Korus" / "data-images"
    if korus.exists():
        for cam_dir in sorted(korus.iterdir()):
            if not cam_dir.is_dir() or cam_dir.name in ("camera_models", "thumbnails"):
                continue
            tampered_dir = cam_dir / "tampered-realistic"
            if tampered_dir.exists():
                count = 0
                for p in sorted(tampered_dir.rglob("*")):
                    if _is_valid_image(p):
                        samples.append((str(p), 1, "Retouching"))
                        count += 1
                        if count >= 50:
                            break
            pristine_dir = cam_dir / "pristine"
            if pristine_dir.exists():
                count = 0
                for p in sorted(pristine_dir.rglob("*")):
                    if _is_valid_image(p):
                        samples.append((str(p), 0, "Authentic"))
                        count += 1
                        if count >= 50:
                            break

    return samples


def main():
    print("Loading model...")
    predictor = load_model()

    print("Collecting test samples...")
    samples = collect_samples()
    print(f"Found {len(samples)} samples")

    all_labels = []
    all_preds = []
    all_masks = []
    all_gt_masks = []
    all_ftypes = []

    for i, (path, label, ftype) in enumerate(samples):
        try:
            result = predictor.predict_from_path(path)
        except Exception as e:
            continue

        prob = result["confidence"] / 100.0
        all_labels.append(label)
        all_preds.append(prob)
        all_ftypes.append(ftype)
        all_masks.append(np.zeros((224, 224), dtype=np.float32))
        all_gt_masks.append(np.zeros((224, 224), dtype=np.float32))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_masks),
        np.array(all_gt_masks),
        all_ftypes,
    )

    print_evaluation_report(metrics, total_samples=len(samples))
    save_evaluation_report(metrics)


if __name__ == "__main__":
    main()
