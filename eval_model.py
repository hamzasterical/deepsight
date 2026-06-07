import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference.predictor import Predictor
from src.models.dual_branch import DualBranchModel
from src.training.metrics import compute_metrics, print_evaluation_report, save_evaluation_report


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str = "models/checkpoints/best_model.pth", config: dict = None) -> Predictor:
    model = DualBranchModel()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return Predictor(model, config=config)


SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".webp", ".heic", ".heif")

def _is_valid_image(p: Path, min_file_size_kb: float = 6.0) -> bool:
    return p.suffix.lower() in SUPPORTED_EXT and p.stat().st_size >= min_file_size_kb * 1024


def _casia_forgery_type(p: Path) -> str:
    stem = p.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        op = parts[1].upper()
        if op == "D":
            return "Copy-Move"
        if op == "S":
            return "Splicing"
    return "Splicing"


def _resolve_casia_dirs(raw: Path):
    """Return (au_dir, tp_dir) for either naming convention."""
    casia = raw / "CASIA_v2"
    au_dir = None
    for name in ["Au", "Au_jpg", "au", "au_jpg"]:
        c = casia / name
        if c.exists():
            au_dir = c
            break
    tp_dir = None
    for name in ["Tp", "Tp_jpg", "tp", "tp_jpg"]:
        c = casia / name
        if c.exists():
            tp_dir = c
            break
    return au_dir, tp_dir


def collect_samples(min_file_size_kb: float = 6.0):
    samples = []
    raw = Path("data/raw")

    au_dir, tp_dir = _resolve_casia_dirs(raw)

    # Authentic samples
    if au_dir and au_dir.exists():
        count = 0
        for p in sorted(au_dir.rglob("*")):
            if _is_valid_image(p, min_file_size_kb):
                samples.append((str(p), 0, "Authentic"))
                count += 1
                if count >= 300:
                    break

    # Tampered samples (Copy-Move + Splicing)
    if tp_dir and tp_dir.exists():
        count = 0
        for p in sorted(tp_dir.rglob("*")):
            if _is_valid_image(p, min_file_size_kb):
                ftype = _casia_forgery_type(p)
                samples.append((str(p), 1, ftype))
                count += 1
                if count >= 300:
                    break

    return samples


def find_optimal_threshold(y_true, y_scores, metric="f1"):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    import numpy as np

    thresholds = np.arange(0.10, 0.91, 0.05)
    best_thresh = 0.5
    best_score  = 0.0

    print("\nThreshold sweep:")
    print(f"{'Threshold':>10} {'F1':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 52)

    for t in thresholds:
        preds = (np.array(y_scores) >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        acc   = accuracy_score(y_true, preds)
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        print(f"{t:>10.2f} {f1:>8.4f} {acc:>10.4f} {prec:>10.4f} {rec:>8.4f}")

        score = f1 if metric == "f1" else acc
        if score > best_score:
            best_score  = score
            best_thresh = t

    print(f"\n-> Best threshold for {metric}: {best_thresh:.2f}  (score={best_score:.4f})")
    print(f"  Set inference.confidence_threshold: {best_thresh:.2f} in configs/config.yaml")
    return best_thresh


def main():
    config = load_config()
    threshold = float(config.get("inference", {}).get("confidence_threshold", 0.45))
    print(f"Using confidence threshold: {threshold:.2f} (from configs/config.yaml)")

    print("Loading model...")
    predictor = load_model(config=config)

    min_file_size_kb = float(config.get("preprocessing", {}).get("min_file_size_kb", 6.0))
    print("Collecting test samples...")
    samples = collect_samples(min_file_size_kb)
    print(f"Found {len(samples)} samples")

    all_labels = []
    all_preds = []
    all_scores = []
    all_ftypes = []

    for i, (path, label, ftype) in enumerate(samples):
        try:
            result = predictor.predict_from_path(path)
        except Exception as e:
            if (i + 1) % 50 == 0 or i < 3:
                print(f"  [{i}] SKIP {path}: {e}")
            continue

        prob = result["confidence"] / 100.0
        all_labels.append(label)
        all_preds.append(prob)
        all_scores.append(prob)
        all_ftypes.append(ftype)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # --- Copy-Move score distribution debug ---
    cm_indices = [i for i, ft in enumerate(all_ftypes) if ft == "Copy-Move"]
    if cm_indices:
        cm_scores = np.array(all_scores)[cm_indices]
        cm_labels = np.array(all_labels)[cm_indices]
        print(f"\n[DEBUG] Copy-Move ({len(cm_indices)} samples):")
        print(f"  Scores — min={cm_scores.min():.4f}  max={cm_scores.max():.4f}  mean={cm_scores.mean():.4f}")
        print(f"  Above threshold ({threshold:.2f}): {(cm_scores >= threshold).sum()} / {len(cm_scores)}")
        print(f"  Labels: {cm_labels.tolist()[:20]}{'...' if len(cm_labels) > 20 else ''}")
    else:
        print("\n[DEBUG] No Copy-Move samples were collected during this evaluation run.")

    # --- Splicing score distribution debug ---
    sp_indices = [i for i, ft in enumerate(all_ftypes) if ft == "Splicing"]
    if sp_indices:
        sp_scores = np.array(all_scores)[sp_indices]
        print(f"\n[DEBUG] Splicing ({len(sp_indices)} samples):")
        print(f"  Scores — min={sp_scores.min():.4f}  max={sp_scores.max():.4f}  mean={sp_scores.mean():.4f}")
        print(f"  Above threshold ({threshold:.2f}): {(sp_scores >= threshold).sum()} / {len(sp_scores)}")

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        all_ftypes,
        threshold=threshold,
    )

    print_evaluation_report(metrics, total_samples=len(samples))
    save_evaluation_report(metrics)

    if len(all_labels) > 0 and len(all_scores) > 0:
        find_optimal_threshold(all_labels, all_scores, metric="f1")


if __name__ == "__main__":
    main()
