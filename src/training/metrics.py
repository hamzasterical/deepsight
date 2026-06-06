from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    pred_binary = (pred_mask > threshold).astype(np.int32)
    gt_binary = (gt_mask > threshold).astype(np.int32)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    masks: np.ndarray,
    gt_masks: np.ndarray,
    forgery_types: Optional[List[str]] = None,
) -> Dict[str, dict]:
    results = {}

    if forgery_types is not None:
        unique_types = list(set(forgery_types))
        for ftype in unique_types:
            idx = [i for i, t in enumerate(forgery_types) if t == ftype]
            if len(idx) < 2:
                continue
            ftype_labels = labels[idx]
            ftype_preds = preds[idx]
            ftype_key = ftype.lower().replace("-", "_").replace(" ", "_")
            results[ftype_key] = _compute_single(ftype_labels, ftype_preds, masks[idx], gt_masks[idx])

    results["overall"] = _compute_single(labels, preds, masks, gt_masks)
    return results


def _compute_single(
    labels: np.ndarray,
    preds: np.ndarray,
    masks: np.ndarray,
    gt_masks: np.ndarray,
) -> dict:
    result = {}
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()

    if len(labels) == 0:
        return {"auc_roc": 0.0, "f1": 0.0, "iou": 0.0, "accuracy": 0.0, "n_samples": 0}

    if len(np.unique(labels)) >= 2:
        result["auc_roc"] = float(roc_auc_score(labels, preds))
    else:
        result["auc_roc"] = float(labels[0])

    result["f1"] = float(f1_score(labels, (preds > 0.5).astype(np.int32)))

    ious = []
    for i in range(len(masks)):
        ious.append(compute_iou(masks[i], gt_masks[i]))
    result["iou"] = float(np.mean(ious)) if ious else 0.0

    result["accuracy"] = float(np.mean((preds > 0.5) == labels))
    result["n_samples"] = len(labels)
    return result


def batch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def print_evaluation_report(metrics: dict, total_samples: int = 0) -> None:
    header = f"{'Category':<20} {'AUC':<10} {'F1':<10} {'Accuracy':<10} {'IoU':<10} {'Samples':<8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("EVALUATION REPORT")
    print(sep)
    if total_samples:
        print(f"Total samples: {total_samples}")
    print(sep)
    print(header)
    print(sep)

    for key in sorted(metrics.keys()):
        m = metrics[key]
        auc = m.get("auc_roc", 0)
        f1 = m.get("f1", 0)
        acc = m.get("accuracy", 0)
        iou = m.get("iou", 0)
        n = m.get("n_samples", "")
        label = key.replace("_", " ").title()
        print(f"{label:<20} {auc:<10.4f} {f1:<10.4f} {acc:<10.4f} {iou:<10.4f} {str(n):<8}")

    print(sep + "\n")


def save_evaluation_report(metrics: dict, output_path: str = "models/exported/evaluation_report.txt") -> None:
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("DEEPSIGHT EVALUATION REPORT")
    lines.append("=" * 60)

    for key in sorted(metrics.keys()):
        m = metrics[key]
        label = key.replace("_", " ").title()
        lines.append(f"\n{label}:")
        lines.append(f"  AUC-ROC : {m.get('auc_roc', 0):.4f}")
        lines.append(f"  F1      : {m.get('f1', 0):.4f}")
        lines.append(f"  Accuracy: {m.get('accuracy', 0):.4f}")
        lines.append(f"  IoU     : {m.get('iou', 0):.4f}")

    lines.append("\n" + "=" * 60)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")
