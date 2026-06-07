from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    forgery_types: Optional[List[str]] = None,
    threshold: float = 0.45,
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
            results[ftype_key] = _compute_single(ftype_labels, ftype_preds, threshold=threshold)

    results["overall"] = _compute_single(labels, preds, threshold=threshold)
    return results


def _compute_single(
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.45,
) -> dict:
    result = {}
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()

    if len(labels) == 0:
        return {"auc_roc": float("nan"), "f1": 0.0, "accuracy": 0.0, "n_samples": 0}

    if len(np.unique(labels)) >= 2:
        result["auc_roc"] = float(roc_auc_score(labels, preds))
    else:
        # roc_auc_score requires at least 2 distinct classes.
        # Return NaN to indicate the metric is undefined for this single-class group
        # (e.g. Authentic-only group has no forged samples to rank against).
        result["auc_roc"] = float("nan")

    binary_preds = (preds >= threshold).astype(np.int32)
    result["f1"] = float(f1_score(labels, binary_preds, zero_division=0))
    result["accuracy"] = float(np.mean(binary_preds == labels))
    result["n_samples"] = len(labels)
    return result


def batch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def print_evaluation_report(metrics: dict, total_samples: int = 0) -> None:
    header = f"{'Category':<20} {'AUC':<10} {'F1':<10} {'Accuracy':<10} {'Samples':<8}"
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
        auc = m.get("auc_roc", float("nan"))
        f1 = m.get("f1", 0)
        acc = m.get("accuracy", 0)
        n = m.get("n_samples", "")
        label = key.replace("_", " ").title()
        auc_str = "N/A       " if (auc != auc) else f"{auc:<10.4f}"  # nan != nan is True
        print(f"{label:<20} {auc_str} {f1:<10.4f} {acc:<10.4f} {str(n):<8}")

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
        auc = m.get('auc_roc', float('nan'))
        auc_str = "N/A" if (auc != auc) else f"{auc:.4f}"
        lines.append(f"  AUC-ROC : {auc_str}")
        lines.append(f"  F1      : {m.get('f1', 0):.4f}")
        lines.append(f"  Accuracy: {m.get('accuracy', 0):.4f}")

    lines.append("\n" + "=" * 60)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")
