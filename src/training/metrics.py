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
    # Labels/preds may arrive shaped [N, 1]; flatten to 1-D so scalar conversion,
    # sklearn metrics, and the accuracy comparison all behave correctly.
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()
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
    return result


def batch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
