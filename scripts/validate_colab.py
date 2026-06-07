"""
DeepSight — Pre-Colab Validation Script
Run this to verify all components work correctly before uploading to Colab.

Usage:
    python scripts/validate_colab.py
"""

import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath("."))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def ok(msg):
    print(f"  [OK]  {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


errors = []


# ── 1. Core imports ────────────────────────────────────────────────────────────
section("1. Core imports")
try:
    import yaml
    import torch
    import timm
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ok(f"torch          {torch.__version__}")
    ok(f"timm           {timm.__version__}")
    ok(f"albumentations {A.__version__}")
    ok(f"cv2            {cv2.__version__}")
except Exception as e:
    fail(str(e)); errors.append(e)


# ── 2. Augmentation pipeline ───────────────────────────────────────────────────
section("2. Augmentation pipeline (albumentations 2.x API)")
try:
    import cv2 as _cv2
    geo = A.Compose([
        A.PadIfNeeded(min_height=224, min_width=224,
                      border_mode=_cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    photo = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.ImageCompression(quality_range=(50, 95), p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])
    norm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img_geo   = geo(image=dummy_img)["image"]
    img_ph    = photo(image=img_geo)["image"]
    img_norm  = norm(image=img_ph)["image"]
    assert img_norm.shape == (3, 224, 224), f"Unexpected shape: {img_norm.shape}"
    ok(f"Geometric + photometric + normalize OK -> {img_norm.shape}")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 3. SRM filters ────────────────────────────────────────────────────────────
section("3. SRM noise extraction")
try:
    from src.preprocessing.srm_filters import SRMFilterLayer, extract_srm_noise_batch
    srm = SRMFilterLayer()
    batch_img = dummy_img[np.newaxis, ...].astype(np.float32)
    noise_out = extract_srm_noise_batch(batch_img, srm)
    assert noise_out.shape == (1, 300, 400, 30), f"Unexpected shape: {noise_out.shape}"
    ok(f"SRM output shape: {noise_out.shape}  (N, H, W, 30 channels)")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 4. ELA ────────────────────────────────────────────────────────────────────
section("4. ELA (Error Level Analysis)")
try:
    from src.preprocessing.ela import compute_ela, ela_to_3channel
    img_aug = geo(image=dummy_img)["image"]
    ela     = compute_ela(img_aug, amplify=20.0)
    ela3    = ela_to_3channel(ela)
    assert ela3.shape == (224, 224, 3), f"Unexpected shape: {ela3.shape}"
    ok(f"ELA output shape: {ela3.shape}")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 5. noise_input channel count ──────────────────────────────────────────────
section("5. noise_input tensor (SRM 30ch + ELA 3ch = 33ch)")
try:
    import torch as _torch
    import numpy as _np
    from src.preprocessing.srm_filters import extract_srm_noise_batch, SRMFilterLayer
    from src.preprocessing.ela import compute_ela, ela_to_3channel

    img_aug = geo(image=dummy_img)["image"]  # uint8 HWC 224x224

    ela_arr   = compute_ela(img_aug, amplify=20.0)
    ela3      = ela_to_3channel(ela_arr)
    ela_t     = _torch.from_numpy(ela3.transpose(2, 0, 1).astype(_np.float32) / 255.0)

    srm_layer = SRMFilterLayer()
    noise_np  = extract_srm_noise_batch(img_aug[_np.newaxis, ...], srm_layer)
    noise_t   = _torch.from_numpy(noise_np[0].transpose(2, 0, 1)).float()

    noise_input = _torch.cat([noise_t, ela_t], dim=0)
    assert noise_input.shape == (33, 224, 224), f"Unexpected: {noise_input.shape}"
    ok(f"noise_input shape: {noise_input.shape}  (expected [33, 224, 224])")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 6. Model forward pass ─────────────────────────────────────────────────────
section("6. Model forward pass (no pretrained download)")
try:
    from pathlib import Path
    import yaml as _yaml
    from src.models.dual_branch import DualBranchModel

    config = _yaml.safe_load(Path("configs/config.yaml").read_text())
    model = DualBranchModel(
        backbone=config["model"]["backbone"],
        pretrained_rgb=False,
        pretrained_noise=False,
        feature_dim=config["model"]["feature_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        freeze_bn=False,
    )
    rgb_t   = torch.randn(2, 3, 224, 224)
    noise_t = torch.randn(2, 33, 224, 224)
    out = model(rgb_t, noise_t)
    assert out.shape == (2, 1), f"Unexpected output: {out.shape}"
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"Output shape  : {out.shape}  (expected [2, 1])")
    ok(f"Total params  : {total_p:,}")
    ok(f"Trainable     : {trainable_p:,}")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 7. Loss function ──────────────────────────────────────────────────────────
section("7. CombinedLoss")
try:
    from src.training.losses import CombinedLoss
    loss_fn = CombinedLoss(pos_weight=1.0, label_smoothing=0.05)
    logits  = torch.tensor([[0.8], [-0.3]])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0, "Loss should be positive"
    ok(f"Loss value: {loss.item():.4f}")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 8. Scheduler ──────────────────────────────────────────────────────────────
section("8. Scheduler (cosine with warmup)")
try:
    from pathlib import Path
    import yaml as _yaml
    from src.training.scheduler import build_scheduler
    config = _yaml.safe_load(Path("configs/config.yaml").read_text())
    model_params = [torch.nn.Parameter(torch.randn(10))]
    opt = torch.optim.AdamW(model_params, lr=1e-4)
    sched = build_scheduler(opt, config)
    assert hasattr(sched, "_is_plateau"), "Scheduler missing _is_plateau attr"
    ok(f"Scheduler type: cosine, _is_plateau={sched._is_plateau}")
    sched.step()
    ok("scheduler.step() OK")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 9. logs/ directory creation ───────────────────────────────────────────────
section("9. logs/ directory auto-creation (Colab fix)")
try:
    import shutil
    log_path = "logs/training_log.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    assert os.path.isdir("logs"), "logs/ dir was not created"
    ok("os.makedirs('logs', exist_ok=True) works correctly")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── 10. Metrics with NaN guard ────────────────────────────────────────────────
section("10. Metrics (single-class NaN guard)")
try:
    from src.training.metrics import compute_metrics
    import numpy as _np
    all_zeros = _np.zeros(50)
    preds     = _np.random.rand(50)
    metrics   = compute_metrics(all_zeros, preds)
    auc = metrics["overall"]["auc_roc"]
    ok(f"Single-class AUC = {auc} (NaN is expected & handled)")
    mixed_labels = _np.array([0]*25 + [1]*25)
    metrics2 = compute_metrics(mixed_labels, preds)
    auc2 = metrics2["overall"]["auc_roc"]
    assert not _np.isnan(auc2), "AUC should not be NaN for mixed labels"
    ok(f"Mixed-class AUC  = {auc2:.4f}")
except Exception as e:
    traceback.print_exc(); errors.append(e)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"  RESULT: {len(errors)} check(s) FAILED")
    print("  Fix the above issues before uploading to Colab.")
    sys.exit(1)
else:
    print("  RESULT: ALL CHECKS PASSED ✓")
    print("  The project is ready to train on Google Colab T4.")
print("="*60)
