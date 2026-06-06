# DeepSight — Project Context for Claude

> **Purpose:** Single-file context dump describing how this image-forgery detection project works, what each module does, and which hyperparameters / data choices most strongly affect accuracy. Use this when asking Claude to suggest improvements, debug, or extend the system.

---

## 1. What the project does

**DeepSight** is a passive (blind) image-forgery detection system. Given a single image with no reference, it must:

1. Decide **authentic vs forged** (binary classification).
2. Localize the forged region as a **224×224 pixel-level mask**.
3. Identify the **forgery type** (Splicing / Copy-Move / Retouching).
4. Run in **< 50 ms / image** on ONNX Runtime + INT8.

Three forgery types:
- **Splicing** — region pasted from a *different* image.
- **Copy-Move** — region duplicated *within* the same image.
- **Retouching** — local pixel edits (blur, healing, smoothing, object removal).

---

## 2. Repository layout

```
E:\SEM Porj\deepsight\
├── train.py                 # Entry point (Phase 1 → Phase 2)
├── eval_model.py            # Loads best_model.pth, samples data/raw, prints metrics
├── configs/config.yaml      # ALL hyperparameters (single source of truth)
├── data/
│   ├── raw/{CASIA_v2, Coverage, Korus}/
│   ├── splits/split_metadata.csv   # image_path, mask_path, label, forgery_type, dataset_source, split
│   └── processed/{train,val,test}/{images,masks}/   (currently empty — pipeline reads raw directly)
├── models/checkpoints/best_model.pth   (~135 MB, current best)
├── src/
│   ├── preprocessing/   # image_input, resize_normalise, srm_filters, ela, augmentation, dataset_builder
│   ├── models/          # dual_branch, rgb_branch, noise_branch, fusion, classification_head, segmentation_head
│   ├── training/        # trainer, losses, metrics, scheduler
│   ├── inference/       # predictor, postprocess, export (ONNX)
│   └── utils/           # logger, file_utils, visualization
├── api/                 # FastAPI app
├── frontend/index.html  # Single-file React UI
├── logs/evaluation_report.json
├── logs/training_log.csv   (header only — no per-epoch rows were ever appended)
├── deepseek_agent_instructions.md   # The original spec this codebase was built from
└── STEP_BREAKDOWN.md    # Theory notes for each pipeline step
```

---

## 3. End-to-end data flow

```
┌──────────────┐     ┌──────────────────────────────────────────┐     ┌──────────────┐
│  Raw image   │ ──▶ │  Geometric aug (PadIfNeeded, 224×224     │ ──▶ │   RGB tensor  │ ──┐
│  (BGR uint8) │     │  RandomCrop, HFlip, VFlip, Rot90)        │     │  3 × 224×224  │   │
└──────────────┘     └──────────────────────────────────────────┘     └──────────────┘   │
                                                                                          │
                       ┌──────────────────────────────────────────┐                        │
                       │  Photometric aug (ColorJitter,           │                        ▼
                       │  ImageCompression 70-95, GaussNoise)     │                  ┌──────────────┐
                       │  Normalize ImageNet mean/std             │                  │  RGB Branch  │
                       └──────────────────────────────────────────┘                  │ EfficientNet │
                                                                                      │   -B0 (PT)   │
                                                                                      └──────┬───────┘
                                                                                             │ 1280-d
                                                                                             ▼
                ┌──────────────┐     ┌──────────────────────────────────────────┐     ┌──────────────┐
                │  Same 224×224│ ──▶ │  SRM 30 fixed high-pass filters (frozen)  │ ──▶ │ Noise Branch │
                │  augmented   │     │  → 30-channel noise map                   │     │ EfficientNet │
                │  RGB image   │     │  + ELA @ q=75, amp=15 (3 ch)             │     │   -B0 (from  │
                │              │     │  → concat → 33-channel input              │     │  scratch)    │
                └──────────────┘     └──────────────────────────────────────────┘     └──────┬───────┘
                                                                                             │ 1280-d
                                                                                             ▼
                                                                                  ┌──────────────────────┐
                                                                                  │  FeatureFusion       │
                                                                                  │  cat(1280+1280)=2560 │
                                                                                  │  Conv1x1 + BN + ReLU │
                                                                                  │  → 512-d             │
                                                                                  └──────────┬───────────┘
                                                                                             │
                                                              ┌──────────────────────────────┴──────────┐
                                                              ▼                                         ▼
                                                ┌────────────────────────┐               ┌────────────────────────┐
                                                │ ClassificationHead     │               │ SegmentationHead        │
                                                │ 512 → Dropout(0.3) → 1 │               │ U-Net decoder with RGB  │
                                                │ → Sigmoid (verdict)    │               │ skip connections        │
                                                │                        │               │ → 224×224 sigmoid mask │
                                                └────────────────────────┘               └────────────────────────┘
```

Forensic features (SRM, ELA) are computed **on the geometrically-augmented 224×224 image** so the noise branch input is spatially aligned with the RGB branch input and the mask. Photometric augmentation (compression, jitter) is applied **only to the RGB input** so SRM/ELA see the same noise statistics every epoch. (`src/preprocessing/dataset_builder.py:206-258`)

---

## 4. What each module does

### Preprocessing

- **`image_input.py`** — Load image bytes, format detection, HEIC handling via `pillow-heif`, EXIF orientation (`ImageOps.exif_transpose`), min size 32×32, min file size 12 KB. RGB conversion.
- **`resize_normalise.py`** — `cv2.resize` to 224×224 (LANCZOS4 for downscaling), ImageNet mean/std normalization on RGB branch only. Preserves original dims for mask upscaling at inference.
- **`srm_filters.py`** — 30 hand-crafted 5×5 high-pass filters (3×3 padded + Sobel + Laplacian + noise + quad + cross + random) loaded as a frozen `nn.Conv2d(3,30,5,padding=2,bias=False)`. Each output channel has the same filter applied to all 3 RGB channels. `requires_grad=False` always.
- **`ela.py`** — Re-save PIL image to JPEG in-memory at quality 75, compare to original, `abs(diff) * 15`, clip 0–255, repeat to 3 channels.
- **`augmentation.py`** — Split into `GEOMETRIC_TRAIN/VAL` (applied to image+mask together) and `PHOTOMETRIC_TRAIN` (RGB only). Includes `ImageCompression(quality_range=(70,95), p=0.5)` and `GaussNoise(p=0.3)`.
- **`dataset_builder.py`** — Scans `data/raw/{CASIA_v2,Coverage,Korus}` and builds `split_metadata.csv` with stratified 80/10/10 splits per forgery type. Each `__getitem__` loads → geometric aug → compute SRM+ELA → photometric aug → normalize → return `{rgb, noise, label, mask, forgery_type}`.

### Models

- **`rgb_branch.py`** — `timm.create_model('efficientnet_b0', pretrained=True, features_only=True)` + a `Conv2d(320,1280,1)` projection head. Returns 1280-channel feature map at 7×7 and saves 5 skip levels for the decoder.
- **`noise_branch.py`** — Same backbone, **pretrained=False**, `in_chans=33`. The first conv layer is reinitialised because SRM+ELA are 33 channels, not 3.
- **`fusion.py`** — `FeatureFusion` concatenates both 1280-channel maps → `Conv2d(2560,512,1)` → BN → ReLU → Dropout2d. (`AdaptiveFusion` with gates is defined but **not used** in `dual_branch.py`.)
- **`classification_head.py`** — `Linear(512,1)`. Sigmoid is applied in the loss / predictor, not here.
- **`segmentation_head.py`** — U-Net-style decoder: 5 `ConvTranspose2d` upsamples (512→256→128→64→32→16) with 4 skip connections from the RGB branch (`skip_channels = [112, 40, 24, 16]`, derived from EfficientNet-B0 feature sizes). Final `Conv2d(16,1,1)` + sigmoid → 224×224 mask.
- **`dual_branch.py`** — Glues everything together. `forward(rgb, noise)` returns `(label, mask)`. `predict()` adds the sigmoid.

### Training

- **`losses.py`** — `CombinedLoss = BCEWithLogitsLoss(label) + 0.5 * DiceLoss(mask)`. Dice smooth = 1.0.
- **`metrics.py`** — Per-forgery-type + overall: AUC-ROC (sklearn), F1 at threshold 0.5, accuracy, pixel IoU.
- **`scheduler.py`** — `ReduceLROnPlateau(mode='max', factor=0.5, patience=3, min_lr=1e-7)`.
- **`trainer.py`** — Phase 1 freezes RGB branch, trains noise+fusion+heads for `phase1_epochs`. Phase 2 unfreezes everything, trains for `phase2_epochs`. Saves `best_model.pth` whenever `val_auc` improves. Keeps last 3 checkpoints. Early stops after `early_stopping_patience=7` non-improving epochs.
- **`train.py`** — Orchestrates the two phases.

### Inference

- **`predictor.py`** — Loads image, runs the same preprocessing (resize → SRM → ELA → normalise), runs the model, applies threshold 0.5, picks forgery type from connected-component count on the predicted mask (`≤1` = Retouching, `≤3` = Splicing, `>3` = Copy-Move), returns `{verdict, confidence, forgery_type, forged_area_percentage, heatmap, processing_time_ms}`.
- **`postprocess.py`** — Resizes 224×224 mask back to original image size, builds JET heatmap, base64-encodes overlay, computes forged-area percentage.
- **`export.py`** — ONNX export + INT8 dynamic quantisation (size 134 MB → expected ~10 MB; not yet run on the current checkpoint).

---

## 5. Current configuration (from `configs/config.yaml`)

```yaml
data:
  raw_dir: "data/raw"
  image_size: 224
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10

preprocessing:
  ela_quality: 75          # JPEG recompression quality for ELA
  ela_amplify: 15          # difference amplification factor
  srm_filter_count: 30     # fixed
  min_file_size_kb: 12
  min_dimension: 32
  apply_orientation: true
  image_interpolation: "linear"
  normalise_scale: 255.0
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std:  [0.229, 0.224, 0.225]

model:
  backbone: "efficientnet_b0"
  pretrained_rgb: true
  pretrained_noise: false
  feature_dim: 1280
  fused_dim: 2560
  hidden_dim: 512
  dropout: 0.3
  num_classes: 1

training:
  phase1_epochs: 10         # noise branch + heads only (RGB frozen)
  phase2_epochs: 30         # full network
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  dice_loss_weight: 0.5
  freeze_bn: true
  early_stopping_patience: 7

inference:
  confidence_threshold: 0.5
  mask_threshold: 0.5
  target_ms: 50
```

---

## 6. Dataset actually loaded

`split_metadata.csv` contains **9,342 images** (stratified 80/10/10 by forgery type).

| Source | Authentic | Splicing | Copy-Move | Retouching | Total |
|---|---|---|---|---|---|
| CASIA_v2 | 7,260 | 2,064 | — | — | 9,324 |
| Coverage | — | — | **0** (mask folder not present) | — | 0 |
| Korus | 14 | — | — | 4 | 18 |
| **Total** | **7,274 (78 %)** | **2,064 (22 %)** | 0 | 4 | **9,342** |

Split sizes: **train 7,473 / val 933 / test 936**. **Class imbalance ≈ 3.5 : 1 (authentic : forged).**

⚠️ The Coverage dataset is referenced in code but its `image/` folder is empty, so it contributes **zero** samples. The Korus dataset is essentially absent (only 18 files) — retouching is a paper-only class right now.

---

## 7. Current training results

`logs/evaluation_report.json` (from `eval_model.py` running on a 900-image sample: 200 CASIA-Au, 200 CASIA-Tp, 100 Coverage, 50 Korus per camera):

| Group | n | AUC-ROC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Splicing | 194 | 1.00 | 0.49 | 1.00 | 0.32 | 0.32 |
| Copy-Move | 55 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Retouching | 200 | 1.00 | 0.01 | 1.00 | 0.005 | 0.005 |
| Authentic | 451 | 0.00 | 0.00 | 0.00 | 0.00 | **0.98** |
| **Overall** | **900** | **0.62** | **0.25** | **0.88** | **0.14** | **0.56** |

`logs/training_log.csv` only contains the header row — **per-epoch metrics were never persisted** (the trainer logs to `logging`, not to this CSV).

⚠️ The user states current accuracy is **77 %**. This matches the **naive "always predict authentic" baseline** (7,274 / 9,342 = 77.9 %), not the model's measured 56 %. The model has clearly collapsed to predicting almost everything as authentic — recall on the forged class is 14 % overall (0 % for Copy-Move, 0.5 % for Retouching, 32 % for Splicing).

---

## 8. Hyperparameters / design choices that most affect accuracy

Ranked by expected impact on the 77 % → target (≥90 % AUC):

### High impact (try first)

1. **Class imbalance fix.** Authentic : forged ≈ 3.5 : 1, with retouching effectively absent.
   - Add `pos_weight` to `BCEWithLogitsLoss` (in `src/training/losses.py:24`). Suggested: `pos_weight = n_authentic / n_forged ≈ 3.5`, tune 2 – 6.
   - Or oversample forgery in `ForgeryDataset.__getitem__` (WeightedRandomSampler).
   - Or undersample authentic (CASIA v2 has 7,260 — drop ~half).
2. **Increase epoch count and lower early-stopping patience bias.** `phase2_epochs=30` with `patience=7` may stop before the model escapes the "always authentic" local minimum. Try `phase2_epochs=50-60`, `patience=10-12`.
3. **Learning-rate schedule.** Current `ReduceLROnPlateau(factor=0.5, patience=3)` may decay too fast when AUC plateaus. Try a warmup + cosine schedule, or `CosineAnnealingLR(T_max=phase2_epochs)`.
4. **Batch size.** `batch_size=32` is fine for an EfficientNet-B0 dual branch. If GPU has memory, try 64 for more stable gradients.
5. **Phase 1 length and what it warms up.** The current Phase 1 freezes RGB and trains noise+fusion+heads. If noise branch starts from scratch with no warm-up, 10 epochs may be insufficient — try 15-20, or do a single-branch warm-up first (just the noise branch with a 2-layer classifier) before adding fusion.
6. **Segmentation head skip-connection mismatch.** The decoder expects 4 skips at channels `[112, 40, 24, 16]` from the **RGB** branch's `features_only` output. Verify these dimensions match EfficientNet-B0's actual feature sizes; mismatches will silently zero-pad (see `DecoderBlock` `if skip is None: zeros(...)`) and waste capacity.

### Medium impact

7. **`freeze_bn: true` during Phase 2.** All BatchNorm running stats are frozen. This is fine for fine-tuning but will hurt the noise branch (which is training from scratch on a 33-channel distribution very different from ImageNet). Consider unfreezing BN for the noise branch only, or set `freeze_bn: false` for Phase 2.
8. **`dropout=0.3` on the classification head.** Could be too high (under-fits small data) or too low (over-fits the dominant authentic class). Try 0.1 – 0.5.
9. **`hidden_dim=512` after fusion.** The 1×1 conv collapses 2560 → 512, losing information. Try 768 or 1024 if VRAM allows.
10. **Dice loss weight.** Currently 0.5, but Dice operates on the (sigmoid-output) mask without logits — using `BCEWithLogitsLoss` on the mask instead of `DiceLoss` on sigmoid would be more numerically stable. Consider adding a mask-BCE term: `bce(label) + bce_mask(mask) + 0.5*dice(mask)`.
11. **Image compression augmentation range.** `(70, 95)` at `p=0.5` simulates social-media recompression. Real WhatsApp/Instagram quality can be 50-70 — consider widening to `(50, 95)`.
12. **`min_file_size_kb=12`** filters out very small images. If the test set contains low-quality social-media images below this threshold, they will fail validation. Consider lowering to 6-8 KB.

### Low impact / fine-tuning

13. **ELA quality 75 / amplify 15.** These are the literature defaults; can be tuned per dataset. Amplify=20 helps when ELA signal is weak (PNG-saved or re-compressed images).
14. **SRM filter count 30.** 30 channels is the steganalysis-standard value. Adding more (e.g., 40) gives marginal gains at higher compute cost.
15. **Mean / std normalisation on RGB branch only.** Already correct.
16. **`padifneeded` then `RandomCrop`.** Pads first if image is < 224×224, then randomly crops. For non-square images, this can lose information at the edges — consider `LongestMaxSize(224)` + `PadIfNeeded` (letterbox) for cleaner forensic feature extraction.
17. **Forgery-type classification by connected components** (`predictor.py:107-122`) is a heuristic. With a 0/0/0.5 % F1, this is essentially guessing.

---

## 9. Known issues / things to flag to Claude

- **The model is predicting almost everything as authentic.** Recall on forged is 14 % overall. This is the dominant problem.
- **No per-epoch metrics persisted** — `logs/training_log.csv` is empty. The trainer's `logger.info(...)` calls go to stdout/file, not to CSV. To diagnose, retrain with the CSV-writer or inspect the console output.
- **`val_metrics` after Phase 1 in `train.py:140` are not actually computed** — `trainer.fit(..., phase1=True)` does run validation each epoch, but the AUC after Phase 1 is not printed. The model loaded by `eval_model.py` is whatever was best during Phase 2.
- **`AdaptiveFusion`** is defined in `fusion.py:75` but never instantiated — only `FeatureFusion` is used.
- **Coverage dataset is missing** — code path runs but finds no files.
- **Retouching class is effectively absent** — only 4 Korus images total.
- **ONNX export + INT8 quantisation have not been run** on the latest checkpoint.
- **The `.venv` is Python 3.14 (cpython-314 caches visible) but the spec said 3.10** — this works but `albumentations` and other libs may not have wheels for 3.14 on Windows.
- **The test set in `data/processed/test/` is empty** — preprocessing writes to this folder but `dataset_builder.py` actually reads from `data/raw/...` paths stored in the CSV. The processed folder is currently dead.

---

## 10. Quick commands

```bash
# Train from scratch (Phase 1 + Phase 2)
python train.py --config configs/config.yaml

# Phase 1 only / skip Phase 1
python train.py --phase1-only
python train.py --skip-phase1

# Resume from checkpoint
python train.py --resume models/checkpoints/best_model.pth

# Evaluate on a 900-image sample from data/raw
python eval_model.py

# Run API
uvicorn api.main:app --reload

# Export to ONNX (not yet wired into a CLI)
python -c "from src.inference.export import export; export('models/checkpoints/best_model.pth', 'models/exported/')"
```

---

## 11. When asking Claude for help, mention

- The model currently **collapses to predicting authentic** (overall accuracy ~56 %, recall on forged 14 %, retouching recall 0.5 %, copy-move recall 0 %).
- Target performance: **AUC-ROC ≥ 0.90, pixel IoU ≥ 0.70** (per `deepseek_agent_instructions.md §21`).
- Hardware: Windows, CUDA 12.6, current `best_model.pth` is 135 MB (FP32, not yet quantised).
- Constraint: must remain under **50 ms / image** at inference.
- Pretrained weights come from ImageNet for the RGB branch only; noise branch is from scratch.
- If you suggest architecture changes, remember the noise branch needs 33 input channels (30 SRM + 3 ELA).
