`# DeepSeek v4 — Coding Agent Instructions
## Image Forgery Detection System

> This document is the single source of truth for all development decisions.
> Follow every instruction here strictly. Do not deviate unless explicitly told to.

---

## 1. Project Overview

You are building a **real-time passive image forgery detection system** that detects three types of image manipulation:
- **Splicing** — a region pasted from a different image
- **Copy-Move** — a region duplicated within the same image
- **Retouching** — local pixel manipulation (blur, healing, smoothing, object removal)

The system must:
- Produce results in **< 50ms per image**
- Output both an **image-level verdict** (forged/authentic) and a **pixel-level heatmap** (where is the forgery)
- Be deployable as a **REST API** with a simple frontend

---

## 2. Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10 exactly |
| ML Framework | PyTorch | Latest stable |
| Model Export | ONNX + ONNX Runtime | Latest stable |
| Backbone | timm (EfficientNet-B0) | Latest stable |
| Image Processing | OpenCV + Pillow | Latest stable |
| Augmentation | Albumentations | Latest stable |
| API Backend | FastAPI | Latest stable |
| Data Handling | NumPy + Pandas | Latest stable |
| Metrics | scikit-learn | Latest stable |
| Progress | tqdm | Latest stable |
| Frontend | React (single HTML file) | — |

### Environment Setup
```bash
conda create -n forgery-detection python=3.10
conda activate forgery-detection
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-learn matplotlib pandas tqdm albumentations timm onnx onnxruntime fastapi uvicorn pillow
```

---

## 3. Project File Structure

```
forgery-detection/
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   │   ├── CASIA_v2/
│   │   ├── Coverage/
│   │   └── Korus/
│   ├── processed/                  # After preprocessing scripts run
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   └── splits/
│       └── split_metadata.csv      # Image paths + labels + forgery type
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── srm_filters.py          # SRM filter implementation (30 fixed filters)
│   │   ├── ela.py                  # ELA map generation
│   │   ├── dataset_builder.py      # Combine CASIA + Coverage + Korus
│   │   └── augmentation.py         # Albumentations pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dual_branch.py          # Main model architecture
│   │   ├── rgb_branch.py           # EfficientNet-B0 RGB branch
│   │   ├── noise_branch.py         # EfficientNet-B0 noise branch
│   │   ├── fusion.py               # Feature concatenation + 1x1 conv
│   │   ├── classification_head.py  # Binary classifier
│   │   └── segmentation_head.py    # Pixel-level mask decoder
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── losses.py               # BCE loss + Dice loss
│   │   ├── metrics.py              # AUC-ROC, F1, IoU per forgery type
│   │   └── scheduler.py            # ReduceLROnPlateau setup
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py            # Full inference pipeline
│   │   ├── postprocess.py          # Heatmap generation + forgery type ID
│   │   └── export.py               # ONNX export + INT8 quantization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py        # Overlay heatmap on image
│       ├── file_utils.py           # Safe file loading, validation
│       └── logger.py               # Logging setup
│
├── api/
│   ├── main.py                     # FastAPI app entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   └── detect.py               # /detect endpoint
│   └── schemas/
│       ├── __init__.py
│       └── response.py             # Pydantic response models
│
├── frontend/
│   └── index.html                  # Single-file React frontend
│
├── models/
│   ├── checkpoints/                # .pth files saved during training
│   └── exported/
│       ├── forgery_model.onnx      # ONNX exported model
│       └── forgery_model_int8.onnx # Quantized model
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_test.ipynb
│   ├── 03_model_test.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── configs/
│   └── config.yaml                 # All hyperparameters and paths
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Configuration File (configs/config.yaml)

All hyperparameters must live here — never hardcode values in scripts.

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  splits_file: "data/splits/split_metadata.csv"
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10
  image_size: 224

preprocessing:
  ela_quality: 75
  ela_amplify: 15
  srm_filter_count: 30
  min_file_size_kb: 12

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
  phase1_epochs: 10
  phase2_epochs: 30
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

paths:
  checkpoint_dir: "models/checkpoints"
  export_dir: "models/exported"
  log_dir: "logs"
```

---

## 5. Model Architecture Rules

### Dual-Branch CNN — Non-Negotiable Decisions
- **RGB Branch:** EfficientNet-B0 from `timm`, pretrained on ImageNet, `features_only=False`
- **Noise Branch:** EfficientNet-B0 from `timm`, `pretrained=False`, same architecture
- **Input size:** Always 224×224 — do not change this
- **Feature vector:** 1280-dim per branch, 2560-dim after concatenation
- **Classification head:** Linear(2560→512) → ReLU → Dropout(0.3) → Linear(512→1) → Sigmoid
- **Segmentation head:** 4× Upsample blocks (each: Upsample×2 → Conv3×3 → ReLU) → Conv1×1 → Sigmoid

### Model Code Pattern
```python
import timm
import torch.nn as nn

class RGBBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,          # removes classifier head
            global_pool='avg'       # global average pooling
        )

    def forward(self, x):
        return self.backbone(x)     # returns 1280-dim vector
```

---

## 6. Preprocessing Rules

### SRM Filters
- Use exactly 30 fixed high-pass filter kernels — do NOT make them learnable
- Filters must be registered as `nn.Parameter` with `requires_grad=False`
- Apply before feeding into noise branch
- Normalize output to same scale as RGB input

### ELA
- Always use quality=75 for re-compression
- Always amplify difference by factor of 15
- Clip values to [0, 255] after amplification
- Convert to 3-channel (repeat grayscale 3 times) to match model input

### Augmentation Pipeline (Albumentations)
```python
# Training augmentation — all of these MUST be included
A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),  # CRITICAL
    A.GaussNoise(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Validation/Test — NO augmentation except normalize + resize
A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

> `ImageCompression` augmentation is **mandatory** — this is the single most important augmentation for real-world generalization.

---

## 7. Training Rules

### Phase 1 — Branch Pretraining
- Train RGB branch alone for 5–10 epochs (classification loss only)
- Train Noise branch alone for 5–10 epochs (classification loss only)
- Use separate optimizers for each branch
- Do NOT train fusion or heads during Phase 1

### Phase 2 — Joint Training
- Unfreeze everything
- Train full network end-to-end
- Use combined loss:

```python
total_loss = bce_loss(pred_label, label) + 0.5 * dice_loss(pred_mask, mask)
```

### Optimizer Settings
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',         # monitor AUC-ROC (higher is better)
    patience=3,
    factor=0.5
)
```

### Checkpoint Saving
- Save checkpoint after every epoch where val AUC-ROC improves
- Always save: model weights, optimizer state, epoch number, val metrics
- Keep best 3 checkpoints only — delete older ones

---

## 8. Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

# Use both together
bce = nn.BCELoss()
dice = DiceLoss()

def combined_loss(pred_label, label, pred_mask, mask, dice_weight=0.5):
    return bce(pred_label, label) + dice_weight * dice(pred_mask, mask)
```

---

## 9. Evaluation Metrics

Compute ALL of these — never report just accuracy:

```python
from sklearn.metrics import roc_auc_score, f1_score

# Per forgery type breakdown is mandatory
def evaluate(preds, labels, masks, gt_masks, forgery_types):
    results = {}
    for ftype in ['splicing', 'copy_move', 'retouching']:
        idx = [i for i, t in enumerate(forgery_types) if t == ftype]
        results[ftype] = {
            'auc_roc': roc_auc_score(labels[idx], preds[idx]),
            'f1': f1_score(labels[idx], preds[idx] > 0.5),
            'iou': compute_iou(masks[idx], gt_masks[idx])
        }
    results['overall'] = {
        'auc_roc': roc_auc_score(labels, preds),
        'f1': f1_score(labels, preds > 0.5),
        'iou': compute_iou(masks, gt_masks)
    }
    return results
```

---

## 10. Inference Pipeline

### Full Pipeline (predictor.py)
```
receive image bytes
    → validate (format, size, not corrupt)
    → resize to 224×224
    → generate 3 streams (RGB, SRM noise, ELA)
    → run ONNX model
    → postprocess (confidence, mask, forgery type)
    → generate heatmap overlay
    → return JSON response
```

### Response Schema
```python
from pydantic import BaseModel

class ForgeryResponse(BaseModel):
    verdict: str                    # "FORGED" or "AUTHENTIC"
    confidence: float               # 0.0 to 100.0
    forgery_type: str               # "Splicing", "Copy-Move", "Retouching", "Unknown"
    forged_area_percentage: float   # 0.0 to 100.0
    heatmap_base64: str             # base64 encoded overlay image
    processing_time_ms: float
```

---

## 11. API Rules

### FastAPI Endpoint
```python
@router.post("/detect", response_model=ForgeryResponse)
async def detect_forgery(file: UploadFile = File(...)):
    # validate → preprocess → infer → postprocess → return
```

### API Rules
- Accept: JPG, JPEG, PNG only
- Max file size: 10MB
- Reject files < 12KB (too small, likely corrupt or thumbnail)
- Always return processing_time_ms
- Never return raw model logits to client — only processed human-readable results
- Use async throughout for non-blocking inference
- Add CORS middleware for frontend access

---

## 12. ONNX Export & Quantization

```python
import torch

def export_to_onnx(model, save_path):
    model.eval()
    dummy_rgb   = torch.randn(1, 3, 224, 224)
    dummy_noise = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_noise),
        save_path,
        opset_version=13,
        input_names=['rgb', 'noise'],
        output_names=['label', 'mask'],
        dynamic_axes={
            'rgb':   {0: 'batch_size'},
            'noise': {0: 'batch_size'},
            'label': {0: 'batch_size'},
            'mask':  {0: 'batch_size'}
        }
    )

# INT8 Quantization after export
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "forgery_model.onnx",
    "forgery_model_int8.onnx",
    weight_type=QuantType.QInt8
)
```

---

## 13. ✅ DOS — Always Do These

- **Always read config.yaml** — never hardcode any number, path, or hyperparameter
- **Always validate input images** before preprocessing — check format, size, not corrupt
- **Always log** epoch number, loss, AUC-ROC, F1, IoU at every epoch
- **Always save checkpoints** with full metadata (epoch, metrics, config hash)
- **Always use `model.eval()` and `torch.no_grad()`** during inference
- **Always normalize** images before feeding to model (ImageNet mean/std)
- **Always apply ImageCompression augmentation** during training
- **Always compute per-forgery-type metrics** separately — not just overall
- **Always use `timm`** for EfficientNet — not torchvision
- **Always use `AdamW`** — not plain Adam
- **Always use `async`** in FastAPI route handlers
- **Always clip ELA values** to [0, 255] after amplification
- **Always keep SRM filter weights frozen** (`requires_grad=False`)
- **Always export to ONNX** before deployment — never serve raw PyTorch in production
- **Always add early stopping** with patience=7 based on val AUC-ROC
- **Always stratify train/val/test splits** by forgery type

---

## 14. ❌ DON'TS — Never Do These

- **Never use `torch.save(model)` directly** — always save `model.state_dict()`
- **Never train on test data** — strict separation, test set touched only once at the very end
- **Never use accuracy alone** as the evaluation metric — always AUC-ROC + F1 + IoU
- **Never hardcode file paths** — always use config.yaml or `pathlib.Path`
- **Never use `ResNet` or `VGG`** as backbone — EfficientNet-B0 only
- **Never make SRM filters learnable** — they must be fixed high-pass filters
- **Never skip JPEG compression augmentation** — this is the most critical augmentation
- **Never use input size other than 224×224** — model architecture is fixed to this
- **Never use `Adam`** — always `AdamW` for better weight regularization
- **Never serve raw PyTorch model in production** — always ONNX Runtime
- **Never use `pickle`** for saving anything model-related — use `.pth` with state_dict
- **Never process the same image stream twice** — generate all 3 streams in one pass
- **Never use `print()`** for logging — always use Python `logging` module
- **Never ignore the mask loss** — segmentation head must always be trained jointly
- **Never accept files larger than 10MB** in the API
- **Never return internal error tracebacks** to API clients — catch and return clean messages
- **Never skip input validation** — always check format and size before preprocessing
- **Never use `random.seed()`** only — always set seeds for torch, numpy, and random together

---

## 15. Seed Everything (Reproducibility)

Always call this at the start of any training script:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 16. Logging Setup

```python
import logging

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
```

Log at minimum:
- Every epoch: loss, val AUC-ROC, val F1, val IoU, learning rate
- Every checkpoint save
- Every error with full context
- Inference time per request in production

---

## 17. Dataset Rules

### Datasets to Use
| Dataset | Forgery Type | Priority |
|---|---|---|
| CASIA v2 | Splicing + Copy-Move | Primary |
| Coverage | Copy-Move | Primary |
| Korus / FODB | Retouching + Splicing | Primary |
| Columbia | Splicing | Secondary |

### Dataset Building Rules
- Skip any file smaller than 12KB
- Skip any corrupt file (wrap `cv2.imread` in try/except)
- Balance classes — no forgery type should dominate more than 2:1 ratio
- Store split metadata in `splits/split_metadata.csv` with columns:
  `image_path, mask_path, label, forgery_type, dataset_source, split`
- Masks must be binary (0 = authentic, 255 = forged) — normalize to [0,1] before loss

---

## 18. Optimization Strategy for Real-Time

### Target: < 50ms total per image

| Step | Expected Time | Optimization |
|---|---|---|
| Input validation | ~1ms | Fast format check only |
| Preprocessing (3 streams) | ~10–15ms | Run streams in parallel |
| ONNX inference | ~25–35ms | INT8 quantized model |
| Postprocessing + heatmap | ~5ms | NumPy operations only |
| JSON serialization | ~1ms | Pydantic |
| **Total** | **~43–57ms** | |

### Speed Rules
- Use **ONNX Runtime** not raw PyTorch for production inference
- Use **INT8 quantization** — reduces model size 4× and speeds up 2–4×
- Use **batch inference** when processing multiple images
- Pre-load model at API startup — never load model per request
- Pre-compute SRM filter kernels at startup — never recompute
- Use **NumPy** not PIL for postprocessing operations
- If using GPU: pin memory in DataLoader (`pin_memory=True`)

---

## 19. Error Handling Pattern

```python
# Every preprocessing step must be wrapped
def safe_preprocess(image_bytes: bytes) -> dict:
    try:
        img = decode_image(image_bytes)
        if img is None:
            raise ValueError("Image decoding failed — file may be corrupt")
        if img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError("Image too small for analysis")
        return {
            'rgb':   preprocess_rgb(img),
            'noise': preprocess_srm(img),
            'ela':   preprocess_ela(img)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal preprocessing error")
```

---

## 20. Git & Code Quality Rules

- One file per class — no god files
- Every function must have a docstring with args and return types
- Every module must have `__init__.py`
- Use `pathlib.Path` everywhere — never string concatenation for paths
- Maximum function length: 50 lines — refactor if longer
- Use type hints on every function signature
- `.gitignore` must exclude: `data/raw/`, `models/`, `logs/`, `.env`, `__pycache__`
- Never commit model weights or dataset files to git

---

## 21. Expected Performance Targets

| Metric | Minimum Acceptable | Target |
|---|---|---|
| Overall AUC-ROC | 0.85 | 0.90+ |
| Splicing AUC-ROC | 0.88 | 0.93+ |
| Copy-Move AUC-ROC | 0.85 | 0.91+ |
| Retouching AUC-ROC | 0.75 | 0.83+ |
| Pixel IoU | 0.60 | 0.70+ |
| Inference time | < 50ms | < 43ms |
| Model size (quantized) | < 15MB | ~10MB |

---

*Agent instruction document — Image Forgery Detection System. Do not modify architecture, stack, or training decisions without explicit user approval.*
