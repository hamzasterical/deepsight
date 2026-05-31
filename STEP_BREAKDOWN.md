# Image Forgery Detection System — Step Breakdown

## Step 1 — Image Input

### Passive detection fundamentals
- **Why passive detection works without a reference image** — detects statistical inconsistencies (noise, compression artifacts) left by editing tools, no original photo needed
- **How images lose metadata (EXIF stripping, recompression)** when shared on WhatsApp, Instagram, Twitter — metadata is stripped, images are recompressed, but noise and compression artifacts persist; your system detects those, not EXIF data
- **What "blind" forgery detection means** — no watermark, no digital signature, no reference image; purely analyzes the image itself for tampering traces

### Image formats you'll encounter
- **JPEG compression internals** — DCT blocks (8×8), quality factor controls quantization table aggressiveness, chroma subsampling (4:2:0 typical)
- **PNG lossless vs JPEG lossy** — PNG has no compression artifacts; ELA relies on JPEG re-compression artifacts, so ELA is unreliable on PNG
- **WebP, HEIC** — convert to PNG using Pillow before entering the pipeline; HEIC needs `pillow-heif` plugin

---

## Step 2 — Resize & Normalise

### Why 224×224
- **EfficientNet-B0's expected input resolution** — the architecture's stem and feature pyramid are designed for 224×224
- **Speed vs accuracy tradeoff** — 300+ px hurts inference time significantly; the model has fixed feature map sizes that assume 224×224 input
- **Aspect ratio distortion** — squeezing a wide image to 224×224 shifts where tampered regions appear in pixel space; masks must be resized with matching interpolation

### ImageNet mean/std normalisation
- **Values**: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **RGB branch only** — the noise branch receives SRM/ELA maps with different statistics, so don't apply ImageNet normalisation there
- **Applied via** `transforms.Normalize()` in PyTorch — normalizes each channel independently

### Pillow & OpenCV for resizing
- `cv2.resize()` vs `PIL.Image.resize()` — use `cv2.INTER_LINEAR` for speed or `cv2.INTER_LANCZOS4` for quality when downscaling; `PIL.Image.LANCZOS` is best for downscaling masks
- **Preserve original dimensions** — store original width/height before resizing; used later to upscale the 224×224 mask back to full resolution

---

## Step 3 — SRM Noise Map Extraction

### What SRM is
- **Steganalysis Rich Model** — originally for steganalysis (detecting hidden messages), adapted for forgery detection
- **Residual noise** — subtract a predicted clean pixel value from the actual pixel; the residual reveals manipulation artifacts

### The 30 fixed high-pass filters
- **What high-pass filters do** — suppress low-frequency content (the actual image), amplify high-frequency content (noise, edges, artifacts)
- **Why fixed (not learned)** — hand-crafted from known camera sensor noise models; learning them wastes parameters since they are well-understood
- **Implementation** — `nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)` with `requires_grad=False` weights
- **Filter kernels** — 3×3 and 5×5 kernels targeting different noise frequencies; defined as numpy arrays loaded into the Conv2d weight tensor

### What SRM reveals for each forgery type
- **Splicing** — noise inconsistency between pasted region and host background
- **Copy-move** — noise pattern repetition across duplicated regions
- **Retouching** — smoothed noise where healing/blur tools have been applied

### Implementation
- Apply SRM as preprocessing via a frozen `nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)` layer
- Output is a 30-channel noise map → feeds into noise branch of EfficientNet-B0
- **Zero inference cost** — filters are deterministic convolutions, not network passes

---

## Step 4 — ELA Map Extraction

### JPEG compression internals
- JPEG divides image into **8×8 DCT blocks**, applies discrete cosine transform, then **quantisation** rounds frequency coefficients
- **Quality factor** controls quantisation table — lower quality = more aggressive rounding, higher quality = more coefficients preserved

### ELA algorithm
1. Save image at **quality 75** using Pillow: `image.save('temp.jpg', quality=75)`
2. Load back and compute absolute difference: `ela = abs(original - resaved)`
3. **Amplify by ×15** so differences are visible to model
4. **Clip to 0–255**, convert to `uint8`
5. Repeat grayscale to 3 channels to match model input

### What ELA reveals
- **Spliced regions** — different JPEG compression history, so they recompress differently
- **Copy-move** — identical ELA patterns create visual symmetry across duplicated region
- **Retouching** — harder; healed/blurred areas show subtle smoothing artifacts

### Limitations
- **ELA is unreliable on PNG** — no JPEG history, so ELA signal is meaningless
- **Multiple JPEG compressions** degrade ELA signal — real-world images shared many times are harder to detect
- **JPEG compression augmentation during training is critical** — simulates real-world degradation

---

## Step 5a — RGB Branch (EfficientNet-B0)

### EfficientNet-B0 architecture
- **Compound scaling** — width, depth, resolution scaled together using a fixed coefficient
- **MBConv blocks** (Mobile Inverted Bottleneck Convolution) — the building block; uses depthwise separable convolutions
- **Depthwise separable convolutions** — depthwise conv + pointwise conv; far fewer parameters than standard conv at similar accuracy
- **Squeeze-and-Excitation (SE) blocks** — channel-wise attention inside each MBConv block; learns which feature channels matter
- **~5.3M params** in backbone; **~9.8M total** in dual-branch model

### Transfer learning with ImageNet pretraining
- Pretrained weights give a head start — learned edge, texture, and object detectors
- **Freeze vs fine-tune** — during Phase 1, freeze the stem and early MBConv blocks; fine-tune later stages
- **Freeze BatchNorm** early — prevents pretrained running statistics from being corrupted by small dataset

### What the RGB branch learns
- **Early layers** — edges, textures, colour transitions at splice boundaries
- **Middle layers** — semantic inconsistencies (lighting, scale mismatch between pasted object and scene)
- **Deep layers** — high-level forgery region representations

### Loading via `timm`
```python
timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
```
- `num_classes=0` removes the classification head; outputs 1280-dim vector after global average pooling

---

## Step 5b — Noise Branch (EfficientNet-B0)

### Same architecture, trained from scratch
- `timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)`
- **No ImageNet pretraining** — ImageNet features encode object semantics, not noise statistics; pretraining would bias the branch away from noise patterns

### Input modification
- Takes SRM noise map (30 channels) + ELA map (3 channels) = **33 channels total**
- Modify first Conv2d layer: re-initialise `model.conv_stem` with `in_channels=33`

### What this branch learns
- **Noise inconsistency** across region boundaries
- **JPEG blocking artifact boundaries** where blocks from different images meet
- **Frequency domain anomalies** — unnatural smoothness or sharpness from editing tools

### Phase 1 training strategy
- Train separately for **5–10 epochs** before joint training
- **Why separate** — pretrained RGB branch would dominate gradients and prevent noise branch from learning

---

## Step 6 — Feature Fusion

### Concatenation
- Each branch outputs **1280-dim** feature vector after global average pooling
- Concatenate along channel dim → **2560-dim** combined vector
- `torch.cat([rgb_features, noise_features], dim=1)`

### 1×1 Convolution
- Reduces 2560 → 512 while learning which RGB-noise feature combinations matter
- `nn.Conv2d(2560, 512, kernel_size=1)` → BatchNorm → ReLU
- The model learns to combine "the image looks suspicious here" with "the noise statistics are inconsistent here"

### Why this fusion over alternatives
- **Early fusion** (combining inputs before branches) — loses specialisation
- **Late fusion** (separate classifiers, average outputs) — no cross-branch interaction
- **Concatenation + 1×1 conv** — middle ground used in MVSS-Net and ManTraNet

---

## Step 7a — Classification Head

### Architecture
- `nn.Linear(512, 1)` → Sigmoid → scalar in [0, 1]
- **Threshold at 0.5**: ≥0.5 = FORGED, <0.5 = AUTHENTIC
- **Confidence** = raw sigmoid × 100

### Binary Cross-Entropy loss
- `nn.BCEWithLogitsLoss()` — numerically stable (sigmoid + BCE in one step)
- **Not softmax cross-entropy** — binary problem; single output neuron is more efficient
- **Class imbalance** — use `pos_weight` argument if dataset has more authentic than forged images

### Forgery type prediction
- JSON output includes `forgery_type` — requires either a separate **3-class head** or **post-hoc classification** based on mask shape analysis

---

## Step 7b — Segmentation Head

### Architecture — U-Net style decoder
- Transposed convolutions (`nn.ConvTranspose2d`) upsample from 7×7 → 224×224
- **Skip connections** from RGB branch encoder (U-Net style) — connect early encoder features to decoder layers
- Final layer: `nn.Conv2d(channels, 1, kernel_size=1)` → Sigmoid → binary mask per pixel

### Skip connections
- Preserve fine spatial detail lost during downsampling — critical for accurate boundary localisation
- Extract intermediate feature maps from EfficientNet-B0 using `features_only=True` in timm

### Dice Loss
- `Dice = 1 - (2 × |pred ∩ target|) / (|pred| + |target|)`
- **Why Dice over BCE** — handles extreme class imbalance (most pixels are authentic, forged region is small)
- **Combined loss**: `total_loss = BCE_classification + 0.5 × Dice_segmentation`
- Why **0.5 weight** on Dice — prevents segmentation from dominating and harming classification accuracy

### Pixel IoU metric
- `intersection / union` of predicted and ground truth mask
- Expected: **0.65–0.76** overall; lower for retouching (**0.55–0.68**)

---

## Step 8 — Post-processing

### Upscaling the mask
- Mask is 224×224 → resize to original image dimensions
- `cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)`
- Use bilinear for smooth boundaries, not nearest-neighbour

### Heatmap overlay
- Apply red colormap via OpenCV: `cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)` or custom red-alpha overlay
- Blend: `cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)`

### Forged area percentage
- `forged_pct = (mask > 0.5).sum() / total_pixels × 100`
- Simple numpy operation on thresholded mask

### Base64 encoding for API response
- Encode heatmap overlay to base64 for JSON:
- `base64.b64encode(cv2.imencode('.jpg', heatmap_overlay)[1]).decode()`

---

## Step 9 — Final Output & Optimisation

### ONNX export
- `torch.onnx.export()` — export trained model to ONNX format
- ONNX Runtime inference: `onnxruntime.InferenceSession`
- Expected speedup: **2–3×** versus raw PyTorch

### INT8 Quantisation
- Post-training static quantisation; requires calibration dataset (a few hundred images)
- Model size drops from **~38MB → ~10MB**
- Expected speedup: additional **2–4×**
- Accuracy trade-off: usually < 1% AUC drop on well-calibrated models

### FastAPI integration
- Async endpoint: `async def detect(file: UploadFile)`
- Load ONNX model once at startup via lifespan events — not per request
- Return JSON: `verdict, confidence, forgery_type, forged_area_percentage, heatmap, processing_time_ms`

### Measuring processing time
- `time.perf_counter()` before preprocessing, `time.perf_counter()` after ONNX inference
- **Target: < 50ms total** — benchmark each step separately to find bottlenecks
