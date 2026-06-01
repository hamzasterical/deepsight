import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.inference.postprocess import postprocess
from src.models.dual_branch import DualBranchModel
from src.preprocessing.ela import compute_ela, ela_to_3channel
from src.preprocessing.resize_normalise import resize_image
from src.preprocessing.srm_filters import SRMFilterLayer, extract_srm_noise_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(
        self,
        model: DualBranchModel,
        device: torch.device = None,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.srm_layer = SRMFilterLayer().to(self.device)
        self.srm_layer.eval()

        logger.debug(
            "Predictor initialized (device=%s, conf_thresh=%.2f, mask_thresh=%.2f)",
            self.device, confidence_threshold, mask_threshold,
        )

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        start = time.perf_counter()

        orig_h, orig_w = image.shape[:2]
        original_bgr = image.copy()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_rgb = resize_image(rgb, target_size=224)
        rgb_normalized = resized_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_normalized = (rgb_normalized - mean) / std
        rgb_tensor = torch.from_numpy(rgb_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()

        ela = compute_ela(resized_rgb)
        ela_3ch = ela_to_3channel(ela)
        ela_normalized = ela_3ch.astype(np.float32) / 255.0
        ela_tensor = torch.from_numpy(ela_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()
        if ela_tensor.shape[1] == 1:
            ela_tensor = ela_tensor.repeat(1, 3, 1, 1)

        noise = extract_srm_noise_batch(resized_rgb[np.newaxis, ...], self.srm_layer)
        noise_tensor = torch.from_numpy(noise).permute(0, 3, 1, 2).to(self.device).float()

        noise_input = torch.cat([noise_tensor, ela_tensor], dim=1)

        pred_label, pred_mask = self.model(rgb_tensor, noise_input)
        prob = torch.sigmoid(pred_label).item()
        mask_np = pred_mask.squeeze().cpu().numpy()

        verdict = "FORGED" if prob >= self.confidence_threshold else "AUTHENTIC"
        confidence = round(prob * 100.0, 2)

        forgery_type = self._classify_forgery_type(mask_np, verdict)

        result = postprocess(
            mask=mask_np,
            original_image=original_bgr,
            original_size=(orig_w, orig_h),
            confidence=confidence,
            verdict=verdict,
            forgery_type=forgery_type,
            mask_threshold=self.mask_threshold,
        )

        elapsed = (time.perf_counter() - start) * 1000
        result["processing_time_ms"] = round(elapsed, 2)

        logger.debug("Prediction completed in %.2fms — %s (%.1f%%)", elapsed, verdict, confidence)
        return result

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return self.predict(image)

    def predict_from_path(self, path: Union[str, Path]) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return self.predict(image)

    @staticmethod
    def _classify_forgery_type(mask: np.ndarray, verdict: str) -> str:
        if verdict != "FORGED":
            return "Unknown"
        if mask.size == 0:
            return "Unknown"

        from skimage.measure import label as connected_components

        binary = (mask > 0.5).astype(np.uint8)
        num_labels = connected_components(binary, connectivity=2, return_num=True)[1]

        if num_labels <= 1:
            return "Retouching"
        if num_labels <= 3:
            return "Splicing"
        return "Copy-Move"
