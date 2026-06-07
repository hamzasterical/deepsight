from pathlib import Path
from typing import Optional, Union

import torch

from src.models.dual_branch import DualBranchModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model: DualBranchModel,
    save_path: Union[str, Path],
    opset_version: int = 13,
) -> str:
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_noise = torch.randn(1, 33, 224, 224)

    device = next(model.parameters()).device
    dummy_rgb = dummy_rgb.to(device)
    dummy_noise = dummy_noise.to(device)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_noise),
        str(save_path),
        opset_version=opset_version,
        input_names=["rgb", "noise"],
        output_names=["label"],
        dynamic_axes={
            "rgb": {0: "batch_size"},
            "noise": {0: "batch_size"},
            "label": {0: "batch_size"},
        },
    )

    logger.info("Model exported to ONNX: %s", save_path)
    return str(save_path)


def quantize_onnx(
    onnx_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.parent / (onnx_path.stem + "_int8" + onnx_path.suffix)

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    logger.info("Quantized model saved: %s", output_path)
    return str(output_path)


def export_and_quantize(
    model: DualBranchModel,
    export_dir: Union[str, Path],
    filename: str = "forgery_model",
) -> dict:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / f"{filename}.onnx"
    int8_path = export_dir / f"{filename}_int8.onnx"

    onnx_path = Path(export_to_onnx(model, onnx_path))
    int8_path = Path(quantize_onnx(onnx_path, int8_path))

    fp_size = onnx_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)

    logger.info("FP32 ONNX: %.2f MB | INT8 ONNX: %.2f MB", fp_size, int8_size)

    return {
        "onnx_path": str(onnx_path),
        "int8_path": str(int8_path),
        "fp32_size_mb": round(fp_size, 2),
        "int8_size_mb": round(int8_size, 2),
    }
