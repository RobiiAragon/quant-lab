"""Export torchvision models to ONNX and produce every precision variant.

Variants
  fp32          reference
  fp16          weights and activations in float16 (graph inputs/outputs stay float32)
  int8-dynamic  int8 per-channel weights, activations quantised on the fly (ConvInteger)
  int8-static   QDQ, uint8 activations (Percentile calibration) + int8 per-channel weights
  w8a16-static  QDQ, uint16 activations + int8 weights: the usual Qualcomm HTP recipe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODELS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "mobilenetv3-large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet-b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}
VARIANTS = ["fp32", "fp16", "int8-dynamic", "int8-static", "w8a16-static"]


def export(name: str, out: Path) -> Path:
    import onnx
    import torch
    import torchvision

    fn, weights_enum = MODELS[name]
    weights = getattr(torchvision.models, weights_enum).IMAGENET1K_V1
    model = getattr(torchvision.models, fn)(weights=weights).eval()
    path = out / f"{name}.fp32.onnx"
    torch.onnx.export(
        model,
        (torch.randn(1, 3, 224, 224),),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    proto = onnx.load(str(path), load_external_data=True)
    onnx.save_model(proto, str(path), save_as_external_data=False)
    for leftover in out.glob(path.name + ".data"):
        leftover.unlink()
    return path


def to_fp16(fp32: Path, out: Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
    onnx.save(model, str(out))
    return out


def to_int8_dynamic(fp32: Path, out: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32),
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    return out


def to_static(fp32: Path, out: Path, calib: np.ndarray, activation_bits: int) -> Path:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter([{"image": calib[i : i + 1]} for i in range(len(calib))])

        def get_next(self):
            return next(self.it, None)

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    try:
        quantize_static(
            str(pre),
            str(out),
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt16 if activation_bits == 16 else QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            # Percentile clips activation outliers; it won the dev-set ablation for 8-bit
            # (scripts/calibration_ablation.py). 16-bit has enough range for MinMax.
            calibrate_method=(
                CalibrationMethod.MinMax if activation_bits == 16 else CalibrationMethod.Percentile
            ),
            extra_options={"UseQDQContribOps": activation_bits == 16},
        )
    finally:
        pre.unlink(missing_ok=True)
    return out


def build(models_dir: Path, names: list[str], calib: np.ndarray) -> dict:
    """``calib``: (N,3,224,224) float32 *training* images (never the evaluation set)."""
    import hashlib

    models_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, dict[str, float]] = {}
    for name in names:
        t = {}
        t0 = time.perf_counter()
        fp32 = export(name, models_dir)
        t["fp32"] = time.perf_counter() - t0
        steps = {
            "fp16": (to_fp16, ()),
            "int8-dynamic": (to_int8_dynamic, ()),
            "int8-static": (to_static, (calib, 8)),
            "w8a16-static": (to_static, (calib, 16)),
        }
        for variant, (fn, extra) in steps.items():
            t0 = time.perf_counter()
            fn(fp32, models_dir / f"{name}.{variant}.onnx", *extra)
            t[variant] = time.perf_counter() - t0
        timings[name] = t
        print(f"built {name}: " + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()))
    manifest = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in sorted(models_dir.glob("*.onnx"))
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (models_dir / "build_info.json").write_text(
        json.dumps(
            {
                "calibration_images": len(calib),
                "calibration_split": "train",
                "int8_static_calibration": "percentile",
                "w8a16_calibration": "minmax",
                "int8_dynamic": "per-channel",
                "seconds": timings,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
