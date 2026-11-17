"""Pick the INT8 recipe on a *dev* set from the training split (never the evaluation set).

Calibration images = the same 128 train images `qlab build` uses; dev = 300 other train images.
Writes results/calibration_ablation_dev.json. Run: python scripts/calibration_ablation.py
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from quant_lab import data
from quant_lab.bench import run_config

RECIPES = {
    "static minmax per-channel": {
        "calibrate_method": CalibrationMethod.MinMax,
        "per_channel": True,
    },
    "static percentile per-channel": {
        "calibrate_method": CalibrationMethod.Percentile,
        "per_channel": True,
    },
    "static entropy per-channel": {
        "calibrate_method": CalibrationMethod.Entropy,
        "per_channel": True,
    },
    "static minmax per-tensor": {
        "calibrate_method": CalibrationMethod.MinMax,
        "per_channel": False,
    },
}


def main() -> None:
    root = data.download()
    train = data.list_split(root, "train")
    calib_s = random.Random(0).sample(train, 128)
    used = {str(s.path) for s in calib_s}
    dev = random.Random(1).sample([s for s in train if str(s.path) not in used], 300)
    calib = np.concatenate([data.to_tensor(data.load_image(s.path)) for s in calib_s])
    imgs, labels = data.load_batch(dev)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter([{"image": calib[i : i + 1]} for i in range(len(calib))])

        def get_next(self):
            return next(self.it, None)

    out: dict[str, dict[str, dict[str, float]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["resnet18", "mobilenetv3-large", "efficientnet-b0"]:
            fp32 = Path(f"models/{name}.fp32.onnx")
            ref_run = run_config(fp32, "cpu", imgs, labels, None, iters=5)
            ref = np.array(ref_run["predictions"])
            res = {"fp32": {"top1": ref_run["top1"], "agreement": 1.0}}
            pre = Path(tmp) / f"{name}.pre.onnx"
            quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
            for tag, kw in RECIPES.items():
                q = Path(tmp) / f"{name}.q.onnx"
                quantize_static(
                    str(pre),
                    str(q),
                    Reader(),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    **kw,
                )
                r = run_config(q, "cpu", imgs, labels, ref, iters=5)
                res[tag] = {"top1": r["top1"], "agreement": r["agreement_with_fp32"]}
            for tag, pc in (("dynamic per-tensor", False), ("dynamic per-channel", True)):
                q = Path(tmp) / f"{name}.d.onnx"
                quantize_dynamic(
                    str(fp32),
                    str(q),
                    weight_type=QuantType.QInt8,
                    per_channel=pc,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                )
                r = run_config(q, "cpu", imgs, labels, ref, iters=5)
                res[tag] = {"top1": r["top1"], "agreement": r["agreement_with_fp32"]}
            out[name] = res
            for tag, v in res.items():
                print(f"{name:<18} {tag:<30} top1 {v['top1']:.3f}  agree {v['agreement']:.3f}")
    path = Path("results/calibration_ablation_dev.json")
    path.write_text(
        json.dumps(
            {"dev_images": len(dev), "calibration_images": len(calib), "results": out}, indent=2
        )
        + "\n"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
