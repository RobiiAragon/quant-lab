from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_lab.data import SYNSETS


def tiny_classifier(path: Path) -> Path:
    """Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm(1000): same I/O as the real models."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    inits = [
        numpy_helper.from_array(rng.normal(0, 0.2, (8, 3, 3, 3)).astype(np.float32), "w"),
        numpy_helper.from_array(np.zeros(8, np.float32), "b"),
        numpy_helper.from_array(rng.normal(0, 0.5, (1000, 8)).astype(np.float32), "fc_w"),
        numpy_helper.from_array(np.zeros(1000, np.float32), "fc_b"),
    ]
    nodes = [
        helper.make_node("Conv", ["image", "w", "b"], ["c"], pads=[1, 1, 1, 1], strides=[4, 4]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["g"]),
        helper.make_node("Flatten", ["g"], ["f"]),
        helper.make_node("Gemm", ["f", "fc_w", "fc_b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1000])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture()
def fake_imagenette(tmp_path: Path) -> Path:
    """imagenette2-160 layout with a few random JPEGs in two classes."""
    from PIL import Image

    root = tmp_path / "imagenette2-160"
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        for synset in list(SYNSETS)[:2]:
            d = root / split / synset
            d.mkdir(parents=True)
            for i in range(4):
                arr = rng.integers(0, 255, (160, 213, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"img_{i}.JPEG")
    return root
