from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from conftest import tiny_classifier

from quant_lab import build, coverage, data, report
from quant_lab.bench import run_all, run_config


def test_synsets_map_to_distinct_imagenet_classes() -> None:
    assert len(data.SYNSETS) == 10 and len(set(data.SYNSETS.values())) == 10
    assert data.SYNSETS["n01440764"] == 0 and data.SYNSETS["n03888257"] == 701


def test_preprocessing(fake_imagenette) -> None:
    val = data.list_split(fake_imagenette, "val")
    assert len(val) == 8 and {s.label for s in val} == {0, 217}
    img = data.load_image(val[0].path)
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    x = data.to_tensor(np.full((224, 224, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224)
    assert np.allclose(x[0, :, 0, 0], (1 - data.MEAN) / data.STD)
    assert data.subset(val, 3, seed=5) == data.subset(val, 3, seed=5)


def test_all_variants_build_and_run(tmp_path, fake_imagenette) -> None:
    fp32 = tiny_classifier(tmp_path / "tiny.fp32.onnx")
    imgs, labels = data.load_batch(data.list_split(fake_imagenette, "val"))
    calib = np.concatenate([data.to_tensor(i) for i in imgs[:4]])
    build.to_fp16(fp32, tmp_path / "tiny.fp16.onnx")
    build.to_int8_dynamic(fp32, tmp_path / "tiny.int8-dynamic.onnx")
    build.to_static(fp32, tmp_path / "tiny.int8-static.onnx", calib, 8)
    build.to_static(fp32, tmp_path / "tiny.w8a16-static.onnx", calib, 16)

    ops = {
        v: {n.op_type for n in onnx.load(str(tmp_path / f"tiny.{v}.onnx")).graph.node}
        for v in build.VARIANTS
    }
    assert "Cast" in ops["fp16"]
    assert "ConvInteger" in ops["int8-dynamic"] or "DynamicQuantizeLinear" in ops["int8-dynamic"]
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops["int8-static"]
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops["w8a16-static"]

    records = run_all(tmp_path, ["tiny"], build.VARIANTS, ["cpu"], imgs, labels, iters=5)
    assert [r["status"] for r in records] == ["ok"] * 5
    assert records[0]["agreement_with_fp32"] == 1.0
    for r in records:
        assert 0 <= r["top1"] <= 1 and 0 <= r["top1_10way"] <= 1
        assert r["latency_ms"]["p50"] > 0 and r["size_mb"] > 0
        assert "agreement_with_fp32" in r
    # fp16 halves the weights
    assert records[1]["size_mb"] < records[0]["size_mb"] * 0.7


def test_missing_variant_is_reported(tmp_path, fake_imagenette) -> None:
    tiny_classifier(tmp_path / "tiny.fp32.onnx")
    imgs, labels = data.load_batch(data.list_split(fake_imagenette, "val")[:2])
    recs = run_all(tmp_path, ["tiny"], ["fp32", "fp16"], ["cpu"], imgs, labels, iters=2)
    assert recs[1]["status"] == "skipped" and "missing" in recs[1]["error"]


def test_run_config_agreement(tmp_path, fake_imagenette) -> None:
    fp32 = tiny_classifier(tmp_path / "t.onnx")
    imgs, labels = data.load_batch(data.list_split(fake_imagenette, "val"))
    first = run_config(fp32, "cpu", imgs, labels, None, iters=3)
    again = run_config(fp32, "cpu", imgs, labels, np.array(first["predictions"]), iters=3)
    assert again["agreement_with_fp32"] == 1.0 and again["ep_coverage"] is None


def test_coverage_parser() -> None:
    log = (
        "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 2 "
        "number of nodes in the graph: 49 number of nodes supported by CoreML: 48"
    )
    assert coverage.parse(log) == {"partitions": 2, "nodes_total": 49, "nodes_on_ep": 48}


def _fake_run() -> dict:
    rows = []
    for model in ("m1", "m2"):
        for i, v in enumerate(build.VARIANTS):
            for ep, scale in (("cpu", 1.0), ("coreml", 0.5)):
                rows.append(
                    {
                        "model": model,
                        "variant": v,
                        "provider": ep,
                        "status": "ok",
                        "top1": 0.8 - 0.02 * i,
                        "top1_10way": 0.97,
                        "agreement_with_fp32": 1 - 0.01 * i,
                        "latency_ms": {"p50": scale * (5 - i * 0.5), "p99": 6.0},
                        "size_mb": 40 / (1 + i),
                        "ep_coverage": None
                        if ep == "cpu"
                        else {"partitions": 2, "nodes_total": 50, "nodes_on_ep": 48},
                    }
                )
    rows.append(
        {
            "model": "m1",
            "variant": "fp16",
            "provider": "qnn",
            "status": "error",
            "error": "QNN <boom>",
        }
    )
    return {
        "created": "now",
        "label": "Test <Mac>",
        "device": {"os": "Darwin"},
        "n_images": 10,
        "results": rows,
    }


def test_report_outputs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    res = tmp_path / "r.json"
    res.write_text(json.dumps(_fake_run()))
    paths = report.write([res], tmp_path / "out")
    names = {p.name for p in paths}
    assert {"tradeoff.png", "accuracy_drop.png", "latency.png", "size.png", "report.html"} <= names
    html = (tmp_path / "out" / "report.html").read_text()
    assert "Test &lt;Mac&gt;" in html and "QNN &lt;boom&gt;" in html
    assert "data:image/png;base64," in html and "<script" not in html
    assert "http://" not in html and "https://" not in html
