"""Measure every (model, variant, execution provider): accuracy, agreement, latency, size."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import coverage, providers
from .data import CLASS_IDS, to_tensor


def _session(path: Path, alias: str) -> tuple[Any, dict[str, int] | None]:
    # Accelerator EPs get a static batch of 1 (NPUs and the ANE require static shapes).
    dims = None if alias == "cpu" else {"batch": 1}
    level = 3 if alias == "cpu" else 1
    rt = providers.ort()

    def make() -> Any:
        so = rt.SessionOptions()
        so.log_severity_level = level
        for k, v in (dims or {}).items():
            so.add_free_dimension_override_by_name(k, v)
        provs: list[Any] = [(providers.EP_ALIASES[alias], providers.options(alias))]
        if alias != "cpu":
            provs.append("CPUExecutionProvider")
        return rt.InferenceSession(str(path), sess_options=so, providers=provs)

    sess, log = coverage.capture_stderr(make)
    return sess, (None if alias == "cpu" else coverage.parse(log))


def run_config(
    path: Path,
    alias: str,
    images: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray | None,
    iters: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    sess, cov = _session(path, alias)
    load_ms = (time.perf_counter() - t0) * 1000
    name = sess.get_inputs()[0].name

    preds = np.empty(len(images), dtype=np.int64)
    top10: list[int] = []
    t0 = time.perf_counter()
    for i, img in enumerate(images):
        logits = sess.run(None, {name: to_tensor(img)})[0][0]
        preds[i] = int(np.argmax(logits))
        top10.append(int(np.argmax(logits[CLASS_IDS])))
        if i == 0:
            first_ms = (time.perf_counter() - t0) * 1000
    eval_s = time.perf_counter() - t0

    x = to_tensor(images[0])
    for _ in range(warmup):
        sess.run(None, {name: x})
    lat = []
    for i in range(iters):
        x = to_tensor(images[i % len(images)])
        a = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - a) * 1000)
    lat_arr = np.array(lat)
    out = {
        "status": "ok",
        "top1": float((preds == labels).mean()),
        "n": len(labels),
        "predictions": preds.tolist(),
        "latency_ms": {
            "p50": float(np.percentile(lat_arr, 50)),
            "p90": float(np.percentile(lat_arr, 90)),
            "p99": float(np.percentile(lat_arr, 99)),
        },
        "images_per_s": len(images) / eval_s,
        "load_ms": load_ms,
        "first_inference_ms": first_ms,
        "size_mb": path.stat().st_size / 1024**2,
        "ep_coverage": cov,
        "providers_in_session": sess.get_providers(),
    }
    # 10-way accuracy: arg-max restricted to the 10 Imagenette classes.
    out["top1_10way"] = float((CLASS_IDS[np.array(top10)] == labels).mean())
    in10 = np.isin(preds, CLASS_IDS)
    out["predicted_outside_imagenette"] = float((~in10).mean())
    if reference is not None:
        out["agreement_with_fp32"] = float((preds == reference).mean())
    return out


def run_all(
    models_dir: Path,
    names: list[str],
    variants: list[str],
    aliases: list[str],
    images: np.ndarray,
    labels: np.ndarray,
    iters: int = 200,
) -> list[dict[str, Any]]:
    records = []
    for name in names:
        reference = None
        for variant in variants:  # fp32 first so it can serve as the agreement reference
            path = models_dir / f"{name}.{variant}.onnx"
            for alias in aliases:
                base = {"model": name, "variant": variant, "provider": alias}
                if not path.exists():
                    records.append({**base, "status": "skipped", "error": f"{path.name} missing"})
                    continue
                try:
                    rec = run_config(path, alias, images, labels, reference, iters)
                except Exception as exc:  # report and keep going
                    rec = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=2),
                    }
                rec = {**base, **rec}
                if variant == "fp32" and alias == "cpu" and rec["status"] == "ok":
                    reference = np.array(rec["predictions"])
                    rec["agreement_with_fp32"] = 1.0
                records.append(rec)
                if rec["status"] == "ok":
                    cov = coverage.summarize(rec["ep_coverage"]) if rec["ep_coverage"] else "all"
                    print(
                        f"{name:<18} {variant:<13} {alias:<7} top1 {rec['top1'] * 100:6.2f}%  "
                        f"agree {rec.get('agreement_with_fp32', float('nan')) * 100:6.2f}%  "
                        f"p50 {rec['latency_ms']['p50']:7.2f} ms  {rec['size_mb']:6.1f} MB  {cov}"
                    )
                else:
                    print(
                        f"{name:<18} {variant:<13} {alias:<7} {rec['status'].upper()}: "
                        f"{rec['error'][:120]}"
                    )
    return records
