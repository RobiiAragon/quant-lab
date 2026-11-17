"""ONNX Runtime execution providers: CPU always, others only when the build has them."""

from __future__ import annotations

import sys
from typing import Any

EP_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}


def ort() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "onnxruntime missing: pip install onnxruntime (Termux: pkg install python-onnxruntime)"
        ) from exc
    return onnxruntime


def options(alias: str) -> dict[str, Any]:
    if alias == "coreml":
        return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    if alias == "qnn":
        return {"backend_path": "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"}
    return {}


def resolve(requested: str) -> tuple[list[str], dict[str, str]]:
    """Return (usable aliases, {alias: reason skipped})."""
    available = set(ort().get_available_providers())
    usable, skipped = [], {}
    for alias in [a.strip().lower() for a in requested.split(",") if a.strip()]:
        name = EP_ALIASES.get(alias)
        if name is None:
            skipped[alias] = "unknown provider alias"
        elif name not in available:
            skipped[alias] = f"{name} is not in this onnxruntime build"
        else:
            usable.append(alias)
    return usable, skipped


def session(path: str, alias: str = "cpu", fixed_dims: dict[str, int] | None = None) -> Any:
    """``fixed_dims`` pins symbolic dimensions (e.g. batch/seq) for EPs that need static shapes."""
    rt = ort()
    so = rt.SessionOptions()
    so.log_severity_level = 3
    for name, value in (fixed_dims or {}).items():
        so.add_free_dimension_override_by_name(name, value)
    providers: list[Any] = [(EP_ALIASES[alias], options(alias))]
    if alias != "cpu":
        providers.append("CPUExecutionProvider")
    return rt.InferenceSession(path, sess_options=so, providers=providers)
