"""How much of a graph an execution provider actually takes.

ONNX Runtime logs a line like
  "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 21
   number of nodes in the graph: 122 number of nodes supported by CoreML: 102"
(QNN, NNAPI and XNNPACK print equivalent lines). Every partition boundary is a
CPU<->accelerator hand-off, so low coverage / many partitions usually means slow.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_PATTERN = re.compile(
    r"number of partitions supported by [\w ]+?:\s*(\d+).*?"
    r"number of nodes in the graph:\s*(\d+).*?"
    r"number of nodes supported by [\w ]+?:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)


def parse(log_text: str) -> dict[str, int] | None:
    """Return the last capability summary found in ORT's log output."""
    matches = list(_PATTERN.finditer(log_text))
    if not matches:
        return None
    parts, total, supported = (int(g) for g in matches[-1].groups())
    return {"partitions": parts, "nodes_total": total, "nodes_on_ep": supported}


def capture_stderr(fn: Callable[[], T]) -> tuple[T, str]:
    """Run ``fn`` while capturing the process-level stderr (where ORT's C++ logger writes)."""
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            result = fn()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        return result, tmp.read().decode("utf-8", errors="replace")


def summarize(info: dict[str, Any] | None) -> str:
    if not info:
        return "—"
    return f"{info['nodes_on_ep']}/{info['nodes_total']} nodes, {info['partitions']} part."
