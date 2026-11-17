"""Hardware description for result files (no hostnames or serial numbers)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["os_version"] = _run(["sw_vers", "-productVersion"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            info["memory_gb"] = round(int(mem) / 1024**3, 1)
    elif shutil.which("getprop"):
        info["os"] = "Android"
        info["model"] = _run(["getprop", "ro.product.model"])
        info["soc"] = _run(["getprop", "ro.soc.model"])
        info["os_version"] = _run(["getprop", "ro.build.version.release"])
    else:
        info["cpu"] = platform.processor()
        info["os_version"] = platform.release()
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info
