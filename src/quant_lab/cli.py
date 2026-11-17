"""``qlab`` command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

MODELS_DIR = Path("models")
RELEASE = "https://github.com/RobiiAragon/quant-lab/releases/download/v{version}/{name}"


def cmd_build(a: argparse.Namespace) -> int:
    import numpy as np

    from . import data
    from .build import MODELS, build

    root = data.download()
    train = data.list_split(root, "train")
    picks = random.Random(0).sample(train, a.calib)
    calib = np.concatenate([data.to_tensor(data.load_image(s.path)) for s in picks])
    names = list(MODELS) if a.models == "all" else a.models.split(",")
    build(a.models_dir, names, calib)
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    manifest = json.loads((a.models_dir / "manifest.json").read_text())
    for name, info in manifest.items():
        model, variant = name.split(".")[0], name.split(".")[1]
        if a.models != "all" and model not in a.models.split(","):
            continue
        if a.variants != "all" and variant not in a.variants.split(","):
            continue
        target = a.models_dir / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == info["sha256"]:
            print(f"ok   {name}")
            continue
        url = RELEASE.format(version=__version__, name=name)
        print(f"get  {name} ({info['bytes'] / 1024**2:.1f} MB)")
        with urllib.request.urlopen(url, timeout=600) as resp:
            blob = resp.read()
        if hashlib.sha256(blob).hexdigest() != info["sha256"]:
            raise SystemExit(f"checksum mismatch for {name}")
        target.write_bytes(blob)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from . import data, device_info, providers
    from .bench import run_all
    from .build import MODELS, VARIANTS

    usable, skipped = providers.resolve(a.providers)
    for alias, why in skipped.items():
        print(f"[skip] {alias}: {why}")
    root = data.download()
    samples = data.subset(data.list_split(root, "val"), a.limit)
    print(f"decoding {len(samples)} validation images ...")
    images, labels = data.load_batch(samples)
    names = list(MODELS) if a.models == "all" else a.models.split(",")
    variants = VARIANTS if a.variants == "all" else a.variants.split(",")
    if "fp32" in variants:  # reference first
        variants = ["fp32"] + [v for v in variants if v != "fp32"]
    records = run_all(a.models_dir, names, variants, usable, images, labels, a.iters)
    info = device_info.collect()
    out = a.out or Path("results") / "local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "version": __version__,
        "label": a.label or info.get("cpu") or info.get("model"),
        "device": info,
        "n_images": len(samples),
        "skipped_providers": skipped,
        "results": records,
    }
    out.write_text(json.dumps(payload) + "\n")
    print(f"wrote {out}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .report import write

    paths = write(a.results, a.out_dir)
    for p in paths:
        print(f"wrote {p}")
    return 0


def cmd_providers(_: argparse.Namespace) -> int:
    from . import providers

    rt = providers.ort()
    print(f"onnxruntime {rt.__version__}: {', '.join(rt.get_available_providers())}")
    _, skipped = providers.resolve(",".join(providers.EP_ALIASES))
    for alias, why in skipped.items():
        print(f"  [skip] {alias}: {why}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qlab", description="Quantization lab")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="export torchvision models + all variants ([train] extra)")
    b.add_argument("--models", default="all")
    b.add_argument("--calib", type=int, default=128, help="calibration images (train split)")
    b.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("download-models", help="fetch released ONNX files (SHA-256 checked)")
    d.add_argument("--models", default="all")
    d.add_argument("--variants", default="all")
    d.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("run", help="accuracy + latency for every model x variant x EP")
    r.add_argument("--models", default="all")
    r.add_argument("--variants", default="all")
    r.add_argument("--providers", default="cpu,coreml,qnn,cuda,nnapi,xnnpack")
    r.add_argument("--limit", type=int, help="evaluate a random subset of N val images")
    r.add_argument("--iters", type=int, default=200, help="latency iterations")
    r.add_argument("--label", default="")
    r.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    r.add_argument("--out", type=Path)
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="charts (PNG) + self-contained HTML report")
    rep.add_argument("results", type=Path, nargs="+")
    rep.add_argument("--out-dir", type=Path, default=Path("docs"))
    rep.set_defaults(func=cmd_report)

    sub.add_parser("providers", help="list ONNX Runtime execution providers").set_defaults(
        func=cmd_providers
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
