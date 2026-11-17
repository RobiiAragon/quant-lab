"""Imagenette (fast.ai, Apache-2.0 repository; images are a 10-class ImageNet subset).

Accuracy is ImageNet top-1 over all 1000 classes: a prediction is correct only if the
model's arg-max is the exact ImageNet class of the Imagenette folder.
"""

from __future__ import annotations

import hashlib
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
# Imagenette synset -> ImageNet-1k class index
SYNSETS = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
CLASS_IDS = np.array(sorted(SYNSETS.values()))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cache_dir() -> Path:
    path = Path(os.environ.get("QLAB_CACHE", Path.home() / ".cache" / "quant-lab"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def download() -> Path:
    """Download + verify + safely extract; returns the imagenette2-160 directory."""
    root = cache_dir()
    out = root / "extract" / "imagenette2-160"
    if (out / "val").is_dir():
        return out
    tgz = root / "imagenette2-160.tgz"
    if not tgz.exists():
        print(f"downloading {URL} (99 MB)")
        urllib.request.urlretrieve(URL, tgz)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if digest != SHA256:
        tgz.unlink()
        raise RuntimeError(f"checksum mismatch for imagenette2-160.tgz: {digest}")
    with tarfile.open(tgz) as tar:
        tar.extractall(root / "extract", filter="data")  # rejects absolute/.. paths and links
    return out


@dataclass
class Sample:
    path: Path
    label: int  # ImageNet-1k index


def list_split(root: Path, split: str) -> list[Sample]:
    samples = []
    for synset, idx in SYNSETS.items():
        for p in sorted((root / split / synset).glob("*.JPEG")):
            samples.append(Sample(p, idx))
    return samples


def subset(samples: list[Sample], n: int | None, seed: int = 0) -> list[Sample]:
    if n is None or n >= len(samples):
        return samples
    rng = random.Random(seed)
    return sorted(rng.sample(samples, n), key=lambda s: str(s.path))


def load_image(path: Path, size: int = 224, resize: int = 256) -> np.ndarray:
    """Standard ImageNet eval transform: resize shorter side to 256, center-crop 224 (uint8)."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = resize / min(w, h)
        im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.uint8)


def to_tensor(img_u8: np.ndarray) -> np.ndarray:
    """HWC uint8 -> 1x3xHxW float32, ImageNet-normalised."""
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def load_batch(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Decoded uint8 images (N,224,224,3) and labels; decoded once, normalised per run."""
    imgs = np.stack([load_image(s.path) for s in samples])
    return imgs, np.array([s.label for s in samples], dtype=np.int64)
