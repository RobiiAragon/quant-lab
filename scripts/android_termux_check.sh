#!/data/data/com.termux/files/usr/bin/bash
# quant-lab on Android via Termux (target: Galaxy Tab S9 Ultra, Snapdragon 8 Gen 2).
# Compares FP32 / FP16 / INT8 dynamic / INT8 static / W8A16 on CPU, XNNPACK and NNAPI.
#
#   pkg install -y git && git clone https://github.com/RobiiAragon/quant-lab
#   cd quant-lab && bash scripts/android_termux_check.sh
#   MODELS=all LIMIT=1000 bash scripts/android_termux_check.sh    # bigger run
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "${PREFIX:-}" == *com.termux* ]] || { echo "This script is meant for Termux."; exit 1; }
MODELS="${MODELS:-mobilenetv3-large}"
LIMIT="${LIMIT:-500}"

pkg update -y >/dev/null
pkg install -y python python-numpy python-onnxruntime python-pillow git >/dev/null
python -m pip install -q --no-deps -e .
qlab providers
qlab download-models --models "$MODELS"     # ~50 MB per model for all five variants

TAG=$(getprop ro.product.model | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
OUT="results/android-$TAG.json"
echo "==> benchmark $MODELS on $LIMIT Imagenette images (downloads 99 MB once) -> $OUT"
qlab run --models "$MODELS" --providers cpu,xnnpack,nnapi --limit "$LIMIT" --iters 100 \
  --label "$(getprop ro.product.model) (Android $(getprop ro.build.version.release))" --out "$OUT"
echo "done. Share $OUT (termux-setup-storage && cp $OUT ~/storage/downloads/)"
