#!/usr/bin/env bash
# macOS end-to-end check: lint, tests, (re)build or download models, full benchmark on CPU +
# CoreML, figures + HTML report, and the executed notebook.
#   scripts/mac_check.sh            # download released models, ~25 min benchmark (3,925 images)
#   QUICK=1 scripts/mac_check.sh    # 300 images, 50 latency iterations (~3 min)
#   BUILD=1 scripts/mac_check.sh    # re-export + re-quantize locally (needs the [train] extra)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[[ -n "$PY" ]] || { echo "Python 3.11+ not found (brew install python@3.12)"; exit 1; }
[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null
python -m pip install -q --upgrade pip
EXTRA="dev,notebook"; [[ -n "${BUILD:-}" ]] && EXTRA="dev,notebook,train"
python -m pip install -q -e ".[$EXTRA]"

echo "==> lint + tests"
ruff check . && ruff format --check . && pytest -q
qlab providers

if [[ -n "${BUILD:-}" ]]; then qlab build; else qlab download-models; fi

ARGS=(--providers cpu,coreml,qnn,cuda --out results/local-mac.json)
[[ -n "${QUICK:-}" ]] && ARGS+=(--limit 300 --iters 50)
echo "==> benchmark ${ARGS[*]}"
qlab run "${ARGS[@]}"
qlab report results/local-mac.json --out-dir /tmp/quant-lab-report
echo "OK - open /tmp/quant-lab-report/report.html"
