#!/usr/bin/env bash
# Everything CI runs, locally: native build + GoogleTest, pytest, then the full regression gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "== native kernels =="
scripts/build_native.sh build
ctest --test-dir build --output-on-failure
./build/bench_kernels --quick

echo "== framework tests =="
"$PY" -m pytest -q

echo "== regression gate =="
BASELINE="baselines/$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m).json"
"$PY" -m neuralprobe regress -c configs/ci.yaml -o artifacts --baseline "$BASELINE"
