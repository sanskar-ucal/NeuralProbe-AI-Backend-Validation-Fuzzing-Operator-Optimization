#!/usr/bin/env bash
# Build the native kernels, GoogleTest suite and micro-benchmark.
#   scripts/build_native.sh [build_dir] [extra cmake args...]
#   scripts/build_native.sh build-asan -DNP_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-build}"
shift || true
cmake -S "$ROOT/cpp" -B "$ROOT/$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/$BUILD" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
