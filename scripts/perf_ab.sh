#!/usr/bin/env bash
# A/B perf comparison of the native kernels between two git refs on this machine.
#   scripts/perf_ab.sh [base_ref=origin/main] [head_ref=HEAD]
# Exits non-zero if head regresses beyond configs/perf.yaml thresholds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"
WORK="$(mktemp -d)"
trap 'git worktree remove --force "$WORK/base" >/dev/null 2>&1 || true; git worktree remove --force "$WORK/head" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

for side in base head; do
  ref=$([ "$side" = base ] && echo "$BASE_REF" || echo "$HEAD_REF")
  git worktree add --detach "$WORK/$side" "$ref" >/dev/null
  if [ ! -d "$WORK/$side/cpp" ]; then
    echo "$side ($ref) has no native kernels; nothing to compare"; exit 0
  fi
  cmake -S "$WORK/$side/cpp" -B "$WORK/$side/build" -DCMAKE_BUILD_TYPE=Release -DNP_BUILD_TESTS=OFF >/dev/null
  cmake --build "$WORK/$side/build" -j >/dev/null
done

# Interleave runs (base, head, base, head) so thermal / frequency drift hits both sides equally.
mkdir -p artifacts
for round in 1 2; do
  for side in base head; do
    lib=$(ls "$WORK/$side"/build/libneuralprobe_kernels.* | head -1)
    NEURALPROBE_NATIVE_LIB="$lib" "$PY" -m neuralprobe bench -c configs/perf.yaml -o "artifacts/perf_${side}_${round}.json"
  done
done

"$PY" - <<'EOF'
import json
for side in ("base", "head"):
    runs = [json.load(open(f"artifacts/perf_{side}_{r}.json")) for r in (1, 2)]
    for res in runs[0]["results"]:
        for b, v in res["backends"].items():
            other = next(x for x in runs[1]["results"] if x["key"] == res["key"])["backends"][b]
            v["median_us"] = min(v["median_us"], other["median_us"])
    json.dump(runs[0], open(f"artifacts/perf_{side}.json", "w"), indent=2)
EOF

"$PY" -m neuralprobe compare-bench artifacts/perf_base.json artifacts/perf_head.json \
  --threshold 0.10 --min-delta-us 3 -o artifacts/perf_diff.md
