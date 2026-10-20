"""neuralprobe command line.

  neuralprobe backends                         list backends and availability
  neuralprobe fuzz    -c configs/ci.yaml       randomized differential testing
  neuralprobe bench   -c configs/ci.yaml       latency benchmarks + bottleneck analysis
  neuralprobe regress -c configs/ci.yaml       fuzz + bench + baseline comparison + reports; CI gate
  neuralprobe compare-bench base.json head.json
  neuralprobe repro   artifacts/repros/<bundle>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import BACKEND_NAMES, environment, load_native, make_backend
from .config import load_config
from .profiler import compare_bench, run_bench
from .report import perf_sections, write_reports
from .repro import replay
from .runner import gate, print_failures, run_fuzz


def _apply_overrides(cfg: dict, args) -> dict:
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "cases", None):
        cfg["fuzz"]["cases_per_op"] = args.cases
    if getattr(args, "ops", None):
        cfg["fuzz"]["ops"] = args.ops.split(",")
    if getattr(args, "backends", None):
        cfg["fuzz"]["backends"] = args.backends.split(",")
    if getattr(args, "no_shrink", False):
        cfg["fuzz"]["shrink"] = False
    return cfg


def _native_isa() -> str:
    try:
        return load_native().np_isa_name().decode()
    except Exception:
        return "unavailable"


def _summary(cfg, fuzz=None, bench=None, regressions=None) -> dict:
    return {"config": {k: cfg[k] for k in ("name", "seed")} | {"fuzz": cfg["fuzz"], "bench": {
        k: v for k, v in cfg["bench"].items() if k != "cases"}}, "environment": environment(),
            "native_isa": _native_isa(), "fuzz": fuzz, "bench": bench, "regressions": regressions,
            "failed": False, "fail_reasons": []}


def _publish(summary: dict, out: Path) -> None:
    paths = write_reports(summary, out)
    print(f"\nreports: {paths['markdown']}  {paths['html']}  {paths['json']}")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(paths["markdown"].read_text())


def cmd_backends(args) -> int:
    for name in BACKEND_NAMES:
        b = make_backend(name)
        ok, why = b.available()
        print(f"  {name:<15} {'available' if ok else 'unavailable'}  {'' if ok else why}")
    print(f"  native ISA: {_native_isa()}")
    return 0


def cmd_fuzz(args) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    out = Path(args.out)
    print(f"[fuzz] config={cfg['name']} seed={cfg['seed']} cases/op={cfg['fuzz']['cases_per_op']}")
    fuzz = run_fuzz(cfg, out)
    summary = _summary(cfg, fuzz=fuzz)
    bad = gate(fuzz)
    print_failures(bad)
    if bad:
        summary["failed"] = True
        summary["fail_reasons"].append(f"{len(bad)} unwaived correctness failure(s)")
    _publish(summary, out)
    return 1 if summary["failed"] else 0


def cmd_bench(args) -> int:
    cfg = load_config(args.config)
    print(f"[bench] {len(cfg['bench']['cases'])} benchmarks, threads={cfg['bench']['threads']}")
    bench = run_bench(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bench, indent=2))
    print(f"wrote {args.out}")
    return 0


def cmd_compare_bench(args) -> int:
    base, head = json.loads(Path(args.base).read_text()), json.loads(Path(args.head).read_text())
    rows = compare_bench(base, head, args.threshold, args.min_delta_us)
    md = "\n\n".join(perf_sections(head, rows))
    if args.out:
        Path(args.out).write_text(md + "\n")
    print(md)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(md + "\n")
    return 1 if any(r["status"] == "REGRESSION" for r in rows) else 0


def cmd_regress(args) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[regress] config={cfg['name']} seed={cfg['seed']}")
    print("[1/3] correctness fuzzing")
    fuzz = run_fuzz(cfg, out)
    bench, regressions = None, None
    if not args.skip_bench and cfg["bench"]["cases"]:
        print("[2/3] benchmarks")
        bench = run_bench(cfg)
        (out / "bench.json").write_text(json.dumps(bench, indent=2))
        baseline = Path(args.baseline) if args.baseline else None
        if baseline and baseline.exists():
            rg = cfg["regression"]
            regressions = compare_bench(json.loads(baseline.read_text()), bench, rg["perf_threshold"],
                                        rg["min_delta_us"])
        if args.update_baseline and baseline:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(json.dumps(bench, indent=2))
            print(f"baseline updated: {baseline}")
    print("[3/3] report")
    summary = _summary(cfg, fuzz, bench, regressions)
    bad = gate(fuzz)
    print_failures(bad)
    if bad:
        summary["failed"] = True
        summary["fail_reasons"].append(f"{len(bad)} unwaived correctness failure(s)")
    n_reg = sum(r["status"] == "REGRESSION" for r in regressions or [])
    if n_reg and cfg["regression"]["fail_on_perf"]:
        summary["failed"] = True
        summary["fail_reasons"].append(f"{n_reg} performance regression(s) vs baseline")
    _publish(summary, out)
    print("RESULT:", "FAIL — " + "; ".join(summary["fail_reasons"]) if summary["failed"] else "PASS")
    return 1 if summary["failed"] else 0


def cmd_repro(args) -> int:
    bundle = Path(args.bundle)
    backend = make_backend(args.backend) if args.backend else None
    status, comparison, error = replay(bundle, backend)
    print(json.loads((bundle / "case.json").read_text())["description"])
    print("status:", status)
    if comparison:
        print(json.dumps({k: v for k, v in comparison.to_dict().items() if v is not None}, indent=2))
    if error:
        print("error:", error)
    return 1 if status == "FAIL" else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neuralprobe", description="AI backend validation, fuzzing & profiling")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backends").set_defaults(fn=cmd_backends)

    def common(sp):
        sp.add_argument("-c", "--config", default="configs/ci.yaml")
        sp.add_argument("-o", "--out", default="artifacts")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--cases", type=int, help="cases per op")
        sp.add_argument("--ops", help="comma-separated op list")
        sp.add_argument("--backends", help="comma-separated backend list")
        sp.add_argument("--no-shrink", action="store_true")

    f = sub.add_parser("fuzz")
    common(f)
    f.set_defaults(fn=cmd_fuzz)

    b = sub.add_parser("bench")
    b.add_argument("-c", "--config", default="configs/ci.yaml")
    b.add_argument("-o", "--out", default="artifacts/bench.json")
    b.set_defaults(fn=cmd_bench)

    cb = sub.add_parser("compare-bench")
    cb.add_argument("base")
    cb.add_argument("head")
    cb.add_argument("--threshold", type=float, default=0.15)
    cb.add_argument("--min-delta-us", type=float, default=5.0)
    cb.add_argument("-o", "--out")
    cb.set_defaults(fn=cmd_compare_bench)

    r = sub.add_parser("regress")
    common(r)
    r.add_argument("--baseline", help="bench JSON to compare against")
    r.add_argument("--update-baseline", action="store_true")
    r.add_argument("--skip-bench", action="store_true")
    r.set_defaults(fn=cmd_regress)

    rp = sub.add_parser("repro")
    rp.add_argument("bundle")
    rp.add_argument("--backend", help="override backend (e.g. replay a CoreML failure on ort-cpu)")
    rp.set_defaults(fn=cmd_repro)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
