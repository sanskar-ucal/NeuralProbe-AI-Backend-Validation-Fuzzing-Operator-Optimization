"""Latency benchmarking, bottleneck analysis and perf-regression detection."""
from __future__ import annotations

import time

import numpy as np

from .backends import OrtBackend, UnsupportedError
from .case import Case, materialize
from .compare import compare
from .runner import resolve_backends

BOTTLENECK_RATIO = 2.0
OVERHEAD_BOUND_PCT = 30.0


def bench_key(case: Case) -> str:
    q = "" if case.quant == "none" else f" {case.quant}"
    return f"{case.op} [{case.spec.label(case.params)}] {case.dtype}{q}"


def time_runner(run, warmup: int, iters: int, repeats: int) -> dict:
    for _ in range(warmup):
        run()
    medians, all_t = [], []
    for _ in range(repeats):
        ts = np.empty(iters)
        for i in range(iters):
            t0 = time.perf_counter_ns()
            run()
            ts[i] = (time.perf_counter_ns() - t0) / 1e3
        medians.append(float(np.median(ts)))
        all_t.append(ts)
    flat = np.concatenate(all_t)
    return {
        "median_us": min(medians),  # min-of-medians: robust to background noise on shared CI runners
        "p90_us": float(np.percentile(flat, 90)),
        "std_us": float(flat.std()),
        "samples": int(flat.size),
    }


def run_bench(cfg: dict, log=print) -> dict:
    bc = cfg["bench"]
    backends, skipped = resolve_backends(bc["backends"], threads=bc["threads"], log=log)
    results = []
    for spec in bc["cases"]:
        case = Case(spec["op"], spec.get("dtype", "float32"), spec.get("quant", "none"), "normal",
                    spec["params"], spec.get("seed", 1234))
        if not case.spec.valid(case.params):
            raise ValueError(f"invalid bench params: {spec}")
        mat = materialize(case)
        flops = case.spec.flops(case.params)
        entry = {"key": bench_key(case), "op": case.op, "dtype": case.dtype, "quant": case.quant,
                 "params": case.params, "flops": flops, "backends": {}}
        for b in backends:
            ok, why = b.supports(case)
            if not ok:
                continue
            try:
                run = b.prepare(case, mat)
                c = compare(case, mat, run(), cfg.get("tolerances"))
                t = time_runner(run, bc["warmup"], bc["iters"], bc["repeats"])
            except UnsupportedError as e:
                entry["backends"][b.name] = {"status": "UNSUPPORTED", "detail": str(e)[:200]}
                continue
            t["gflops"] = flops / (t["median_us"] * 1e3)
            t["correct"] = c.status != "FAIL"
            if bc.get("profile") and isinstance(b, OrtBackend):
                try:
                    t["ort_profile"] = b.profile(case, mat)
                except Exception as e:
                    t["ort_profile"] = {"error": str(e)[:200]}
            entry["backends"][b.name] = t
        entry["analysis"] = analyze(entry)
        results.append(entry)
        timed = {k: v for k, v in entry["backends"].items() if "median_us" in v}
        log(f"  {entry['key'][:70]:<70} " + "  ".join(f"{k}={v['median_us']:.1f}us" for k, v in timed.items()))
    return {"threads": bc["threads"], "results": results, "skipped_backends": skipped}


def analyze(entry: dict) -> dict:
    timed = {k: v for k, v in entry["backends"].items() if "median_us" in v}
    if not timed:
        return {}
    best = min(timed, key=lambda k: timed[k]["median_us"])
    best_us = timed[best]["median_us"]
    findings = []
    for name, t in timed.items():
        ratio = t["median_us"] / best_us
        t["vs_best"] = ratio
        if name != best and name != "native-ref" and ratio >= BOTTLENECK_RATIO:
            findings.append(f"{name} is {ratio:.1f}x slower than {best}")
        prof = t.get("ort_profile") or {}
        if prof.get("overhead_pct", 0) >= OVERHEAD_BOUND_PCT:
            findings.append(f"{name}: {prof['overhead_pct']:.0f}% of run time is session/dispatch overhead "
                            f"(kernel {prof['kernel_us']:.1f}us of {prof['run_us']:.1f}us) -> overhead-bound")
        if not t.get("correct", True):
            findings.append(f"{name} produced INCORRECT output on this benchmark")
    if "native-opt" in timed and "native-ref" in timed:
        findings.append(f"native-opt speedup over native-ref: "
                        f"{timed['native-ref']['median_us'] / timed['native-opt']['median_us']:.1f}x")
    return {"fastest": best, "findings": findings}


def compare_bench(base: dict, head: dict, threshold: float, min_delta_us: float) -> list[dict]:
    """Per (benchmark, backend): REGRESSION if head is >threshold slower and by more than min_delta_us."""
    base_idx = {(r["key"], b): v for r in base["results"] for b, v in r["backends"].items() if "median_us" in v}
    rows = []
    for r in head["results"]:
        for b, v in r["backends"].items():
            if "median_us" not in v:
                continue
            old = base_idx.pop((r["key"], b), None)
            if old is None:
                rows.append({"key": r["key"], "backend": b, "base_us": None, "head_us": v["median_us"],
                             "ratio": None, "status": "NEW"})
                continue
            ratio = v["median_us"] / old["median_us"]
            delta = v["median_us"] - old["median_us"]
            status = "OK"
            if ratio > 1 + threshold and delta > min_delta_us:
                status = "REGRESSION"
            elif ratio < 1 / (1 + threshold) and -delta > min_delta_us:
                status = "IMPROVEMENT"
            rows.append({"key": r["key"], "backend": b, "base_us": old["median_us"], "head_us": v["median_us"],
                         "ratio": ratio, "status": status})
    for (key, b), old in base_idx.items():
        rows.append({"key": key, "backend": b, "base_us": old["median_us"], "head_us": None, "ratio": None,
                     "status": "MISSING"})
    return rows
