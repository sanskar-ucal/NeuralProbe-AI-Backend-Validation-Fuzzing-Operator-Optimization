"""Randomized differential-testing campaign across backends."""
from __future__ import annotations

import fnmatch
import re
import time
from pathlib import Path

import numpy as np

from .backends import Backend, UnsupportedError, make_backend
from .case import Case, materialize, sample_case
from .compare import compare
from .repro import save_bundle, shrink
from .triage import triage


def _save_minimized(case: Case, b: Backend, tol_over: dict, repro_dir: Path, out_dir: Path, row: dict,
                    tri: dict | None) -> None:
    small = shrink(case, b, tol_over)
    if small is None:
        return
    smat = materialize(small)
    try:
        sact = b.run(small, smat)
        scmp, serr = compare(small, smat, sact, tol_over), None
    except Exception as e:
        sact, scmp, serr = None, None, f"{type(e).__name__}: {e}"[:500]
    sb = save_bundle(repro_dir, small, smat, b, scmp, sact, serr, minimized_from=case.case_id,
                     tolerance_overrides=tol_over, triage=tri)
    row["minimized_repro"] = str(sb.relative_to(out_dir))
    row["minimized_label"] = small.spec.label(small.params)


def match_waiver(row: dict, waivers: list[dict]) -> dict | None:
    """Waivers match on glob fields (op, backend, dtype, quant, dist, category) and an optional label_regex."""
    for w in waivers:
        keys = [k for k in ("op", "backend", "dtype", "quant", "dist", "category") if k in w]
        if not all(fnmatch.fnmatch(str(row.get(k)), str(w[k])) for k in keys):
            continue
        if "label_regex" in w and not re.search(w["label_regex"], row.get("label", "")):
            continue
        return w
    return None


def resolve_backends(names: list[str], threads: int = 1, log=print) -> tuple[list[Backend], dict]:
    backends, skipped = [], {}
    for name in names:
        b = make_backend(name, threads)
        ok, why = b.available()
        if ok:
            backends.append(b)
        else:
            skipped[name] = why
            log(f"  [skip backend] {name}: {why}")
    return backends, skipped


def run_fuzz(cfg: dict, out_dir: Path, log=print) -> dict:
    fz = cfg["fuzz"]
    tol_over = cfg.get("tolerances") or {}
    waivers = cfg.get("waivers") or []
    repro_dir = out_dir / "repros"
    rng = np.random.default_rng(cfg["seed"])
    backends, skipped = resolve_backends(fz["backends"], log=log)
    rows: list[dict] = []
    repro_counts: dict[tuple, int] = {}
    first_seen: set[tuple] = set()
    t_start = time.time()
    budget_hit = False

    for op in fz["ops"]:
        op_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        counts: dict[str, int] = {}
        for _ in range(fz["cases_per_op"]):
            if time.time() - t_start > fz["time_budget_s"]:
                budget_hit = True
                break
            case = sample_case(op, op_rng, fz["dtypes"], fz["quant"], fz["edge_prob"])
            mat = materialize(case)
            for b in backends:
                ok, why = b.supports(case)
                if not ok:
                    continue
                row = {"case_id": case.case_id, "op": op, "dtype": case.dtype, "quant": case.quant,
                       "dist": case.dist, "edge": case.edge, "label": case.spec.label(case.params),
                       "backend": b.name, "category": None, "detail": "", "repro": None}
                t0 = time.perf_counter()
                actual, comparison, error = None, None, None
                try:
                    actual = b.run(case, mat)
                    comparison = compare(case, mat, actual, tol_over)
                    row["status"], row["category"] = comparison.status, comparison.category
                    row["detail"] = comparison.detail
                    row.update({k: getattr(comparison, k) for k in
                                ("max_abs", "max_rel", "max_ulp", "torch_max_ulp", "mismatch_frac", "cosine")})
                except UnsupportedError as e:
                    row["status"], row["detail"] = "UNSUPPORTED", str(e)[:200]
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"[:500]
                    row["status"], row["category"], row["detail"] = "FAIL", "backend_error", error
                row["ms"] = round((time.perf_counter() - t0) * 1e3, 2)

                if row["status"] == "FAIL":
                    w = match_waiver(row, waivers)
                    if w:
                        row["waived"], row["waiver_reason"] = True, w.get("reason", "")
                    sig = (op, b.name, case.dtype, case.quant, row["category"])
                    if repro_counts.get(sig, 0) < fz["max_repros_per_signature"]:
                        repro_counts[sig] = repro_counts.get(sig, 0) + 1
                        first = sig not in first_seen
                        first_seen.add(sig)
                        tri = triage(case, b, tol_over) if first and fz.get("triage", True) else None
                        if tri:
                            row["triage"] = tri["hint"]
                        bundle = save_bundle(repro_dir, case, mat, b, comparison, actual, error,
                                             tolerance_overrides=tol_over, triage=tri)
                        row["repro"] = str(bundle.relative_to(out_dir))
                        if fz["shrink"] and first and not w:
                            _save_minimized(case, b, tol_over, repro_dir, out_dir, row, tri)
                rows.append(row)
                key = row["status"] + ("(waived)" if row.get("waived") else "")
                counts[key] = counts.get(key, 0) + 1
        log(f"  {op:<18} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if budget_hit:
            log("  [time budget exhausted]")
            break

    return {"rows": rows, "skipped_backends": skipped, "backends": [b.info() for b in backends],
            "elapsed_s": round(time.time() - t_start, 2), "budget_hit": budget_hit}


def gate(fuzz_result: dict) -> list[dict]:
    """Rows that should fail CI: unwaived FAILs."""
    return [r for r in fuzz_result["rows"] if r["status"] == "FAIL" and not r.get("waived")]
