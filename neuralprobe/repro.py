"""Failure reproduction bundles and test-case minimization.

A bundle is a self-contained directory that can be attached to a bug report:
  case.json   case spec, backend config, verdict, metrics, environment
  model.onnx  single-op model (QDQ for quantized cases)
  inputs.npz  graph inputs (activations); weights live in the model
  tensors.npz every tensor incl. weights (for non-ONNX backends)
  golden.npy  fp64 oracle output;  actual.npy  backend output
  repro.py    standalone script: numpy + onnxruntime only
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx

from .backends import Backend, OrtBackend, UnsupportedError, environment
from .case import Case, Materialized, materialize
from .compare import Comparison, compare, tolerance_for
from .onnx_builder import build_model, feeds

REPRO_TEMPLATE = '''#!/usr/bin/env python3
"""Standalone reproduction for NeuralProbe failure {case_id} on {backend}.

{description}
Verdict: {category} -- {detail}
Triage: {triage}
Requires only numpy + onnxruntime.  Run:  python repro.py
"""
import json, pathlib, sys
import numpy as np
import onnxruntime as ort

here = pathlib.Path(__file__).parent
meta = json.loads((here / "case.json").read_text())
providers = [tuple(p) if isinstance(p, list) else p
             for p in meta["backend"].get("providers") or ["CPUExecutionProvider"]]
so = ort.SessionOptions()
if not meta["backend"].get("optimize", True):
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
if meta["backend"].get("disable_cpu_fallback"):
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
for k, v in meta["backend"].get("session_config", {{}}).items():
    so.add_session_config_entry(k, v)
sess = ort.InferenceSession(str(here / "model.onnx"), so, providers=providers)
inputs = dict(np.load(here / "inputs.npz"))
actual = sess.run(None, inputs)[0].astype(np.float64)
golden = np.load(here / "golden.npy")
fin = np.isfinite(actual) & np.isfinite(golden)
err = np.abs(actual[fin] - golden[fin])
print("providers:", sess.get_providers())
print("max abs error:", err.max() if err.size else 0.0, " recorded:", meta["comparison"]["max_abs"])
print("NaN mismatches:", int((np.isnan(actual) != np.isnan(golden.astype(actual.dtype))).sum()))
wi = meta["comparison"].get("worst_index")
if wi:
    print(f"worst element {{wi}}: actual={{actual[tuple(wi)]!r}} expected={{golden[tuple(wi)]!r}}")
sys.exit(1 if meta["comparison"]["status"] == "FAIL" else 0)
'''


def _backend_meta(backend: Backend) -> dict:
    meta = backend.info()
    if isinstance(backend, OrtBackend):
        meta["providers"] = [[backend.provider, backend.provider_options]] if backend.provider_options else [backend.provider]
        if backend.provider != "CPUExecutionProvider" and not backend.disable_cpu_fallback:
            meta["providers"].append("CPUExecutionProvider")
        meta["disable_cpu_fallback"] = backend.disable_cpu_fallback
    return meta


def save_bundle(root: Path, case: Case, mat: Materialized, backend: Backend, comparison: Comparison | None,
                actual: np.ndarray | None, error: str | None = None, minimized_from: str | None = None,
                tolerance_overrides: dict | None = None, triage: dict | None = None) -> Path:
    d = root / f"{case.case_id}__{backend.name}"
    d.mkdir(parents=True, exist_ok=True)
    cmp_dict = comparison.to_dict() if comparison else {"status": "FAIL", "category": "backend_error",
                                                        "detail": error, "max_abs": None}
    meta = {
        "case": case.to_dict(),
        "description": case.describe(),
        "backend": _backend_meta(backend),
        "comparison": cmp_dict,
        "tolerance": tolerance_for(case, tolerance_overrides),
        "error": error,
        "triage": triage,
        "minimized_from": minimized_from,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment(),
    }
    (d / "case.json").write_text(json.dumps(meta, indent=2, default=str))
    onnx.save(build_model(case, mat), str(d / "model.onnx"))
    np.savez(d / "inputs.npz", **feeds(case, mat))
    np.savez(d / "tensors.npz", **mat.inputs)
    np.save(d / "golden.npy", mat.golden)
    if actual is not None:
        np.save(d / "actual.npy", np.asarray(actual))
    (d / "repro.py").write_text(REPRO_TEMPLATE.format(
        case_id=case.case_id, backend=backend.name, description=case.describe(),
        category=cmp_dict.get("category"), detail=(cmp_dict.get("detail") or "").replace('"""', "'"),
        triage=(triage or {}).get("hint", "n/a")))
    return d


def evaluate(case: Case, backend: Backend, overrides: dict | None = None):
    """Runs one case on one backend -> (status, comparison | None, actual | None, error | None, mat)."""
    mat = materialize(case)
    try:
        actual = backend.run(case, mat)
    except UnsupportedError as e:
        return "UNSUPPORTED", None, None, str(e), mat
    except Exception as e:
        return "FAIL", None, None, f"{type(e).__name__}: {e}"[:500], mat
    c = compare(case, mat, actual, overrides)
    return c.status, c, actual, None, mat


def _signature(status, comparison, error) -> tuple:
    if comparison is not None:
        return status, comparison.category
    return status, "backend_error" if error else None


def shrink(case: Case, backend: Backend, overrides: dict | None = None, budget: int = 40) -> Case | None:
    """Greedy delta-debugging over shape params: keep any smaller case that fails the same way."""
    status, comparison, _, error, _ = evaluate(case, backend, overrides)
    target = _signature(status, comparison, error)
    if status != "FAIL":
        return None
    best, tries, improved = case, 0, True
    while improved and tries < budget:
        improved = False
        for params in best.spec.shrink(best.params):
            tries += 1
            cand = best.with_params(params)
            s, c, _, e, _ = evaluate(cand, backend, overrides)
            if _signature(s, c, e) == target:
                best, improved = cand, True
                break
            if tries >= budget:
                break
    return best if best is not case else None


def replay(bundle: Path, backend: Backend | None = None) -> tuple[str, Comparison | None, str | None]:
    """Re-runs a saved bundle through the framework (works for native backends too)."""
    from .backends import make_backend

    meta = json.loads((bundle / "case.json").read_text())
    case = Case.from_dict(meta["case"])
    backend = backend or make_backend(meta["backend"]["name"])
    status, comparison, _, error, _ = evaluate(case, backend)
    return status, comparison, error
