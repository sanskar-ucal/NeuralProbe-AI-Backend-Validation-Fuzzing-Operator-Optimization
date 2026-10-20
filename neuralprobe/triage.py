"""Automatic failure triage: re-run a failing case under component toggles to localize the culprit.

For an ORT failure we bisect the runtime configuration (graph optimizer, weight prepacking, KleidiAI
micro-kernels, and for accelerator EPs the CPU EP itself). For the optimized native kernel we re-run the
naive reference kernel. A toggle under which the failure disappears points at the responsible component.
"""
from __future__ import annotations

from .backends import Backend, NativeBackend, OrtBackend
from .case import Case
from .repro import evaluate

ORT_TOGGLES = [
    ("graph optimizations disabled", {"optimize": False}),
    ("weight prepacking disabled", {"session_config": {"session.disable_prepacking": "1"}}),
    ("KleidiAI micro-kernels disabled", {"session_config": {"mlas.disable_kleidiai": "1"}}),
]


def _variants(backend: Backend) -> list[tuple[str, Backend]]:
    if isinstance(backend, OrtBackend):
        out = []
        if backend.provider != "CPUExecutionProvider":
            out.append(("same model on ORT CPU EP", backend.variant(
                provider="CPUExecutionProvider", provider_options={}, disable_cpu_fallback=False)))
        else:
            out += [(label, backend.variant(**changes)) for label, changes in ORT_TOGGLES]
        return out
    if isinstance(backend, NativeBackend) and backend.variant == "opt":
        return [("naive native-ref kernel", NativeBackend("native-ref", "ref")),
                ("optimized kernel, 1 thread", NativeBackend(backend.name, "opt", threads=1))]
    return []


def triage(case: Case, backend: Backend, overrides: dict | None = None) -> dict:
    results = []
    for label, b in _variants(backend):
        status, comparison, _, error, _ = evaluate(case, b, overrides)
        results.append({"toggle": label, "status": status,
                        "max_abs": comparison.max_abs if comparison else None, "error": error})
    fixed = [r["toggle"] for r in results if r["status"] in ("PASS", "WARN")]
    if not results:
        hint = "no triage variants for this backend"
    elif fixed:
        hint = "failure disappears with: " + "; ".join(fixed)
    else:
        hint = "failure persists under all toggles (core kernel or op semantics)"
    return {"hint": hint, "variants": results}
