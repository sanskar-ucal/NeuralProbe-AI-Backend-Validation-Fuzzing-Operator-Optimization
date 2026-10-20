"""Numerical comparison of a backend output against the fp64 oracle.

Tolerance model (per element i), with R = reduction length and m_i = OpSpec.magnitude():
  tol_i = rtol * |ref_i| + atol + k * eps(dtype) * sqrt(R) * m_i
    -- for matmul/conv m = op(|A|, |B|), i.e. a probabilistic forward-error bound for a length-R dot
       product, so large-K cases aren't flagged for legitimate rounding and outputs that cancel to ~0
       aren't hidden by a fixed atol. LayerNorm/Softmax provide their own magnitudes.
  quantized outputs: + 1 output quantization step (requantization may round a tie differently)

Error is reported in condition-normalized ULPs: err / ulp(max(|ref|, m, atol)). For quantized cases the
unit is one output LSB instead.

Verdicts:
  FAIL  any element exceeds its tolerance, NaN/Inf pattern differs, or shapes differ
  WARN  within tolerance but the error is much larger than PyTorch's own error at the same dtype
  PASS  otherwise
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .case import NP_DTYPES, Case, Materialized

DEFAULT_TOLERANCES = {
    "float32": {"rtol": 1e-5, "atol": 1e-6, "k": 4.0},
    "float16": {"rtol": 2e-3, "atol": 1e-3, "k": 4.0},
}
WARN_ULP_RATIO = 8.0
WARN_MIN_ULP = 16.0


@dataclass
class Comparison:
    status: str
    category: str | None
    max_abs: float = 0.0
    max_rel: float = 0.0
    max_ulp: float = 0.0
    mean_abs: float = 0.0
    n_mismatch: int = 0
    mismatch_frac: float = 0.0
    nonfinite_mismatch: int = 0
    cosine: float = 1.0
    worst_index: list | None = None
    worst_actual: float | None = None
    worst_expected: float | None = None
    worst_tol: float | None = None
    torch_max_ulp: float | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in asdict(self).items()}


def tolerance_for(case: Case, overrides: dict | None = None) -> dict:
    tol = dict(DEFAULT_TOLERANCES[case.dtype])
    if overrides:
        tol.update(overrides.get("default", {}).get(case.dtype, {}))
        tol.update(overrides.get(case.op, {}).get(case.dtype, {}))
    return tol


class _UlpScale:
    """Maps absolute errors to condition-normalized ULPs (or output LSBs for quantized cases)."""

    def __init__(self, case: Case, mat: Materialized, atol: float):
        self.dtype = NP_DTYPES[case.dtype]
        self.lsb = mat.out_scale if case.quant != "none" else 0.0
        ref = np.abs(np.nan_to_num(mat.golden, nan=0.0, posinf=0.0, neginf=0.0))
        if mat.absref is not None:
            ref = np.maximum(ref, np.nan_to_num(mat.absref, nan=0.0, posinf=0.0, neginf=0.0))
        # Below atol errors are irrelevant by definition; flooring here keeps outputs that underflow toward
        # zero (e.g. gelu(-10) ~ 1e-23) from reporting meaningless million-ULP errors.
        tiny = max(float(np.finfo(self.dtype).smallest_normal), atol)
        with np.errstate(over="ignore"):
            self.unit = np.spacing(np.maximum(ref, tiny).astype(self.dtype)).astype(np.float64)

    def __call__(self, err: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        if self.lsb > 0:
            return float(err[mask].max() / self.lsb)
        return float(np.max(err[mask] / self.unit[mask]))


def compare(case: Case, mat: Materialized, actual: np.ndarray, overrides: dict | None = None) -> Comparison:
    dtype = NP_DTYPES[case.dtype]
    golden64 = mat.golden
    actual = np.asarray(actual)
    if actual.shape != golden64.shape:
        return Comparison("FAIL", "shape_mismatch", detail=f"got {actual.shape}, expected {golden64.shape}")

    a = actual.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        g_cast = golden64.astype(dtype).astype(np.float64)  # what a perfect backend can represent

    nan_bad = np.isnan(a) != np.isnan(g_cast)
    inf_bad = (np.isposinf(a) != np.isposinf(g_cast)) | (np.isneginf(a) != np.isneginf(g_cast))
    nonfinite_bad = nan_bad | inf_bad

    tol = tolerance_for(case, overrides)
    fin = np.isfinite(a) & np.isfinite(g_cast)
    err = np.zeros_like(a)
    err[fin] = np.abs(a[fin] - golden64[fin])
    bound = tol["rtol"] * np.abs(np.nan_to_num(golden64, nan=0.0, posinf=0.0, neginf=0.0)) + tol["atol"]
    if mat.absref is not None:
        r = case.spec.reduction(case.params)
        bound = bound + tol["k"] * float(np.finfo(dtype).eps) * np.sqrt(max(r, 1)) * np.nan_to_num(
            mat.absref, nan=0.0, posinf=0.0, neginf=0.0)
    if case.quant != "none":
        bound = bound + 1.01 * mat.out_scale
    exceed = fin & (err > bound)
    bad = exceed | nonfinite_bad
    ulp = _UlpScale(case, mat, tol["atol"])

    n = max(1, a.size)
    c = Comparison("PASS", None)
    if fin.any():
        c.max_abs = float(err[fin].max())
        rel = err[fin] / np.maximum(np.abs(golden64[fin]), float(np.finfo(dtype).smallest_normal))
        c.max_rel = float(rel.max())
        c.mean_abs = float(err[fin].mean())
        c.max_ulp = ulp(err, fin)
        av, gv = a[fin], golden64[fin]
        denom = np.linalg.norm(av) * np.linalg.norm(gv)
        c.cosine = float(np.dot(av, gv) / denom) if denom > 0 else 1.0
    c.n_mismatch = int(bad.sum())
    c.mismatch_frac = c.n_mismatch / n
    c.nonfinite_mismatch = int(nonfinite_bad.sum())
    if mat.torch_same is not None:
        t_fin = np.isfinite(golden64) & np.isfinite(mat.torch_same)
        with np.errstate(invalid="ignore"):
            t_err = np.abs(np.where(t_fin, mat.torch_same - golden64, 0.0))
        c.torch_max_ulp = ulp(t_err, t_fin)

    if bad.any():
        score = np.where(fin, err / np.maximum(bound, 1e-300), 0.0)
        score[nonfinite_bad] = np.inf
        wi = np.unravel_index(int(np.argmax(score)), a.shape)
        c.worst_index = [int(i) for i in wi]
        c.worst_actual = float(a[wi])
        c.worst_expected = float(golden64[wi])
        c.worst_tol = float(bound[wi])
        c.status = "FAIL"
        if c.nonfinite_mismatch:
            c.category = "nonfinite_mismatch"
            c.detail = (f"{int(nan_bad.sum())} NaN / {int(inf_bad.sum())} Inf pattern mismatches; e.g. at "
                        f"{c.worst_index}: got {c.worst_actual:.6g}, expected {c.worst_expected:.6g}")
        else:
            c.category = "numerical_mismatch"
            c.detail = (f"{c.n_mismatch}/{a.size} elements outside tolerance; worst at {c.worst_index}: "
                        f"got {c.worst_actual:.6g}, expected {c.worst_expected:.6g} (tol {c.worst_tol:.3g})")
        return c

    if (case.quant == "none" and c.torch_max_ulp is not None and c.max_ulp > WARN_MIN_ULP
            and c.max_ulp > WARN_ULP_RATIO * max(c.torch_max_ulp, 0.5)):
        c.status = "WARN"
        c.category = "precision_degradation"
        c.detail = f"max {c.max_ulp:.1f} ULP vs PyTorch {c.torch_max_ulp:.1f} ULP at {case.dtype}"
    return c
