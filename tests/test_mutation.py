"""Mutation testing of the harness itself: inject realistic kernel bugs and require the fuzzer to catch them."""
import numpy as np
import pytest

from neuralprobe.backends import ROOT
from neuralprobe.case import Case, materialize
from neuralprobe.config import load_config
from neuralprobe.repro import shrink
from neuralprobe.runner import gate, run_fuzz

from .conftest import MutantBackend


def _tail_bug(case, y):
    """Off-by-one SIMD tail: last column left unwritten when the row length isn't a multiple of 8."""
    if y.ndim and y.shape[-1] % 8:
        y[..., -1] = 0
    return y


def _nan_to_zero(case, y):
    """max(x, 0)-style kernel that swallows NaN."""
    return np.nan_to_num(y, nan=0.0)


def _scale_drift(case, y):
    """Slight systematic scale error (0.05%), e.g. a wrong requantization multiplier."""
    return (y * np.asarray(1.0005, y.dtype)).astype(y.dtype)


def _softmax_wrong_axis(case, y):
    if case.op != "softmax":
        return y
    x = materialize(case, with_reference=False).inputs["X"].astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(y.dtype)


@pytest.mark.parametrize("mutate,ops", [
    (_tail_bug, ["matmul", "relu"]),
    (_nan_to_zero, ["relu", "gelu"]),
    (_scale_drift, ["matmul", "conv2d", "layernorm"]),
    (_softmax_wrong_axis, ["softmax"]),
])
def test_mutant_is_caught(cfg, tmp_path, monkeypatch, mutate, ops):
    import neuralprobe.runner as runner

    mutant = MutantBackend(f"mutant-{mutate.__name__}", mutate)
    monkeypatch.setattr(runner, "resolve_backends", lambda names, threads=1, log=print: ([mutant], {}))
    cfg["fuzz"].update({"ops": ops, "cases_per_op": 25, "edge_prob": 0.7, "dtypes": ["float32"], "quant": ["none"]})
    result = run_fuzz(cfg, tmp_path, log=lambda *_: None)
    caught = gate(result)
    assert caught, f"{mutate.__name__} escaped detection"
    assert any(r.get("repro") for r in caught)
    assert (tmp_path / caught[0]["repro"] / "model.onnx").exists()


def test_shrinker_finds_minimal_trigger():
    """A bug that only fires for K > 16 should minimize to K=17 with M=N=1."""
    def k_bug(case, y):
        return y + 1.0 if case.params["K"] > 16 else y

    case = Case("matmul", "float32", "none", "normal", {"batch": [], "M": 40, "K": 300, "N": 33, "b_batched": False}, 3)
    small = shrink(case, MutantBackend("mutant-k", k_bug), budget=80)
    assert small is not None
    assert small.params["K"] == 17 and small.params["M"] == 1 and small.params["N"] == 1


def test_clean_backend_passes(cfg, tmp_path):
    """Unmutated ort-cpu must pass the gate (with the repo's known-issue waivers, exactly as CI applies them)."""
    cfg["fuzz"].update({"ops": ["matmul", "conv2d", "relu", "add"], "dtypes": ["float32"]})
    cfg["waivers"] = load_config(ROOT / "configs" / "ci.yaml")["waivers"]
    result = run_fuzz(cfg, tmp_path, log=lambda *_: None)
    assert not gate(result), [(r["op"], r["quant"], r["dist"], r["detail"][:120]) for r in gate(result)]
