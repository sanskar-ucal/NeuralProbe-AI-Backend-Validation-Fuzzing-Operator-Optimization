import json
import subprocess
import sys

import numpy as np
import pytest

from neuralprobe.backends import make_backend
from neuralprobe.case import Case, act_qparams, dequantize, fake_quant, materialize, quantize, sample_case, weight_qparams
from neuralprobe.ops import REGISTRY
from neuralprobe.profiler import compare_bench
from neuralprobe.repro import evaluate, replay, save_bundle
from neuralprobe.runner import match_waiver, run_fuzz

from .conftest import MutantBackend, needs_native


@pytest.mark.parametrize("op", sorted(REGISTRY))
def test_every_op_matches_ort_cpu(op):
    rng = np.random.default_rng(11)
    backend = make_backend("ort-cpu")
    for _ in range(6):
        case = sample_case(op, rng, ["float32"], ["none"], edge_prob=0.0)
        status, comparison, _, error, _ = evaluate(case, backend)
        assert status in ("PASS", "WARN"), (case.describe(), error, comparison and comparison.detail)


def test_sampling_is_deterministic():
    a = [sample_case("conv2d", np.random.default_rng(5), ["float32", "float16"], ["none", "int8_qdq"], 0.5)
         for _ in range(3)]
    b = [sample_case("conv2d", np.random.default_rng(5), ["float32", "float16"], ["none", "int8_qdq"], 0.5)
         for _ in range(3)]
    assert [c.case_id for c in a] == [c.case_id for c in b]
    m1, m2 = materialize(a[0]), materialize(b[0])
    assert all(np.array_equal(m1.inputs[k], m2.inputs[k], equal_nan=True) for k in m1.inputs)


def test_quantization_roundtrip():
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    q = act_qparams(x)
    assert np.max(np.abs(fake_quant(x, q) - x)) <= q.scale * 0.5 + 1e-7
    w = np.random.default_rng(1).standard_normal((8, 3, 3, 3)).astype(np.float32)
    wq = weight_qparams(w, axis=0)
    assert wq.scale.shape == (8,)
    assert np.max(np.abs(dequantize(quantize(w, wq), wq) - w)) <= wq.scale.max() * 0.5 + 1e-7


@pytest.mark.parametrize("quant", ["int8_qdq", "int8_qdq_per_channel"])
def test_qdq_conv_matches_fake_quant_reference(quant):
    """Our fake-quant oracle must match ONNX QDQ semantics exactly. Uses the unfused graph (ort-cpu-noopt): the
    fused int8 kernels are ISA-dependent on x86 (U8S8 saturation, waiver NP-005) and are covered by fuzzing."""
    p = {"N": 1, "groups": 1, "cin_per_group": 8, "cout_per_group": 16, "KH": 3, "KW": 3, "stride": [1, 1],
         "dilation": [1, 1], "pads": [1, 1, 1, 1], "bias": False, "H": 12, "W": 12}
    case = Case("conv2d", "float32", quant, "normal", p, 9)
    status, comparison, _, error, _ = evaluate(case, make_backend("ort-cpu-noopt"))
    assert status == "PASS", (error, comparison and comparison.detail)
    assert comparison.max_ulp <= 1.01  # within one output LSB


@needs_native
@pytest.mark.parametrize("op", ["matmul", "depthwise_conv2d"])
@pytest.mark.parametrize("backend", ["native-opt", "native-ref"])
def test_native_kernels_fuzz_clean(cfg, tmp_path, op, backend):
    cfg["fuzz"].update({"ops": [op], "backends": [backend], "cases_per_op": 40, "dtypes": ["float32"],
                        "quant": ["none"], "edge_prob": 0.6})
    result = run_fuzz(cfg, tmp_path, log=lambda *_: None)
    bad = [r for r in result["rows"] if r["status"] == "FAIL"]
    assert not bad, bad[:3]
    assert sum(r["status"] == "PASS" for r in result["rows"]) > 10


def test_repro_bundle_is_standalone(tmp_path):
    case = Case("relu", "float32", "none", "normal", {"shape": [3, 5]}, 1)
    mutant = MutantBackend("mutant", lambda c, y: y * np.float32(1.01))
    status, comparison, actual, error, mat = evaluate(case, mutant)
    assert status == "FAIL"
    bundle = save_bundle(tmp_path, case, mat, mutant, comparison, actual)
    meta = json.loads((bundle / "case.json").read_text())
    assert meta["case"]["case_id"] == case.case_id
    for f in ("model.onnx", "inputs.npz", "golden.npy", "actual.npy", "repro.py"):
        assert (bundle / f).exists()
    # The standalone script runs the *real* ort-cpu (no mutation) -> it should exit 1 only if ORT is buggy.
    proc = subprocess.run([sys.executable, str(bundle / "repro.py")], capture_output=True, text=True)
    assert "max abs error" in proc.stdout, proc.stderr
    status, _, _ = replay(bundle, make_backend("ort-cpu"))
    assert status in ("PASS", "WARN")


def test_waiver_matching():
    row = {"op": "softmax", "backend": "ort-coreml", "dtype": "float16", "quant": "none", "dist": "normal",
           "category": "numerical_mismatch", "label": "axis=-3,shape=[1, 1, 2]"}
    w = [{"op": "softmax", "backend": "ort-*", "label_regex": r"^axis=-3,shape=\[\d+, \d+, \d+\]$", "reason": "x"}]
    assert match_waiver(row, w)
    assert not match_waiver(dict(row, label="axis=-3,shape=[1, 1, 2, 2]"), w)
    assert not match_waiver(dict(row, op="layernorm"), w)
    scoped = [{"op": ["matmul", "conv2d"], "machine": ["x86_64", "AMD64"], "reason": "x"}]
    assert match_waiver(dict(row, op="conv2d"), scoped, machine="x86_64")
    assert not match_waiver(dict(row, op="conv2d"), scoped, machine="aarch64")
    assert not match_waiver(row, scoped, machine="x86_64")


def test_perf_regression_detection():
    def bench(us):
        return {"results": [{"key": k, "backends": {"native-opt": {"median_us": v}}} for k, v in us.items()]}
    base = bench({"a": 100.0, "b": 100.0, "c": 2.0, "d": 100.0})
    head = bench({"a": 130.0, "b": 105.0, "c": 3.0, "e": 50.0})
    rows = {r["key"]: r["status"] for r in compare_bench(base, head, threshold=0.15, min_delta_us=5.0)}
    assert rows == {"a": "REGRESSION", "b": "OK", "c": "OK", "e": "NEW", "d": "MISSING"}
