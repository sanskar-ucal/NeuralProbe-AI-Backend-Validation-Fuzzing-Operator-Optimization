import numpy as np

from neuralprobe.case import Case, materialize
from neuralprobe.compare import compare

MATMUL = {"batch": [], "M": 9, "K": 300, "N": 17, "b_batched": False}


def _case(op="matmul", params=None, dtype="float32", quant="none", dist="normal"):
    return Case(op, dtype, quant, dist, params or MATMUL, seed=7)


def test_exact_golden_passes():
    case = _case()
    mat = materialize(case)
    assert compare(case, mat, mat.golden.astype(np.float32)).status == "PASS"


def test_pytorch_same_dtype_is_within_bounds():
    for dtype in ("float32", "float16"):
        case = _case(dtype=dtype)
        mat = materialize(case)
        assert compare(case, mat, mat.torch_same).status in ("PASS", "WARN")


def test_relative_perturbation_fails():
    case = _case()
    mat = materialize(case)
    out = mat.golden.astype(np.float32)
    out[3, 5] *= 1.001
    c = compare(case, mat, out)
    assert c.status == "FAIL" and c.category == "numerical_mismatch"
    assert c.worst_index == [3, 5] and c.n_mismatch == 1


def test_nan_pattern_mismatch_detected():
    case = _case(dist="special")
    mat = materialize(case)
    assert np.isnan(mat.golden).any() or np.isinf(mat.golden).any()
    out = np.nan_to_num(mat.golden.astype(np.float32), nan=0.0)
    if np.isnan(mat.golden).any():
        c = compare(case, mat, out)
        assert c.status == "FAIL" and c.category == "nonfinite_mismatch"


def test_shape_mismatch():
    case = _case()
    mat = materialize(case)
    assert compare(case, mat, mat.golden[:, :-1]).category == "shape_mismatch"


def test_quantized_one_lsb_ok_three_lsb_fail():
    case = _case(quant="int8_qdq")
    mat = materialize(case)
    lsb = mat.out_scale
    out = mat.golden.astype(np.float32)
    out[0, 0] += lsb
    assert compare(case, mat, out).status == "PASS"
    out[0, 0] += 2 * lsb
    assert compare(case, mat, out).status == "FAIL"


def test_large_k_rounding_not_flagged():
    """fp32 sequential accumulation over K=4096 has legit error >> rtol; the forward-error bound must allow it."""
    params = {"batch": [], "M": 4, "K": 4096, "N": 4, "b_batched": False}
    case = _case(params=params, dist="large")
    mat = materialize(case)
    A, B = mat.inputs["A"], mat.inputs["B"]
    naive = np.zeros((4, 4), np.float32)
    for k in range(4096):
        naive += np.outer(A[:, k], B[k, :]).astype(np.float32)
    assert compare(case, mat, naive).status in ("PASS", "WARN")
