"""Test cases: deterministic sampling, input materialization, quantization and the fp64 oracle."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .ops import OpSpec, get_op

NP_DTYPES = {"float32": np.float32, "float16": np.float16}
TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16}
QUANT_MODES = ("none", "int8_qdq", "int8_qdq_per_channel")


@dataclass
class Case:
    op: str
    dtype: str
    quant: str
    dist: str
    params: dict
    seed: int
    edge: bool = False

    @property
    def case_id(self) -> str:
        blob = json.dumps([self.op, self.dtype, self.quant, self.dist, self.params, self.seed], sort_keys=True)
        return f"{self.op}-{hashlib.sha1(blob.encode()).hexdigest()[:10]}"

    @property
    def spec(self) -> OpSpec:
        return get_op(self.op)

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(**{k: d[k] for k in ("op", "dtype", "quant", "dist", "params", "seed", "edge")})

    def with_params(self, params: dict) -> "Case":
        return Case(self.op, self.dtype, self.quant, self.dist, params, self.seed, self.edge)

    def describe(self) -> str:
        q = "" if self.quant == "none" else f" {self.quant}"
        return f"{self.op} [{self.spec.label(self.params)}] {self.dtype}{q} dist={self.dist}"


@dataclass
class QParams:
    scale: np.ndarray  # float32, scalar or 1-D
    zero_point: np.ndarray  # uint8 / int8, same shape as scale
    axis: int | None = None


@dataclass
class Materialized:
    inputs: dict[str, np.ndarray]  # float tensors in case dtype, before quantization
    qparams: dict[str, QParams] = field(default_factory=dict)  # per quantized tensor, plus "__out__"
    golden: np.ndarray | None = None  # fp64 oracle
    absref: np.ndarray | None = None  # OpSpec.magnitude() in fp64: scale of rounding error per element
    torch_same: np.ndarray | None = None  # PyTorch at the case dtype (what a user of PyTorch would get)
    torch_same_error: str | None = None

    @property
    def out_scale(self) -> float:
        q = self.qparams.get("__out__")
        return float(np.max(q.scale)) if q is not None else 0.0


# --------------------------------------------------------------------------------------- sampling
def sample_case(op_name: str, rng: np.random.Generator, dtypes, quant_modes, edge_prob: float) -> Case:
    spec = get_op(op_name)
    edge = bool(rng.random() < edge_prob)
    dtype = str(rng.choice(dtypes))
    quant = "none"
    if spec.quantizable and dtype == "float32":
        q_choices = [q for q in quant_modes if q in QUANT_MODES]
        if q_choices:
            quant = str(rng.choice(q_choices))
    dists = spec.dists if quant == "none" else ("normal", "uniform", "large", "small")
    if edge:
        dist = str(rng.choice(dists))
    else:
        dist = str(rng.choice(["normal", "uniform"]))
    for _ in range(100):
        params = spec.sample(rng, edge, quant)
        if spec.valid(params):
            break
    else:
        raise RuntimeError(f"could not sample valid params for {op_name}")
    return Case(op_name, dtype, quant, dist, params, int(rng.integers(0, 2**31 - 1)), edge)


def gen_tensor(rng: np.random.Generator, shape, dist: str, dtype: str, large: float) -> np.ndarray:
    np_dtype = NP_DTYPES[dtype]
    if dist == "normal":
        x = rng.standard_normal(shape)
    elif dist == "uniform":
        x = rng.uniform(-1.0, 1.0, shape)
    elif dist == "large":
        x = rng.standard_normal(shape) * large
    elif dist == "small":
        x = rng.standard_normal(shape) * 1e-3
    elif dist == "offset":  # large common-mode offset: stresses E[x^2]-E[x]^2 style variance and max-subtraction
        x = large + rng.standard_normal(shape)
    elif dist == "constant":
        x = np.full(shape, float(rng.uniform(-3, 3)))
    elif dist == "zeros":
        x = np.zeros(shape)
    elif dist == "denormal":
        x = rng.standard_normal(shape) * float(np.finfo(np_dtype).smallest_normal) * 0.5
    elif dist == "special":
        x = rng.standard_normal(shape)
        flat = x.reshape(-1)
        if flat.size:
            k = int(rng.integers(1, min(3, flat.size) + 1))
            idx = rng.choice(flat.size, size=k, replace=False)
            flat[idx] = rng.choice([np.nan, np.inf, -np.inf], size=k)
    else:
        raise ValueError(f"unknown dist {dist}")
    return np.asarray(x).astype(np_dtype)


# ----------------------------------------------------------------------------------- quantization
def _round_half_even_f32(x: np.ndarray) -> np.ndarray:
    return np.rint(x.astype(np.float32))


def act_qparams(x: np.ndarray) -> QParams:
    """Asymmetric uint8, per tensor (typical for activations on QNN HTP)."""
    finite = x[np.isfinite(x)]
    lo = min(float(finite.min()) if finite.size else 0.0, 0.0)
    hi = max(float(finite.max()) if finite.size else 0.0, 0.0)
    scale = np.float32((hi - lo) / 255.0) if hi > lo else np.float32(1.0)
    zp = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
    return QParams(np.asarray(scale, np.float32), np.asarray(zp, np.uint8))


def weight_qparams(w: np.ndarray, axis: int | None) -> QParams:
    """Symmetric int8, per tensor or per channel along `axis`."""
    if axis is None:
        m = float(np.max(np.abs(w))) if w.size else 0.0
        scale = np.asarray(m / 127.0 if m > 0 else 1.0, np.float32)
        return QParams(scale, np.zeros((), np.int8))
    ax = axis % w.ndim
    m = np.max(np.abs(w), axis=tuple(i for i in range(w.ndim) if i != ax))
    scale = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    return QParams(scale, np.zeros(scale.shape, np.int8), ax)


def _bcast(q: QParams, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    if q.axis is None:
        return q.scale, q.zero_point
    shape = [1] * ndim
    shape[q.axis] = -1
    return q.scale.reshape(shape), q.zero_point.reshape(shape)


def quantize(x: np.ndarray, q: QParams) -> np.ndarray:
    s, zp = _bcast(q, x.ndim)
    info = np.iinfo(q.zero_point.dtype)
    v = _round_half_even_f32(x.astype(np.float32) / s) + zp.astype(np.float32)
    return np.clip(v, info.min, info.max).astype(q.zero_point.dtype)


def dequantize(qx: np.ndarray, q: QParams) -> np.ndarray:
    s, zp = _bcast(q, qx.ndim)
    return ((qx.astype(np.int32) - zp.astype(np.int32)).astype(np.float32) * s).astype(np.float32)


def fake_quant(x: np.ndarray, q: QParams) -> np.ndarray:
    return dequantize(quantize(x, q), q)


# ------------------------------------------------------------------------------------ materialize
def _run_torch(spec: OpSpec, arrays: dict[str, np.ndarray], params: dict, dtype: torch.dtype) -> np.ndarray:
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dtype) for k, v in arrays.items()}
    with torch.no_grad():
        return spec.torch_fn(t, params).detach().cpu().numpy()


def materialize(case: Case, with_reference: bool = True) -> Materialized:
    spec = case.spec
    rng = np.random.default_rng(case.seed)
    inputs: dict[str, np.ndarray] = {}
    for i, t in enumerate(spec.tensors(case.params)):
        dist = case.dist if i == 0 or (t.role == "input" and case.dist == "special") else "normal"
        inputs[t.name] = gen_tensor(rng, t.shape, dist, case.dtype, spec.large_scale(case.dtype))
    mat = Materialized(inputs)

    effective = dict(inputs)
    if case.quant != "none":
        axis = spec.weight_quant_axis(case.params) if case.quant == "int8_qdq_per_channel" else None
        for t in spec.tensors(case.params):
            x = inputs[t.name]
            q = act_qparams(x) if t.role == "input" else weight_qparams(x, axis)
            mat.qparams[t.name] = q
            effective[t.name] = fake_quant(x, q)

    if not with_reference:
        return mat

    golden = _run_torch(spec, effective, case.params, torch.float64)
    try:
        same = _run_torch(spec, effective, case.params, TORCH_DTYPES[case.dtype]).astype(np.float64)
    except Exception as e:  # e.g. op not implemented for Half on CPU
        same, mat.torch_same_error = None, f"{type(e).__name__}: {e}"

    if case.quant != "none":
        out_q = act_qparams(golden.astype(np.float32))
        mat.qparams["__out__"] = out_q
        golden = fake_quant(golden.astype(np.float32), out_q).astype(np.float64)
        if same is not None:
            same = fake_quant(same.astype(np.float32), out_q).astype(np.float64)

    mat.golden = golden
    mat.torch_same = same
    t64 = {k: torch.from_numpy(np.ascontiguousarray(v)).to(torch.float64) for k, v in effective.items()}
    with torch.no_grad():
        mag = spec.magnitude(t64, case.params)
    mat.absref = None if mag is None else mag.numpy()
    return mat
