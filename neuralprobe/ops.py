"""Operator registry.

Each OpSpec knows how to:
  * sample random (optionally edge-case biased) parameters,
  * describe its tensors (activations vs. weights),
  * compute a PyTorch reference,
  * emit the equivalent ONNX subgraph,
  * report FLOPs / reduction length (used for error bounds), and
  * propose smaller parameter sets for failure minimization (shrinking).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from onnx import helper, numpy_helper

# Sizes around common SIMD widths / tile edges (4, 8, 16 lanes; 8x8 micro-tiles; 256 K-blocks).
SIMD_EDGE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    role: str  # "input" (graph input / activation) or "weight" (initializer)


def _dim(rng: np.random.Generator, edge: bool, lo: int = 1, hi: int = 48) -> int:
    if edge and rng.random() < 0.6:
        cands = [s for s in SIMD_EDGE_SIZES if lo <= s <= max(hi, lo)]
        if cands:
            return int(rng.choice(cands))
    return int(rng.integers(lo, hi + 1))


def _shape(rng, edge, rank_lo=1, rank_hi=4, hi=24) -> list[int]:
    rank = int(rng.integers(rank_lo, rank_hi + 1))
    return [_dim(rng, edge, 1, hi) for _ in range(rank)]


def conv_out(n: int, k: int, s: int, d: int, p0: int, p1: int) -> int:
    span = n + p0 + p1 - d * (k - 1) - 1
    return span // s + 1 if span >= 0 else 0


class OpSpec:
    name: str = ""
    linear: bool = False  # multilinear in its inputs -> error bound scales with |x|*|w|
    quantizable: bool = False
    dists: tuple[str, ...] = ("normal", "uniform", "large", "small", "special", "denormal")
    shrink_keys: tuple[str, ...] = ()

    def sample(self, rng: np.random.Generator, edge: bool, quant: str) -> dict:
        raise NotImplementedError

    def valid(self, p: dict) -> bool:
        return True

    def tensors(self, p: dict) -> list[TensorSpec]:
        raise NotImplementedError

    def torch_fn(self, t: dict[str, torch.Tensor], p: dict) -> torch.Tensor:
        raise NotImplementedError

    def onnx_nodes(self, p: dict, src: dict[str, str], out: str, np_dtype) -> tuple[list, list]:
        raise NotImplementedError

    def reduction(self, p: dict) -> int:
        return 1

    def magnitude(self, t: dict[str, torch.Tensor], p: dict) -> torch.Tensor | None:
        """Per-element magnitude of the intermediate terms (fp64), scaling the rounding-error bound.
        For multilinear ops this is op(|inputs|); None means a plain rtol/atol bound."""
        if self.linear:
            return self.torch_fn({k: v.abs() for k, v in t.items()}, p)
        return None

    def flops(self, p: dict) -> float:
        return float(np.prod(self.tensors(p)[0].shape))

    def large_scale(self, dtype: str) -> float:
        return 1e3 if dtype == "float32" else 50.0

    def weight_quant_axis(self, p: dict) -> int | None:
        """Axis for per-channel weight quantization (None: op has no per-channel weight)."""
        return None

    def label(self, p: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(p.items()))

    def shrink(self, p: dict) -> Iterator[dict]:
        for key in self.shrink_keys:
            v = p.get(key)
            if isinstance(v, list):
                for i, d in enumerate(v):
                    for nd in _smaller(d):
                        q = dict(p)
                        q[key] = v[:i] + [nd] + v[i + 1:]
                        if self.valid(q):
                            yield q
            elif isinstance(v, int) and not isinstance(v, bool):
                for nv in _smaller(v):
                    q = dict(p, **{key: nv})
                    if self.valid(q):
                        yield q


def _smaller(v: int) -> list[int]:
    return sorted({c for c in (1, v // 2, v - 1) if 1 <= c < v})


def _const(name: str, value, np_dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=np_dtype), name)


# --------------------------------------------------------------------------------------------- MatMul
class MatMul(OpSpec):
    name = "matmul"
    linear = True
    quantizable = True
    shrink_keys = ("M", "K", "N", "batch")

    def sample(self, rng, edge, quant):
        batch = [] if rng.random() < 0.6 else _shape(rng, edge, 1, 2, hi=4)
        k_hi = 300 if rng.random() < 0.2 else 96
        return {
            "batch": batch,
            "M": _dim(rng, edge, 1, 96),
            "K": _dim(rng, edge, 1, k_hi),
            "N": _dim(rng, edge, 1, 96),
            "b_batched": bool(batch) and rng.random() < 0.4,
        }

    def valid(self, p):
        return min(p["M"], p["K"], p["N"]) >= 1 and all(b >= 1 for b in p["batch"])

    def tensors(self, p):
        b_shape = (*p["batch"], p["K"], p["N"]) if p["b_batched"] else (p["K"], p["N"])
        return [TensorSpec("A", (*p["batch"], p["M"], p["K"]), "input"), TensorSpec("B", b_shape, "weight")]

    def torch_fn(self, t, p):
        return torch.matmul(t["A"], t["B"])

    def onnx_nodes(self, p, src, out, np_dtype):
        return [helper.make_node("MatMul", [src["A"], src["B"]], [out])], []

    def reduction(self, p):
        return p["K"]

    def flops(self, p):
        return 2.0 * math.prod(p["batch"]) * p["M"] * p["N"] * p["K"]

    def large_scale(self, dtype):
        return 1e3 if dtype == "float32" else 8.0  # keep fp16 accumulations below 65504

    def weight_quant_axis(self, p):
        return -1  # per output column

    def label(self, p):
        b = "x".join(map(str, p["batch"])) + ("·" if p["batch"] else "")
        return f"{b}{p['M']}x{p['K']} @ {p['K']}x{p['N']}" + (" (batched B)" if p["b_batched"] else "")


# --------------------------------------------------------------------------------------------- Conv2D
class Conv2d(OpSpec):
    name = "conv2d"
    linear = True
    quantizable = True
    shrink_keys = ("N", "cin_per_group", "cout_per_group", "groups", "H", "W")

    def sample(self, rng, edge, quant):
        groups = int(rng.choice([1, 1, 1, 2, 3, 4]))
        kh = int(rng.choice([1, 3, 3, 5, 7])) if rng.random() < 0.8 else int(rng.integers(1, 6))
        kw = kh if rng.random() < 0.8 else int(rng.integers(1, 6))
        p = {
            "N": int(rng.choice([1, 1, 2, 3])),
            "groups": groups,
            "cin_per_group": _dim(rng, edge, 1, 12),
            "cout_per_group": _dim(rng, edge, 1, 12),
            "KH": kh,
            "KW": kw,
            "stride": [int(rng.choice([1, 1, 2, 3])), int(rng.choice([1, 1, 2, 3]))],
            "dilation": [1, 1] if rng.random() < 0.8 else [int(rng.integers(1, 3)), int(rng.integers(1, 3))],
            "bias": quant == "none" and rng.random() < 0.7,
        }
        if rng.random() < 0.7:
            p["pads"] = [kh // 2, kw // 2, kh // 2, kw // 2]
        else:
            p["pads"] = [int(rng.integers(0, kh)), int(rng.integers(0, kw)), int(rng.integers(0, kh)),
                         int(rng.integers(0, kw))]
        p["H"] = max(self._min_in(p, 0), _dim(rng, edge, 1, 28))
        p["W"] = max(self._min_in(p, 1), _dim(rng, edge, 1, 28))
        return p

    @staticmethod
    def _min_in(p, axis):
        k = p["KH"] if axis == 0 else p["KW"]
        pad = p["pads"][axis] + p["pads"][axis + 2]
        return max(1, p["dilation"][axis] * (k - 1) + 1 - pad)

    def out_hw(self, p):
        (pt, pl, pb, pr), (sh, sw), (dh, dw) = p["pads"], p["stride"], p["dilation"]
        return conv_out(p["H"], p["KH"], sh, dh, pt, pb), conv_out(p["W"], p["KW"], sw, dw, pl, pr)

    def valid(self, p):
        oh, ow = self.out_hw(p)
        return oh >= 1 and ow >= 1 and min(p["N"], p["groups"], p["cin_per_group"], p["cout_per_group"]) >= 1

    def tensors(self, p):
        cin, cout = p["groups"] * p["cin_per_group"], p["groups"] * p["cout_per_group"]
        ts = [TensorSpec("X", (p["N"], cin, p["H"], p["W"]), "input"),
              TensorSpec("W", (cout, p["cin_per_group"], p["KH"], p["KW"]), "weight")]
        if p["bias"]:
            ts.append(TensorSpec("B", (cout,), "weight"))
        return ts

    def torch_fn(self, t, p):
        pt, pl, pb, pr = p["pads"]
        x = F.pad(t["X"], (pl, pr, pt, pb))
        return F.conv2d(x, t["W"], t.get("B"), stride=tuple(p["stride"]), dilation=tuple(p["dilation"]),
                        groups=p["groups"])

    def onnx_nodes(self, p, src, out, np_dtype):
        ins = [src["X"], src["W"]] + ([src["B"]] if p["bias"] else [])
        node = helper.make_node("Conv", ins, [out], kernel_shape=[p["KH"], p["KW"]], strides=p["stride"],
                                dilations=p["dilation"], pads=p["pads"], group=p["groups"])
        return [node], []

    def reduction(self, p):
        return p["cin_per_group"] * p["KH"] * p["KW"]

    def flops(self, p):
        oh, ow = self.out_hw(p)
        return 2.0 * p["N"] * p["groups"] * p["cout_per_group"] * oh * ow * self.reduction(p)

    def large_scale(self, dtype):
        return 1e3 if dtype == "float32" else 8.0

    def weight_quant_axis(self, p):
        return 0

    def label(self, p):
        cin, cout = p["groups"] * p["cin_per_group"], p["groups"] * p["cout_per_group"]
        return (f"N{p['N']} {cin}->{cout} {p['H']}x{p['W']} k{p['KH']}x{p['KW']} s{p['stride']} "
                f"d{p['dilation']} p{p['pads']} g{p['groups']}")


# ------------------------------------------------------------------------------------ Depthwise Conv2D
class DepthwiseConv2d(Conv2d):
    name = "depthwise_conv2d"
    shrink_keys = ("N", "C", "H", "W")

    def sample(self, rng, edge, quant):
        kh, kw = [(3, 3), (3, 3), (3, 3), (5, 5), (1, 1), (7, 7), (3, 1), (1, 3)][int(rng.integers(0, 8))]
        if rng.random() < 0.85:
            stride = [int(rng.choice([1, 1, 2, 2, 3]))] * 2
        else:
            stride = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
        p = {
            "N": int(rng.choice([1, 1, 2])),
            "C": _dim(rng, edge, 1, 64),
            "KH": kh,
            "KW": kw,
            "stride": stride,
            "dilation": [1, 1] if rng.random() < 0.9 else [2, 2],
            "bias": quant == "none" and rng.random() < 0.7,
        }
        if rng.random() < 0.6:
            p["pads"] = [kh // 2, kw // 2, kh // 2, kw // 2]
        else:  # includes TF-style "SAME" asymmetric padding
            p["pads"] = [int(rng.integers(0, kh)), int(rng.integers(0, kw)), int(rng.integers(0, kh)),
                         int(rng.integers(0, kw))]
        p["H"] = max(self._min_in(p, 0), _dim(rng, edge, 1, 40))
        p["W"] = max(self._min_in(p, 1), _dim(rng, edge, 1, 40))
        return p

    def _as_conv(self, p):
        return dict(p, groups=p["C"], cin_per_group=1, cout_per_group=1)

    def valid(self, p):
        return p["C"] >= 1 and super().valid(self._as_conv(p))

    def tensors(self, p):
        ts = [TensorSpec("X", (p["N"], p["C"], p["H"], p["W"]), "input"),
              TensorSpec("W", (p["C"], 1, p["KH"], p["KW"]), "weight")]
        if p["bias"]:
            ts.append(TensorSpec("B", (p["C"],), "weight"))
        return ts

    def torch_fn(self, t, p):
        return super().torch_fn(t, self._as_conv(p))

    def onnx_nodes(self, p, src, out, np_dtype):
        return super().onnx_nodes(self._as_conv(p), src, out, np_dtype)

    def reduction(self, p):
        return p["KH"] * p["KW"]

    def flops(self, p):
        oh, ow = self.out_hw(p)
        return 2.0 * p["N"] * p["C"] * oh * ow * p["KH"] * p["KW"]

    def label(self, p):
        return (f"N{p['N']} C{p['C']} {p['H']}x{p['W']} k{p['KH']}x{p['KW']} s{p['stride']} "
                f"d{p['dilation']} p{p['pads']}")


# -------------------------------------------------------------------------------------------- Softmax
class Softmax(OpSpec):
    name = "softmax"
    dists = ("normal", "uniform", "large", "small", "offset", "constant", "special", "denormal")
    shrink_keys = ("shape",)

    def sample(self, rng, edge, quant):
        shape = _shape(rng, edge, 1, 4, hi=16)
        if rng.random() < 0.25:
            shape[-1] = int(rng.choice([1000, 1024, 4096, 32000]))  # classifier / vocab-sized rows
        return {"shape": shape, "axis": int(rng.integers(-len(shape), len(shape)))}

    def valid(self, p):
        return -len(p["shape"]) <= p["axis"] < len(p["shape"])

    def tensors(self, p):
        return [TensorSpec("X", tuple(p["shape"]), "input")]

    def torch_fn(self, t, p):
        return torch.softmax(t["X"], dim=p["axis"])

    def onnx_nodes(self, p, src, out, np_dtype):
        return [helper.make_node("Softmax", [src["X"]], [out], axis=p["axis"])], []

    def reduction(self, p):
        return p["shape"][p["axis"]]

    def magnitude(self, t, p):
        return self.torch_fn(t, p)

    def flops(self, p):
        return 5.0 * math.prod(p["shape"])

    def large_scale(self, dtype):
        return 1e4 if dtype == "float32" else 300.0


# ---------------------------------------------------------------------------------------- LayerNorm
class LayerNorm(OpSpec):
    name = "layernorm"
    dists = ("normal", "uniform", "large", "small", "offset", "constant", "special")
    shrink_keys = ("shape",)

    def sample(self, rng, edge, quant):
        shape = _shape(rng, edge, 2, 4, hi=24)
        if rng.random() < 0.3:
            shape[-1] = int(rng.choice([768, 1024, 4096]))  # transformer hidden sizes
        return {"shape": shape, "axis": int(rng.integers(1, len(shape))),
                "eps": float(rng.choice([1e-5, 1e-6, 1e-12]))}

    def valid(self, p):
        return 1 <= p["axis"] < len(p["shape"])

    def tensors(self, p):
        norm = tuple(p["shape"][p["axis"]:])
        return [TensorSpec("X", tuple(p["shape"]), "input"), TensorSpec("gamma", norm, "weight"),
                TensorSpec("beta", norm, "weight")]

    def torch_fn(self, t, p):
        return F.layer_norm(t["X"], tuple(p["shape"][p["axis"]:]), t["gamma"], t["beta"], eps=p["eps"])

    def onnx_nodes(self, p, src, out, np_dtype):
        node = helper.make_node("LayerNormalization", [src["X"], src["gamma"], src["beta"]], [out],
                                axis=p["axis"], epsilon=p["eps"])
        return [node], []

    def reduction(self, p):
        return math.prod(p["shape"][p["axis"]:])

    def magnitude(self, t, p):
        # Rounding in (x - mean) is ~eps*(|x|+|mean|), amplified by 1/sigma: inputs with a large common-mode
        # offset or near-zero variance are ill-conditioned for any fp32 algorithm, so scale the bound by it.
        x, dims = t["X"], tuple(range(p["axis"], len(p["shape"])))
        mean = x.mean(dim=dims, keepdim=True)
        sigma = torch.sqrt(x.var(dim=dims, unbiased=False, keepdim=True) + p["eps"])
        xhat = (x - mean) / sigma
        return t["gamma"].abs() * (xhat.abs() + 1.0 + (x.abs() + mean.abs()) / sigma)

    def flops(self, p):
        return 8.0 * math.prod(p["shape"])


# --------------------------------------------------------------------------------------------- GELU
class Gelu(OpSpec):
    """Exact (erf) GELU, emitted as the decomposed subgraph exporters produce; ORT fuses it into a Gelu kernel,
    so this also exercises the graph-fusion path."""

    name = "gelu"
    shrink_keys = ("shape",)

    def sample(self, rng, edge, quant):
        return {"shape": _shape(rng, edge, 1, 4, hi=32)}

    def tensors(self, p):
        return [TensorSpec("X", tuple(p["shape"]), "input")]

    def torch_fn(self, t, p):
        return F.gelu(t["X"])

    def onnx_nodes(self, p, src, out, np_dtype):
        inits = [_const("gelu_sqrt2", math.sqrt(2.0), np_dtype), _const("gelu_one", 1.0, np_dtype),
                 _const("gelu_half", 0.5, np_dtype)]
        nodes = [
            helper.make_node("Div", [src["X"], "gelu_sqrt2"], ["gelu_div"]),
            helper.make_node("Erf", ["gelu_div"], ["gelu_erf"]),
            helper.make_node("Add", ["gelu_erf", "gelu_one"], ["gelu_add"]),
            helper.make_node("Mul", [src["X"], "gelu_add"], ["gelu_mul"]),
            helper.make_node("Mul", ["gelu_mul", "gelu_half"], [out]),
        ]
        return nodes, inits

    def flops(self, p):
        return 10.0 * math.prod(p["shape"])

    def large_scale(self, dtype):
        return 30.0


# ---------------------------------------------------------------------------------------------- ReLU
class Relu(OpSpec):
    name = "relu"
    shrink_keys = ("shape",)

    def sample(self, rng, edge, quant):
        return {"shape": _shape(rng, edge, 1, 4, hi=32)}

    def tensors(self, p):
        return [TensorSpec("X", tuple(p["shape"]), "input")]

    def torch_fn(self, t, p):
        return torch.relu(t["X"])

    def onnx_nodes(self, p, src, out, np_dtype):
        return [helper.make_node("Relu", [src["X"]], [out])], []


# ------------------------------------------------------------------------------------- Broadcast Add
class Add(OpSpec):
    name = "add"
    shrink_keys = ("a_shape", "b_shape")

    def sample(self, rng, edge, quant):
        full = _shape(rng, edge, 1, 4, hi=20)
        k = int(rng.integers(1, len(full) + 1))
        other = [d if rng.random() < 0.6 else 1 for d in full[-k:]]
        a, b = (full, other) if rng.random() < 0.7 else (other, full)
        return {"a_shape": a, "b_shape": b}

    def valid(self, p):
        try:
            np.broadcast_shapes(tuple(p["a_shape"]), tuple(p["b_shape"]))
            return True
        except ValueError:
            return False

    def tensors(self, p):
        return [TensorSpec("A", tuple(p["a_shape"]), "input"), TensorSpec("B", tuple(p["b_shape"]), "input")]

    def torch_fn(self, t, p):
        return t["A"] + t["B"]

    def onnx_nodes(self, p, src, out, np_dtype):
        return [helper.make_node("Add", [src["A"], src["B"]], [out])], []

    def flops(self, p):
        return float(math.prod(np.broadcast_shapes(tuple(p["a_shape"]), tuple(p["b_shape"]))))


REGISTRY: dict[str, OpSpec] = {
    op.name: op for op in (MatMul(), Conv2d(), DepthwiseConv2d(), Softmax(), LayerNorm(), Gelu(), Relu(), Add())
}


def get_op(name: str) -> OpSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown op '{name}', available: {sorted(REGISTRY)}")
    return REGISTRY[name]
