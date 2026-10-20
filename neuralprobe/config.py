"""YAML configuration with defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "default",
    "seed": 0,
    "fuzz": {
        "cases_per_op": 30,
        "ops": ["matmul", "conv2d", "depthwise_conv2d", "softmax", "layernorm", "gelu", "relu", "add"],
        "dtypes": ["float32", "float16"],
        "quant": ["none", "int8_qdq", "int8_qdq_per_channel"],
        "edge_prob": 0.4,
        "backends": ["ort-cpu", "ort-cpu-noopt", "native-opt"],
        "shrink": True,
        "triage": True,
        "max_repros_per_signature": 3,
        "time_budget_s": 900,
    },
    "tolerances": {},
    "waivers": [],
    "bench": {
        "threads": 1,
        "warmup": 5,
        "iters": 30,
        "repeats": 3,
        "profile": True,
        "backends": ["torch", "ort-cpu", "native-ref", "native-opt"],
        "cases": [],
    },
    "regression": {"perf_threshold": 0.15, "min_delta_us": 5.0, "fail_on_perf": True},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        path = Path(path)
        cfg = _merge(cfg, yaml.safe_load(path.read_text()) or {})
        if isinstance(cfg.get("waivers"), str):
            wpath = (path.parent / cfg["waivers"]).resolve()
            cfg["waivers"] = (yaml.safe_load(wpath.read_text()) or {}).get("waivers", [])
    return cfg
