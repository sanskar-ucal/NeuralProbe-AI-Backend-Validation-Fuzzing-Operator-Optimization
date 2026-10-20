"""Inference backends under test.

  ort-cpu        ONNX Runtime CPU EP, full graph optimizations (fusions on)
  ort-cpu-noopt  ONNX Runtime CPU EP, graph optimizations disabled (differential test of the optimizer)
  ort-coreml     ONNX Runtime CoreML EP (Apple ANE/GPU), CPU fallback disabled
  ort-qnn-htp    ONNX Runtime QNN EP, Hexagon HTP backend (Snapdragon), CPU fallback disabled
  ort-qnn-cpu    ONNX Runtime QNN EP, QNN CPU reference backend
  native-ref     NeuralProbe naive C++ kernels
  native-opt     NeuralProbe optimized C++ kernels (ARM NEON when available)
  torch          PyTorch eager at the case dtype (benchmark baseline)

Accelerator EPs run with `session.disable_cpu_ep_fallback=1` so a partially-offloaded graph is reported as
UNSUPPORTED instead of silently passing on the CPU EP.
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch

from .case import TORCH_DTYPES, Case, Materialized
from .onnx_builder import build_model, feeds

ROOT = Path(__file__).resolve().parent.parent
Runner = Callable[[], np.ndarray]
ort.set_default_logger_severity(3)  # backend errors are captured per case; keep ORT's own log quiet


class UnsupportedError(Exception):
    """Backend cannot run this case (missing kernel, partition rejected, ...). Not a test failure."""


class Backend:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def supports(self, case: Case) -> tuple[bool, str]:
        return True, ""

    def prepare(self, case: Case, mat: Materialized) -> Runner:
        raise NotImplementedError

    def run(self, case: Case, mat: Materialized) -> np.ndarray:
        return self.prepare(case, mat)()

    def info(self) -> dict:
        return {"name": self.name}


# ----------------------------------------------------------------------------------- ONNX Runtime
_UNSUPPORTED_MARKERS = ("NOT_IMPLEMENTED", "Could not find an implementation", "not supported",
                        "fallback", "Unsupported", "unsupported", "is not in the supported list")


class OrtBackend(Backend):
    def __init__(self, name: str, provider: str, provider_options: dict | None = None,
                 optimize: bool = True, disable_cpu_fallback: bool = False, threads: int = 1,
                 session_config: dict | None = None):
        self.name = name
        self.provider = provider
        self.provider_options = provider_options or {}
        self.optimize = optimize
        self.disable_cpu_fallback = disable_cpu_fallback
        self.threads = threads
        self.session_config = session_config or {}

    def variant(self, **changes) -> "OrtBackend":
        kw = {"name": self.name, "provider": self.provider, "provider_options": dict(self.provider_options),
              "optimize": self.optimize, "disable_cpu_fallback": self.disable_cpu_fallback,
              "threads": self.threads, "session_config": dict(self.session_config)}
        extra_cfg = changes.pop("session_config", {})
        kw.update(changes)
        kw["session_config"].update(extra_cfg)
        return OrtBackend(**kw)

    def available(self):
        if self.provider not in ort.get_available_providers():
            return False, f"{self.provider} not in onnxruntime {ort.__version__} build"
        return True, ""

    def session_options(self, profile_dir: str | None = None) -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if self.optimize
                                       else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        if self.disable_cpu_fallback:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        for k, v in self.session_config.items():
            so.add_session_config_entry(k, v)
        if profile_dir:
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(profile_dir, "ort_profile")
        return so

    def create_session(self, model_bytes: bytes, profile_dir: str | None = None) -> ort.InferenceSession:
        providers = [(self.provider, self.provider_options)]
        if self.provider != "CPUExecutionProvider" and not self.disable_cpu_fallback:
            providers.append("CPUExecutionProvider")
        try:
            return ort.InferenceSession(model_bytes, self.session_options(profile_dir), providers=providers)
        except Exception as e:
            msg = str(e)
            if any(m in msg for m in _UNSUPPORTED_MARKERS):
                raise UnsupportedError(msg.strip().splitlines()[0][:300]) from e
            raise

    def prepare(self, case, mat):
        sess = self.create_session(build_model(case, mat).SerializeToString())
        f = feeds(case, mat)
        return lambda: sess.run(None, f)[0]

    def profile(self, case: Case, mat: Materialized, iters: int = 20) -> dict:
        """Per-kernel time breakdown from ORT's built-in profiler (kernel time vs. session overhead)."""
        with tempfile.TemporaryDirectory() as d:
            sess = self.create_session(build_model(case, mat).SerializeToString(), profile_dir=d)
            f = feeds(case, mat)
            for _ in range(iters):
                sess.run(None, f)
            events = json.loads(Path(sess.end_profiling()).read_text())
        runs = [e["dur"] for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"]
        kernels: dict[str, dict] = {}
        for e in events:
            if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                args = e.get("args", {})
                key = f"{args.get('op_name', '?')}@{args.get('provider', '?')}"
                k = kernels.setdefault(key, {"us": 0.0, "calls": 0})
                k["us"] += e["dur"]
                k["calls"] += 1
        n = max(1, len(runs))
        run_us = sum(runs[1:]) / max(1, len(runs) - 1) if len(runs) > 1 else (runs[0] if runs else 0.0)
        kernel_us = sum(k["us"] for k in kernels.values()) / n
        top = sorted(((name, v["us"] / n) for name, v in kernels.items()), key=lambda kv: -kv[1])[:5]
        return {
            "run_us": run_us,
            "kernel_us": kernel_us,
            "overhead_pct": 100.0 * max(0.0, run_us - kernel_us) / run_us if run_us else 0.0,
            "top_kernels": [{"kernel": k, "us": round(v, 2)} for k, v in top],
        }

    def info(self):
        return {"name": self.name, "provider": self.provider, "provider_options": self.provider_options,
                "optimize": self.optimize, "session_config": self.session_config, "onnxruntime": ort.__version__}


def _qnn_backend_path(kind: str) -> str:
    env = os.environ.get(f"NEURALPROBE_QNN_{kind.upper()}_PATH")
    if env:
        return env
    lib = f"Qnn{kind.capitalize()}"
    return f"{lib}.dll" if sys.platform == "win32" else f"lib{lib}.so"


# ------------------------------------------------------------------------------------------ native
_LIB = None


def native_lib_path() -> Path | None:
    env = os.environ.get("NEURALPROBE_NATIVE_LIB")
    if env:
        return Path(env)
    for pattern in ("build/libneuralprobe_kernels.*", "build/*/libneuralprobe_kernels.*", "build/neuralprobe_kernels.dll"):
        hits = sorted(glob.glob(str(ROOT / pattern)))
        if hits:
            return Path(hits[0])
    return None


def load_native(path: Path | None = None):
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    path = path or native_lib_path()
    if path is None or not path.exists():
        raise FileNotFoundError("native kernels not built: run scripts/build_native.sh (or set NEURALPROBE_NATIVE_LIB)")
    lib = ctypes.CDLL(str(path))
    P, I64, INT = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.np_has_neon.restype = INT
    lib.np_isa_name.restype = ctypes.c_char_p
    lib.np_matmul_f32_ref.argtypes = [P, P, P, I64, I64, I64]
    lib.np_matmul_f32_opt.argtypes = [P, P, P, I64, I64, I64, INT]
    dw = [P, P, P, P] + [I64] * 12
    lib.np_dwconv2d_f32_ref.argtypes = dw
    lib.np_dwconv2d_f32_opt.argtypes = dw + [INT]
    _LIB = lib
    return lib


def _ptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data


class NativeBackend(Backend):
    OPS = ("matmul", "depthwise_conv2d")

    def __init__(self, name: str, variant: str, threads: int = 1):
        self.name = name
        self.variant = variant
        self.threads = threads

    def available(self):
        try:
            load_native()
            return True, ""
        except (FileNotFoundError, OSError) as e:
            return False, str(e)

    def supports(self, case):
        if case.op not in self.OPS:
            return False, f"op {case.op} not implemented natively"
        if case.dtype != "float32" or case.quant != "none":
            return False, "native kernels are fp32 only"
        if case.op == "depthwise_conv2d" and case.params["dilation"] != [1, 1]:
            return False, "native dwconv supports dilation 1 only"
        return True, ""

    def prepare(self, case, mat):
        lib = load_native()
        p = case.params
        if case.op == "matmul":
            fn = lib.np_matmul_f32_opt if self.variant == "opt" else lib.np_matmul_f32_ref
            extra = (self.threads,) if self.variant == "opt" else ()
            A = np.ascontiguousarray(mat.inputs["A"], np.float32)
            B = np.ascontiguousarray(mat.inputs["B"], np.float32)
            M, K, N = p["M"], p["K"], p["N"]
            out_shape = (*p["batch"], M, N)
            if not p["b_batched"]:  # fold batch into M: one big GEMM
                A2 = A.reshape(-1, K)
                C = np.empty((A2.shape[0], N), np.float32)

                def run():
                    fn(_ptr(A2), _ptr(B), _ptr(C), A2.shape[0], N, K, *extra)
                    return C.reshape(out_shape)
                return run
            A3, B3 = A.reshape(-1, M, K), B.reshape(-1, K, N)
            C3 = np.empty((A3.shape[0], M, N), np.float32)

            def run_batched():
                for b in range(A3.shape[0]):
                    fn(_ptr(A3[b]), _ptr(B3[b]), _ptr(C3[b]), M, N, K, *extra)
                return C3.reshape(out_shape)
            return run_batched

        fn = lib.np_dwconv2d_f32_opt if self.variant == "opt" else lib.np_dwconv2d_f32_ref
        extra = (self.threads,) if self.variant == "opt" else ()
        x = np.ascontiguousarray(mat.inputs["X"], np.float32)
        w = np.ascontiguousarray(mat.inputs["W"], np.float32)
        b = np.ascontiguousarray(mat.inputs["B"], np.float32) if p["bias"] else None
        oh, ow = case.spec.out_hw(p)
        y = np.empty((p["N"], p["C"], oh, ow), np.float32)
        pt, pl, pb, pr = p["pads"]

        def run_dw():
            fn(_ptr(x), _ptr(w), _ptr(b), _ptr(y), p["N"], p["C"], p["H"], p["W"], p["KH"], p["KW"],
               p["stride"][0], p["stride"][1], pt, pl, pb, pr, *extra)
            return y
        return run_dw

    def info(self):
        try:
            isa = load_native().np_isa_name().decode()
        except Exception:
            isa = "unavailable"
        return {"name": self.name, "variant": self.variant, "isa": isa, "threads": self.threads}


# ------------------------------------------------------------------------------------------ torch
class TorchBackend(Backend):
    name = "torch"

    def __init__(self, threads: int = 1):
        self.threads = threads

    def supports(self, case):
        if case.quant != "none":
            return False, "torch eager backend runs float graphs only"
        return True, ""

    def prepare(self, case, mat):
        torch.set_num_threads(self.threads)
        dt = TORCH_DTYPES[case.dtype]
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dt) for k, v in mat.inputs.items()}
        spec, params = case.spec, case.params

        def run():
            with torch.inference_mode():
                return spec.torch_fn(t, params).numpy()
        return run

    def info(self):
        return {"name": self.name, "torch": torch.__version__, "threads": self.threads}


# ----------------------------------------------------------------------------------------- factory
BACKEND_NAMES = ("ort-cpu", "ort-cpu-noopt", "ort-coreml", "ort-qnn-htp", "ort-qnn-cpu", "native-ref",
                 "native-opt", "torch")


def make_backend(name: str, threads: int = 1) -> Backend:
    if name == "ort-cpu":
        return OrtBackend(name, "CPUExecutionProvider", threads=threads)
    if name == "ort-cpu-noopt":
        return OrtBackend(name, "CPUExecutionProvider", optimize=False, threads=threads)
    if name == "ort-coreml":
        return OrtBackend(name, "CoreMLExecutionProvider",
                          {"ModelFormat": "MLProgram", "MLComputeUnits": os.environ.get("NEURALPROBE_COREML_UNITS", "ALL")},
                          disable_cpu_fallback=True, threads=threads)
    if name == "ort-qnn-htp":
        return OrtBackend(name, "QNNExecutionProvider",
                          {"backend_path": _qnn_backend_path("htp"), "htp_performance_mode": "burst",
                           "enable_htp_fp16_precision": "1"},
                          disable_cpu_fallback=True, threads=threads)
    if name == "ort-qnn-cpu":
        return OrtBackend(name, "QNNExecutionProvider", {"backend_path": _qnn_backend_path("cpu")},
                          disable_cpu_fallback=True, threads=threads)
    if name == "native-ref":
        return NativeBackend(name, "ref", threads)
    if name == "native-opt":
        return NativeBackend(name, "opt", threads)
    if name == "torch":
        return TorchBackend(threads)
    raise KeyError(f"unknown backend '{name}', choose from {BACKEND_NAMES}")


def environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "ort_providers": ort.get_available_providers(),
        "cpu_count": os.cpu_count(),
    }
