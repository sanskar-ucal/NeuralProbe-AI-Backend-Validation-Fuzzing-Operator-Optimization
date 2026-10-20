"""Builds a single-operator ONNX model for a Case.

Quantized cases use the QDQ format consumed by QNN / ORT quantized kernels:
  activations: float -> QuantizeLinear -> DequantizeLinear -> op
  weights:     int8 initializer -> DequantizeLinear -> op   (static weights, as the HTP requires)
  output:      op -> QuantizeLinear -> DequantizeLinear -> float
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .case import NP_DTYPES, Case, Materialized, QParams, quantize

OPSET = 17
ONNX_TYPES = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}


def _qdq_inits(name: str, q: QParams) -> list:
    return [numpy_helper.from_array(q.scale, f"{name}_scale"), numpy_helper.from_array(q.zero_point, f"{name}_zp")]


def _axis_attr(q: QParams) -> dict:
    return {} if q.axis is None else {"axis": int(q.axis)}


def build_model(case: Case, mat: Materialized) -> onnx.ModelProto:
    spec = case.spec
    np_dtype = NP_DTYPES[case.dtype]
    elem = ONNX_TYPES[case.dtype]
    nodes, inits, graph_inputs, src = [], [], [], {}

    for t in spec.tensors(case.params):
        arr = mat.inputs[t.name]
        q = mat.qparams.get(t.name)
        if t.role == "input":
            graph_inputs.append(helper.make_tensor_value_info(t.name, elem, list(arr.shape)))
            if q is None:
                src[t.name] = t.name
            else:
                inits += _qdq_inits(t.name, q)
                nodes.append(helper.make_node("QuantizeLinear", [t.name, f"{t.name}_scale", f"{t.name}_zp"],
                                              [f"{t.name}_q"], **_axis_attr(q)))
                nodes.append(helper.make_node("DequantizeLinear", [f"{t.name}_q", f"{t.name}_scale", f"{t.name}_zp"],
                                              [f"{t.name}_dq"], **_axis_attr(q)))
                src[t.name] = f"{t.name}_dq"
        else:
            if q is None:
                inits.append(numpy_helper.from_array(arr.astype(np_dtype), t.name))
                src[t.name] = t.name
            else:
                inits.append(numpy_helper.from_array(quantize(arr, q), f"{t.name}_q"))
                inits += _qdq_inits(t.name, q)
                nodes.append(helper.make_node("DequantizeLinear", [f"{t.name}_q", f"{t.name}_scale", f"{t.name}_zp"],
                                              [f"{t.name}_dq"], **_axis_attr(q)))
                src[t.name] = f"{t.name}_dq"

    out_q = mat.qparams.get("__out__")
    raw_out = "Y_raw" if out_q is not None else "Y"
    op_nodes, op_inits = spec.onnx_nodes(case.params, src, raw_out, np_dtype)
    nodes += op_nodes
    inits += op_inits
    if out_q is not None:
        inits += _qdq_inits("Y", out_q)
        nodes.append(helper.make_node("QuantizeLinear", [raw_out, "Y_scale", "Y_zp"], ["Y_q"]))
        nodes.append(helper.make_node("DequantizeLinear", ["Y_q", "Y_scale", "Y_zp"], ["Y"]))

    out_shape = list(mat.golden.shape) if mat.golden is not None else None
    graph = helper.make_graph(nodes, f"neuralprobe_{case.op}", graph_inputs,
                              [helper.make_tensor_value_info("Y", elem, out_shape)], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="neuralprobe")
    model.ir_version = 8
    return model


def feeds(case: Case, mat: Materialized) -> dict[str, np.ndarray]:
    return {t.name: mat.inputs[t.name] for t in case.spec.tensors(case.params) if t.role == "input"}
