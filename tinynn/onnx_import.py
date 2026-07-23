"""Load a small ONNX model into a TinyNN graph. Optional - needs `pip install onnx`.

Only handles simple MLP-ish models with 1D activations. onnx is imported lazily
inside import_onnx so the rest of TinyNN doesn't depend on it.

What maps to what:
  Gemm     -> Linear   (weight/bias are initializers; transB=1 transposes;
                        alpha/beta must be 1, transA must be 0)
  MatMul   -> Linear (zero bias) if the 2nd arg is a 2D initializer,
              or MatMul if both args are 2D activations
  Relu/Tanh/Sigmoid/Softmax/Add/Sub/Mul -> the obvious thing
  Identity -> just an alias, no node

A (1, N) input gets its batch dim of 1 squeezed to (N,). Anything weirder
(symbolic dims, batch != 1, rank > 2, unsupported ops, multiple outputs) raises.
TinyNN node names are just the ONNX output tensor names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .graph import Graph, GraphBuilder


def import_onnx(model: Any) -> Graph:
    """Turn an ONNX model (a path or a loaded ModelProto) into a TinyNN Graph."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - only hit without onnx
        raise ImportError(
            "The 'onnx' package is required for tinynn.onnx_import.import_onnx; "
            "install it with `pip install onnx`."
        ) from exc

    if isinstance(model, (str, Path)):
        model = onnx.load(str(model))
    if not isinstance(model, onnx.ModelProto):
        raise TypeError(
            f"import_onnx expects a path or onnx.ModelProto, got {type(model).__name__}"
        )

    onnx_graph = model.graph

    # the weights/biases (name -> array). these aren't graph inputs.
    initializers: Dict[str, np.ndarray] = {
        init.name: np.asarray(numpy_helper.to_array(init), dtype=np.float64)
        for init in onnx_graph.initializer
    }

    builder = GraphBuilder()
    tensor_map: Dict[str, str] = {}  # onnx tensor name -> our node name (Identity aliases)
    shapes: Dict[str, Tuple[int, ...]] = {}  # onnx tensor name -> its shape

    for value_info in onnx_graph.input:
        if value_info.name in initializers:
            continue  # weights listed as graph inputs (older exporters)
        shape = _input_shape(value_info)
        builder.input(value_info.name, shape)
        tensor_map[value_info.name] = value_info.name
        shapes[value_info.name] = shape

    for node in onnx_graph.node:
        _import_node(node, builder, tensor_map, shapes, initializers)

    if len(onnx_graph.output) != 1:
        raise ValueError(
            "Only single-output ONNX models are supported, got "
            f"{len(onnx_graph.output)} outputs: "
            f"{[o.name for o in onnx_graph.output]}"
        )
    out_name = onnx_graph.output[0].name
    if out_name not in tensor_map:
        raise ValueError(
            f"ONNX graph output {out_name!r} was never produced by any node"
        )
    return builder.build(output_node=tensor_map[out_name])


def _input_shape(value_info: Any) -> Tuple[int, ...]:
    # pull out the shape, dropping a leading batch dim of 1 if there is one
    name = value_info.name
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.WhichOneof("value") != "dim_value":
            raise ValueError(
                f"Input tensor {name!r} has a symbolic/unknown dimension "
                f"({dim.dim_param!r}); only concrete integer dims are supported"
            )
        dims.append(int(dim.dim_value))

    if len(dims) == 1:
        return (dims[0],)
    if len(dims) == 2:
        if dims[0] != 1:
            raise ValueError(
                f"Input tensor {name!r} has shape {tuple(dims)}; only a leading "
                f"batch dim of exactly 1 can be squeezed (got batch {dims[0]})"
            )
        return (dims[1],)
    raise ValueError(
        f"Input tensor {name!r} has shape {tuple(dims)} (rank {len(dims)}); "
        f"only rank-1 shapes or rank-2 shapes with a leading batch dim of 1 "
        f"are supported"
    )


def _attrs(node: Any) -> Dict[str, Any]:
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _activation(
    tensor_name: str,
    node: Any,
    tensor_map: Dict[str, str],
    initializers: Dict[str, np.ndarray],
) -> str:
    # look up an operand that should be an activation, not a weight
    if tensor_name in initializers:
        raise ValueError(
            f"unsupported initializer usage: {node.op_type} node "
            f"(output {node.output[0]!r}) uses initializer {tensor_name!r} as "
            f"an operand; initializers may only be Gemm/MatMul weights/biases"
        )
    if tensor_name not in tensor_map:
        raise ValueError(
            f"{node.op_type} node (output {node.output[0]!r}) references "
            f"unknown tensor {tensor_name!r}"
        )
    return tensor_map[tensor_name]


def _import_node(
    node: Any,
    builder: GraphBuilder,
    tensor_map: Dict[str, str],
    shapes: Dict[str, Tuple[int, ...]],
    initializers: Dict[str, np.ndarray],
) -> None:
    op = node.op_type
    out = node.output[0]

    if op == "Identity":
        src = _activation(node.input[0], node, tensor_map, initializers)
        tensor_map[out] = src
        shapes[out] = shapes[node.input[0]]
        return

    if op == "Gemm":
        _import_gemm(node, builder, tensor_map, shapes, initializers)
    elif op == "MatMul":
        _import_matmul(node, builder, tensor_map, shapes, initializers)
    elif op in ("Relu", "Tanh", "Sigmoid"):
        src = _activation(node.input[0], node, tensor_map, initializers)
        method = {"Relu": builder.relu, "Tanh": builder.tanh, "Sigmoid": builder.sigmoid}[op]
        method(out, src)
        tensor_map[out] = out
        shapes[out] = shapes[node.input[0]]
    elif op == "Softmax":
        _import_softmax(node, builder, tensor_map, shapes, initializers)
    elif op in ("Add", "Sub", "Mul"):
        a = _activation(node.input[0], node, tensor_map, initializers)
        b = _activation(node.input[1], node, tensor_map, initializers)
        a_shape = shapes[node.input[0]]
        b_shape = shapes[node.input[1]]
        if a_shape != b_shape:
            raise ValueError(
                f"{op} node (output {out!r}) requires operands of equal shape, "
                f"got {a_shape} and {b_shape}"
            )
        method = {"Add": builder.add, "Sub": builder.sub, "Mul": builder.mul}[op]
        method(out, a, b)
        tensor_map[out] = out
        shapes[out] = a_shape
    else:
        raise ValueError(
            f"Unsupported ONNX op type {op!r} (node output {out!r}); supported "
            f"ops are: Gemm, MatMul, Relu, Tanh, Sigmoid, Softmax, Add, Sub, "
            f"Mul, Identity"
        )


def _import_gemm(
    node: Any,
    builder: GraphBuilder,
    tensor_map: Dict[str, str],
    shapes: Dict[str, Tuple[int, ...]],
    initializers: Dict[str, np.ndarray],
) -> None:
    out = node.output[0]
    attrs = _attrs(node)
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))

    if alpha != 1.0:
        raise ValueError(f"Gemm node (output {out!r}) has alpha={alpha}; only alpha=1.0 is supported")
    if beta != 1.0:
        raise ValueError(f"Gemm node (output {out!r}) has beta={beta}; only beta=1.0 is supported")
    if trans_a != 0:
        raise ValueError(f"Gemm node (output {out!r}) has transA={trans_a}; only transA=0 is supported")

    a_name, b_name = node.input[0], node.input[1]
    src = _activation(a_name, node, tensor_map, initializers)
    if b_name not in initializers:
        raise ValueError(
            f"Gemm node (output {out!r}) operand B ({b_name!r}) must be an "
            f"initializer (constant weight)"
        )
    weight = initializers[b_name]
    if weight.ndim != 2:
        raise ValueError(
            f"Gemm node (output {out!r}) weight {b_name!r} must be 2D, "
            f"got shape {weight.shape}"
        )
    if trans_b == 1:
        weight = weight.T  # (out, in) -> (in, out)
    in_features, out_features = int(weight.shape[0]), int(weight.shape[1])

    if len(node.input) >= 3 and node.input[2]:
        c_name = node.input[2]
        if c_name not in initializers:
            raise ValueError(
                f"Gemm node (output {out!r}) operand C ({c_name!r}) must be an "
                f"initializer (constant bias)"
            )
        bias = initializers[c_name]
        if bias.ndim != 1 or bias.shape[0] != out_features:
            raise ValueError(
                f"Gemm node (output {out!r}) bias {c_name!r} must be 1D of "
                f"length {out_features}, got shape {bias.shape}"
            )
    else:
        bias = np.zeros(out_features, dtype=np.float64)

    src_shape = shapes[a_name]
    if src_shape != (in_features,):
        raise ValueError(
            f"Gemm node (output {out!r}) expects a 1D activation of shape "
            f"({in_features},), got {src_shape} from tensor {a_name!r}"
        )

    builder.linear(out, src, weight, bias)
    tensor_map[out] = out
    shapes[out] = (out_features,)


def _import_matmul(
    node: Any,
    builder: GraphBuilder,
    tensor_map: Dict[str, str],
    shapes: Dict[str, Tuple[int, ...]],
    initializers: Dict[str, np.ndarray],
) -> None:
    out = node.output[0]
    a_name, b_name = node.input[0], node.input[1]

    if a_name in initializers:
        raise ValueError(
            f"unsupported initializer usage: MatMul node (output {out!r}) "
            f"has initializer {a_name!r} as its first operand; only "
            f"MatMul(activation, initializer) is supported"
        )

    if b_name in initializers:
        # Linear with zero bias.
        src = _activation(a_name, node, tensor_map, initializers)
        weight = initializers[b_name]
        if weight.ndim != 2:
            raise ValueError(
                f"MatMul node (output {out!r}) weight {b_name!r} must be 2D "
                f"(in, out), got shape {weight.shape}"
            )
        src_shape = shapes[a_name]
        if src_shape != (int(weight.shape[0]),):
            raise ValueError(
                f"MatMul node (output {out!r}) expects a 1D activation of "
                f"shape ({int(weight.shape[0])},), got {src_shape} from "
                f"tensor {a_name!r}"
            )
        bias = np.zeros(int(weight.shape[1]), dtype=np.float64)
        builder.linear(out, src, weight, bias)
        tensor_map[out] = out
        shapes[out] = (int(weight.shape[1]),)
        return

    # two activations - only 2D @ 2D
    a = _activation(a_name, node, tensor_map, initializers)
    b = _activation(b_name, node, tensor_map, initializers)
    a_shape, b_shape = shapes[a_name], shapes[b_name]
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(
            f"MatMul node (output {out!r}) of two activations requires both "
            f"to be 2D, got shapes {a_shape} and {b_shape}"
        )
    builder.matmul(out, a, b)
    tensor_map[out] = out
    shapes[out] = (a_shape[0], b_shape[1])


def _import_softmax(
    node: Any,
    builder: GraphBuilder,
    tensor_map: Dict[str, str],
    shapes: Dict[str, Tuple[int, ...]],
    initializers: Dict[str, np.ndarray],
) -> None:
    out = node.output[0]
    src = _activation(node.input[0], node, tensor_map, initializers)
    src_shape = shapes[node.input[0]]
    if len(src_shape) != 1:
        raise ValueError(
            f"Softmax node (output {out!r}) requires a 1D activation, "
            f"got shape {src_shape}"
        )
    attrs = _attrs(node)
    axis = int(attrs.get("axis", -1))
    if axis not in (-1, 0):
        raise ValueError(
            f"Softmax node (output {out!r}) has axis={axis}; only the last "
            f"axis (-1 or 0 for a 1D tensor) is supported"
        )
    builder.softmax(out, src)
    tensor_map[out] = out
    shapes[out] = src_shape
