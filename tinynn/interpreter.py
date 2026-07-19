"""NumPy interpreter - the reference we check everything else against.

run() just walks the graph in order and evaluates each node with plain numpy,
then hands back the output node's value.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from .graph import Graph, Node
from .ops import (
    ADD,
    CONST,
    FUSED_LINEAR_RELU,
    INPUT,
    LINEAR,
    MATMUL,
    MUL,
    OUTPUT,
    QUANTIZED_FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    SUB,
    TANH,
)

Inputs = Union[Dict[str, np.ndarray], np.ndarray]


def _quantize_symmetric(v: np.ndarray):
    """int8 quantize v symmetrically. Returns (int64 levels in [-127,127], scale).

    scale is max(abs(v))/127, or 1.0 if v is all zeros so we don't divide by 0.
    Rounding has to match C++ std::lround (round half away from zero), so I do it
    by hand - np.round rounds half to even and would disagree with the C++.
    """
    maxabs = float(np.max(np.abs(v)))
    scale = maxabs / 127.0 if maxabs > 0.0 else 1.0
    scaled = v / scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)  # half away from zero
    clipped = np.clip(rounded, -127, 127)
    q = clipped.astype(np.int64)
    return q, scale


def _eval_quantized_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # cast to int64 before the matmul so the accumulation is real integer math,
    # exactly like the `long long acc` in the generated C++. only the final
    # rescale and bias add are floating point.
    x_q, s_x = _quantize_symmetric(x)
    w_q, s_w = _quantize_symmetric(weight)
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64)
    return acc.astype(np.float64) * (s_x * s_w) + bias


def run(graph: Graph, inputs: Inputs) -> np.ndarray:
    """Run the graph and return the output value.

    inputs is either a {name: array} dict, or a bare array if there's only one
    Input node.
    """
    provided = _resolve_inputs(graph, inputs)

    values: Dict[str, np.ndarray] = {}
    for node in graph.nodes:
        if node.op == INPUT:
            values[node.name] = _eval_input(node, provided)
        elif node.op == CONST:
            values[node.name] = node.weight
        elif node.op == LINEAR:
            x = _single_input_value(node, values)
            values[node.name] = x @ node.weight + node.bias
        elif node.op == RELU:
            x = _single_input_value(node, values)
            values[node.name] = np.maximum(x, 0.0)
        elif node.op == FUSED_LINEAR_RELU:
            x = _single_input_value(node, values)
            values[node.name] = np.maximum(x @ node.weight + node.bias, 0.0)
        elif node.op == QUANTIZED_LINEAR:
            x = _single_input_value(node, values)
            values[node.name] = _eval_quantized_linear(x, node.weight, node.bias)
        elif node.op == QUANTIZED_FUSED_LINEAR_RELU:
            # same as QuantizedLinear but max(0, ...) on the end
            x = _single_input_value(node, values)
            values[node.name] = np.maximum(
                _eval_quantized_linear(x, node.weight, node.bias), 0.0
            )
        elif node.op == OUTPUT:
            values[node.name] = _single_input_value(node, values)
        elif node.op == ADD:
            a, b = _two_input_values(node, values)
            values[node.name] = a + b
        elif node.op == SUB:
            a, b = _two_input_values(node, values)
            values[node.name] = a - b
        elif node.op == MUL:
            a, b = _two_input_values(node, values)
            values[node.name] = a * b
        elif node.op == MATMUL:
            a, b = _two_input_values(node, values)
            values[node.name] = a @ b
        elif node.op == SOFTMAX:
            x = _single_input_value(node, values)
            e = np.exp(x - np.max(x))
            values[node.name] = e / np.sum(e)
        elif node.op == TANH:
            x = _single_input_value(node, values)
            values[node.name] = np.tanh(x)
        elif node.op == SIGMOID:
            x = _single_input_value(node, values)
            values[node.name] = 1.0 / (1.0 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported op {node.op!r} on node {node.name!r}")

    if graph.output_node not in values:
        raise KeyError(f"Graph output node {graph.output_node!r} was never computed")

    return values[graph.output_node]


def _resolve_inputs(graph: Graph, inputs: Inputs) -> Dict[str, np.ndarray]:
    # turn a bare array into a {name: array} dict if we can
    if isinstance(inputs, dict):
        return inputs

    input_nodes = graph.input_nodes()
    if len(input_nodes) != 1:
        raise ValueError(
            "A bare array may only be passed as `inputs` when the graph has "
            f"exactly one Input node; this graph has {len(input_nodes)} "
            f"({[n.name for n in input_nodes]})"
        )
    return {input_nodes[0].name: inputs}


def _eval_input(node: Node, provided: Dict[str, np.ndarray]) -> np.ndarray:
    if node.name not in provided:
        raise KeyError(f"Missing value for Input node {node.name!r}")

    value = np.asarray(provided[node.name], dtype=np.float64)
    if value.shape != node.shape:
        raise ValueError(
            f"Input node {node.name!r} expected shape {node.shape}, got {value.shape}"
        )
    return value


def _single_input_value(node: Node, values: Dict[str, np.ndarray]) -> np.ndarray:
    input_name = node.inputs[0]
    if input_name not in values:
        raise KeyError(
            f"Node {node.name!r} references input {input_name!r} before it was computed"
        )
    return values[input_name]


def _two_input_values(node: Node, values: Dict[str, np.ndarray]):
    a_name, b_name = node.inputs[0], node.inputs[1]
    if a_name not in values:
        raise KeyError(
            f"Node {node.name!r} references input {a_name!r} before it was computed"
        )
    if b_name not in values:
        raise KeyError(
            f"Node {node.name!r} references input {b_name!r} before it was computed"
        )
    return values[a_name], values[b_name]
