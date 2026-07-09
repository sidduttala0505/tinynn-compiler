"""Reference NumPy interpreter for the TinyNN Graph IR.

``run`` walks a :class:`tinynn.graph.Graph` in its stored (topological) order,
evaluating each node with plain NumPy, and returns the value produced by the
graph's declared output node. This module owns no IR types of its own -- it
consumes the shared ``Node``/``Graph`` dataclasses from :mod:`tinynn.graph`.
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
    QUANTIZED_LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    SUB,
    TANH,
)

Inputs = Union[Dict[str, np.ndarray], np.ndarray]


def _quantize_symmetric(v: np.ndarray):
    """Symmetric int8 quantization of ``v``, shared by weight and activation.

    Returns ``(q, scale)`` where ``q`` is an ``int64`` array holding the
    quantized (integer-valued) levels in ``[-127, 127]`` and ``scale`` is the
    float scale such that ``q * scale`` approximates ``v``.

    ``scale = max(abs(v)) / 127.0``, or ``1.0`` if ``v`` is all zeros (avoids
    division by zero; any scale works when every quantized value is 0).

    Rounding MUST match C++ ``std::lround`` (round half away from zero), NOT
    NumPy's ``np.round`` (round half to even) -- so we round manually via
    ``sign(v) * floor(abs(v) + 0.5)`` instead of calling ``np.round``.
    """
    maxabs = float(np.max(np.abs(v)))
    scale = maxabs / 127.0 if maxabs > 0.0 else 1.0
    scaled = v / scale
    # Round half away from zero (matches std::lround), not np.round's
    # round-half-to-even.
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    clipped = np.clip(rounded, -127, 127)
    q = clipped.astype(np.int64)
    return q, scale


def _eval_quantized_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation of the ``QuantizedLinear`` op.

    Weight quantization uses the node's stored float weight (deterministic,
    independent of any particular call); activation quantization is computed
    fresh from ``x`` on every call. Both quantized arrays are cast to
    ``int64`` BEFORE the matmul so the accumulation itself is pure integer
    arithmetic, matching the C++ side's ``long long acc`` accumulator
    exactly; only the final rescale (times scale product) and bias add use
    floating point.
    """
    x_q, s_x = _quantize_symmetric(x)
    w_q, s_w = _quantize_symmetric(weight)
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64)
    return acc.astype(np.float64) * (s_x * s_w) + bias


def run(graph: Graph, inputs: Inputs) -> np.ndarray:
    """Evaluate ``graph`` on ``inputs`` and return the output node's value.

    ``inputs`` may be a ``dict`` mapping Input-node names to arrays, or a bare
    array/array-like -- but a bare value is only accepted when ``graph`` has
    exactly one Input node.
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
    """Normalize ``inputs`` to a ``name -> array`` dict."""
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
