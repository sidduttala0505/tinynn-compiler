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
from .ops import INPUT, LINEAR, OUTPUT, RELU

Inputs = Union[Dict[str, np.ndarray], np.ndarray]


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
        elif node.op == LINEAR:
            x = _single_input_value(node, values)
            values[node.name] = x @ node.weight + node.bias
        elif node.op == RELU:
            x = _single_input_value(node, values)
            values[node.name] = np.maximum(x, 0.0)
        elif node.op == OUTPUT:
            values[node.name] = _single_input_value(node, values)
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
