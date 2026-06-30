"""Tiny NumPy reference interpreter for a neural network graph.

This file is intentionally self-contained and simple. The ``Node`` and
``Graph`` classes mirror the external IR shape that the compiler will consume
later, while ``run`` stays close to the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None


@dataclass
class Graph:
    nodes: list[Node]
    output_node: str


def _input_value(node: Node, values: dict[str, np.ndarray]) -> np.ndarray:
    if len(node.inputs) != 1:
        raise ValueError(f"{node.op} node {node.name!r} needs exactly one input")

    input_name = node.inputs[0]
    if input_name not in values:
        raise KeyError(f"Node {node.name!r} references missing input {input_name!r}")

    return values[input_name]


def _linear_params(node: Node) -> tuple[np.ndarray, np.ndarray]:
    if node.weight is None:
        raise ValueError(f"{node.op} node {node.name!r} is missing weight")
    if node.bias is None:
        raise ValueError(f"{node.op} node {node.name!r} is missing bias")
    return node.weight, node.bias


def run(graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    values: dict[str, np.ndarray] = {}

    for node in graph.nodes:
        if node.op == "Input":
            if node.name not in inputs:
                raise KeyError(f"Missing input for Input node {node.name!r}")
            values[node.name] = inputs[node.name]

        elif node.op == "Linear":
            x = _input_value(node, values)
            weight, bias = _linear_params(node)
            values[node.name] = x @ weight + bias

        elif node.op == "ReLU":
            x = _input_value(node, values)
            values[node.name] = np.maximum(x, 0)

        elif node.op == "FusedLinearReLU":
            x = _input_value(node, values)
            weight, bias = _linear_params(node)
            values[node.name] = np.maximum(x @ weight + bias, 0)

        else:
            raise ValueError(f"Unknown op {node.op!r} on node {node.name!r}")

    if graph.output_node not in values:
        raise KeyError(f"Graph output node {graph.output_node!r} was not computed")

    return values[graph.output_node]


def _test_linear_relu_linear() -> None:
    x = np.array([1.0, -2.0, 3.0])

    w0 = np.array(
        [
            [0.5, -1.0, 2.0, 0.0],
            [1.5, 0.25, -0.5, 1.0],
            [-1.0, 0.75, 0.5, -2.0],
        ]
    )
    b0 = np.array([0.1, -0.2, 0.3, 0.4])

    w1 = np.array(
        [
            [1.0, -0.5],
            [0.0, 2.0],
            [-1.5, 1.0],
            [0.25, 0.75],
        ]
    )
    b1 = np.array([-0.1, 0.2])

    graph = Graph(
        nodes=[
            Node(name="x", op="Input"),
            Node(name="linear_0", op="Linear", inputs=["x"], weight=w0, bias=b0),
            Node(name="relu_0", op="ReLU", inputs=["linear_0"]),
            Node(name="linear_1", op="Linear", inputs=["relu_0"], weight=w1, bias=b1),
        ],
        output_node="linear_1",
    )

    actual = run(graph, {"x": x})
    expected = np.maximum(x @ w0 + b0, 0) @ w1 + b1

    assert np.allclose(actual, expected)


def _test_fused_linear_relu() -> None:
    x = np.array([-1.0, 0.5, 2.0])
    weight = np.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
            [-0.25, 1.5],
        ]
    )
    bias = np.array([0.75, -2.0])

    graph = Graph(
        nodes=[
            Node(name="x", op="Input"),
            Node(name="fused_0", op="FusedLinearReLU", inputs=["x"], weight=weight, bias=bias),
        ],
        output_node="fused_0",
    )

    actual = run(graph, {"x": x})
    expected = np.maximum(x @ weight + bias, 0)

    assert np.allclose(actual, expected)


if __name__ == "__main__":
    _test_linear_relu_linear()
    _test_fused_linear_relu()
    print("interpreter correct ✓")
