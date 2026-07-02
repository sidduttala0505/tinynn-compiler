"""Tests for the TinyNN NumPy reference interpreter.

These build small, deterministic graphs with :class:`GraphBuilder` and check
``tinynn.interpreter.run`` against hand-computed NumPy expressions.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinynn.graph import GraphBuilder
from tinynn.interpreter import run


# --------------------------------------------------------------------------- #
# Input -> Linear
# --------------------------------------------------------------------------- #
def test_input_linear():
    x = np.array([1.0, -2.0, 3.0])
    w = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bias = np.array([0.5, -0.5])

    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=w, bias=bias)
    graph = b.build(output_node=l0)

    actual = run(graph, {"x": x})
    expected = x @ w + bias
    assert np.allclose(actual, expected)


# --------------------------------------------------------------------------- #
# Input -> Linear -> ReLU (mixed sign pre-activations)
# --------------------------------------------------------------------------- #
def test_input_linear_relu():
    x = np.array([1.0, -2.0, 3.0])
    w = np.array(
        [
            [0.5, -1.0, 2.0, 0.0],
            [1.5, 0.25, -0.5, 1.0],
            [-1.0, 0.75, 0.5, -2.0],
        ]
    )
    bias = np.array([0.1, -0.2, 0.3, 0.4])

    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=w, bias=bias)
    r0 = b.relu("relu_0", l0)
    graph = b.build(output_node=r0)

    pre_activation = x @ w + bias
    # sanity: make sure the test actually exercises both branches of ReLU
    assert (pre_activation < 0).any()
    assert (pre_activation > 0).any()

    actual = run(graph, {"x": x})
    expected = np.maximum(pre_activation, 0.0)
    assert np.allclose(actual, expected)

    # output_node pointing directly at a computational node (ReLU here).
    assert graph.output_node == "relu_0"


# --------------------------------------------------------------------------- #
# Input -> Linear -> ReLU -> Linear
# --------------------------------------------------------------------------- #
def test_input_linear_relu_linear():
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

    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=w1, bias=b1)
    graph = b.build(output_node=l1)

    actual = run(graph, {"x": x})
    expected = np.maximum(x @ w0 + b0, 0.0) @ w1 + b1
    assert np.allclose(actual, expected)


# --------------------------------------------------------------------------- #
# Input -> Linear -> Output (explicit Output node)
# --------------------------------------------------------------------------- #
def test_input_linear_output():
    x = np.array([2.0, -1.0])
    w = np.array([[1.0, 2.0], [0.5, -0.5]])
    bias = np.array([0.0, 1.0])

    b = GraphBuilder()
    xn = b.input("x", shape=(2,))
    l0 = b.linear("linear_0", xn, weight=w, bias=bias)
    out = b.output("y", l0)
    graph = b.build(output_node=out)

    assert graph.output_node == "y"

    actual = run(graph, {"x": x})
    expected = x @ w + bias
    assert np.allclose(actual, expected)


# --------------------------------------------------------------------------- #
# Bare ndarray input (single Input node) matches dict form
# --------------------------------------------------------------------------- #
def test_bare_array_input_matches_dict():
    x = np.array([1.0, -2.0, 3.0])
    w = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bias = np.array([0.5, -0.5])

    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=w, bias=bias)
    graph = b.build(output_node=l0)

    via_dict = run(graph, {"x": x})
    via_bare = run(graph, x)
    assert np.allclose(via_dict, via_bare)


# --------------------------------------------------------------------------- #
# Error cases
# --------------------------------------------------------------------------- #
def test_missing_input_value_raises():
    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=np.zeros((3, 2)), bias=np.zeros((2,)))
    graph = b.build(output_node=l0)

    with pytest.raises(KeyError):
        run(graph, {})


def test_wrong_input_shape_raises():
    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=np.zeros((3, 2)), bias=np.zeros((2,)))
    graph = b.build(output_node=l0)

    with pytest.raises(ValueError):
        run(graph, {"x": np.array([1.0, 2.0])})


# NOTE: every non-Input op in the current IR (Linear/ReLU/Output) takes
# exactly one input, so there is no way to build a graph with two Input nodes
# that both feed into computation using the current GraphBuilder/ops surface.
# We therefore do not test the "multiple Input nodes + bare array" error path
# here; `_resolve_inputs`'s branch for that case is covered by inspection
# instead of an executable test.
