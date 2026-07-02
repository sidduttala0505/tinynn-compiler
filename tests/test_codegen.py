"""End-to-end tests for the TinyNN C++ codegen + compiler driver.

These tests generate C++ from a Graph IR instance, compile it with a system
``g++``, run the resulting binary, and compare its output against:

    * the NumPy reference interpreter (``tinynn.interpreter.run``), and
    * a hand-computed NumPy expression (so correctness does not solely rely
      on the interpreter also being correct).

The whole module is skipped if ``g++`` is not available on PATH.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.codegen import compile_graph, generate_cpp, sanitize_name
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run as interp_run

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)


# --------------------------------------------------------------------------- #
# Case 1: Input -> Linear
# --------------------------------------------------------------------------- #
def test_input_linear(tmp_path):
    x = np.array([1.0, 2.0, -1.5])
    w = np.array(
        [
            [0.5, -1.0],
            [1.0, 0.5],
            [-0.5, 2.0],
        ]
    )
    b = np.array([0.1, -0.2])

    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    graph = builder.build(output_node=l0)

    # output_node points directly at a computational (Linear) node, not an
    # Output node -- exercise that explicitly (case 5 from the spec).
    assert graph.output_node == "linear_0"
    assert graph.get_node(graph.output_node).op == "Linear"

    model = compile_graph(graph, tmp_path / "case1")
    actual = model.run(x)

    expected = x @ w + b
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    interp_expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, interp_expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 2: Input -> Linear -> ReLU (some pre-activations negative)
# --------------------------------------------------------------------------- #
def test_input_linear_relu(tmp_path):
    x = np.array([1.0, -2.0, 0.5])
    w = np.array(
        [
            [0.5, -1.0, 2.0, 0.0],
            [1.5, 0.25, -0.5, 1.0],
            [-1.0, 0.75, 0.5, -2.0],
        ]
    )
    b = np.array([0.1, -0.2, 0.3, 0.4])

    pre_activation = x @ w + b
    # Sanity check the test fixture actually exercises ReLU's negative branch.
    assert np.any(pre_activation < 0)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    r0 = builder.relu("relu_0", l0)
    graph = builder.build(output_node=r0)

    model = compile_graph(graph, tmp_path / "case2")
    actual = model.run(x)

    expected = np.maximum(pre_activation, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    interp_expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, interp_expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 3: Input -> Linear -> ReLU -> Linear
# --------------------------------------------------------------------------- #
def test_input_linear_relu_linear(tmp_path):
    x = np.array([1.0, -2.0, 0.5])
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

    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    l0 = builder.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = builder.relu("relu_0", l0)
    l1 = builder.linear("linear_1", r0, weight=w1, bias=b1)
    graph = builder.build(output_node=l1)

    model = compile_graph(graph, tmp_path / "case3")
    actual = model.run(x)

    expected = np.maximum(x @ w0 + b0, 0.0) @ w1 + b1
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    interp_expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, interp_expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 4: Input -> Linear -> Output (explicit Output node)
# --------------------------------------------------------------------------- #
def test_input_linear_output_node(tmp_path):
    x = np.array([2.0, 0.5])
    w = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, -2.0, 1.5],
        ]
    )
    b = np.array([0.0, 1.0, -0.5])

    builder = GraphBuilder()
    xn = builder.input("x", shape=(2,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    out = builder.output("y", l0)
    graph = builder.build(output_node=out)

    assert graph.output_node == "y"
    assert graph.get_node(graph.output_node).op == "Output"

    model = compile_graph(graph, tmp_path / "case4")
    actual = model.run(x)

    expected = x @ w + b
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    interp_expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, interp_expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# generate_cpp / sanitize_name: identifier sanitization, source-only
# --------------------------------------------------------------------------- #
def test_generate_cpp_sanitizes_identifiers():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.5, -0.5])

    builder = GraphBuilder()
    xn = builder.input("layer-0.input", shape=(2,))
    l0 = builder.linear("layer-0", xn, weight=w, bias=b)
    graph = builder.build(output_node=l0)

    source = generate_cpp(graph)

    expected_input_ident = sanitize_name("layer-0.input")
    expected_linear_ident = sanitize_name("layer-0")
    assert expected_input_ident == "v_layer_0_input"
    assert expected_linear_ident == "v_layer_0"

    assert expected_input_ident in source
    assert expected_linear_ident in source
    # Weight/bias globals are named from the (sanitized) node identifier.
    assert f"{expected_linear_ident}_w" in source
    assert f"{expected_linear_ident}_b" in source


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #
def test_generate_cpp_rejects_multiple_input_nodes():
    builder = GraphBuilder()
    x1 = builder.input("x1", shape=(2,))
    builder.input("x2", shape=(2,))
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    l0 = builder.linear("linear_0", x1, weight=w, bias=b)
    graph = builder.build(output_node=l0)

    with pytest.raises(ValueError, match="one Input node"):
        generate_cpp(graph)


def test_generate_cpp_rejects_non_1d_shape():
    # The Graph IR only restricts Linear/ReLU/Output shapes to match their
    # input; an Input node's declared shape is not otherwise constrained, so
    # this is a legal (if unusual) way to construct a 2D-shape node and
    # exercise codegen's Tier-0 "1D only" validation.
    builder = GraphBuilder()
    builder.input("x", shape=(2, 3))
    graph = builder.build(output_node="x")

    with pytest.raises(ValueError, match="1D"):
        generate_cpp(graph)
