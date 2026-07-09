"""Tests for the Tier 2 ops: Add, Sub, Mul, MatMul, Softmax, Tanh, Sigmoid.

Covers:
    * interpreter semantics vs hand-computed NumPy expressions,
    * end-to-end compiled (C++) results vs the interpreter oracle,
    * CompiledModel.run's dict-input contract and error cases,
    * IR validation errors at Node/Graph construction time,
    * that the existing optimization pipeline stays safe alongside new ops.

End-to-end tests are skipped when g++ is not available on PATH.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.codegen import compile_graph
from tinynn.graph import Graph, GraphBuilder, Node
from tinynn.interpreter import run as interp_run
from tinynn.passes import default_pipeline

requires_gpp = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)


# --------------------------------------------------------------------------- #
# Interpreter semantics vs hand-computed NumPy
# --------------------------------------------------------------------------- #
def test_interp_add():
    a = np.array([1.0, -2.0, 3.5])
    b = np.array([0.5, 4.0, -1.0])
    builder = GraphBuilder()
    an = builder.input("a", shape=(3,))
    bn = builder.input("b", shape=(3,))
    y = builder.add("y", an, bn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, {"a": a, "b": b})
    np.testing.assert_allclose(out, a + b, rtol=1e-15, atol=1e-15)


def test_interp_sub_order_matters():
    a = np.array([1.0, -2.0, 3.5])
    b = np.array([0.5, 4.0, -1.0])
    builder = GraphBuilder()
    an = builder.input("a", shape=(3,))
    bn = builder.input("b", shape=(3,))
    y = builder.sub("y", an, bn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, {"a": a, "b": b})
    np.testing.assert_allclose(out, a - b, rtol=1e-15, atol=1e-15)
    # a - b, never b - a (the fixture makes the two visibly different).
    assert not np.allclose(out, b - a)


def test_interp_mul():
    a = np.array([[1.0, -2.0], [3.5, 0.25]])
    b = np.array([[0.5, 4.0], [-1.0, 2.0]])
    builder = GraphBuilder()
    an = builder.input("a", shape=(2, 2))
    bn = builder.input("b", shape=(2, 2))
    y = builder.mul("y", an, bn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, {"a": a, "b": b})
    np.testing.assert_allclose(out, a * b, rtol=1e-15, atol=1e-15)


def test_interp_matmul_2d():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4))
    builder = GraphBuilder()
    an = builder.input("a", shape=(2, 3))
    bn = builder.input("b", shape=(3, 4))
    y = builder.matmul("y", an, bn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, {"a": a, "b": b})
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, a @ b, rtol=1e-15, atol=1e-15)


def test_interp_softmax():
    x = np.array([1.0, 2.0, 3.0])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    y = builder.softmax("y", xn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, x)
    e = np.exp(x - np.max(x))
    np.testing.assert_allclose(out, e / np.sum(e), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.sum(out), 1.0, rtol=1e-12)


def test_interp_softmax_large_values_stable():
    x = np.array([1000.0, 1001.0, 1002.0])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    y = builder.softmax("y", xn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.sum(out), 1.0, rtol=1e-12)
    # Same as softmax of [0, 1, 2] (shift invariance).
    e = np.exp(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(out, e / np.sum(e), rtol=1e-12, atol=1e-15)


def test_interp_tanh():
    x = np.array([[-2.0, 0.0], [0.5, 3.0]])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(2, 2))
    y = builder.tanh("y", xn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, x)
    np.testing.assert_allclose(out, np.tanh(x), rtol=1e-15, atol=1e-15)


def test_interp_sigmoid():
    x = np.array([-3.0, -0.5, 0.0, 1.5])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(4,))
    y = builder.sigmoid("y", xn)
    graph = builder.build(output_node=y)

    out = interp_run(graph, x)
    np.testing.assert_allclose(out, 1.0 / (1.0 + np.exp(-x)), rtol=1e-15, atol=1e-15)


# --------------------------------------------------------------------------- #
# End-to-end: compiled C++ vs interpreter oracle
# --------------------------------------------------------------------------- #
@requires_gpp
def test_e2e_diamond_tanh_sigmoid_add(tmp_path):
    # Diamond: one Input feeding both Tanh and Sigmoid, joined by Add.
    x = np.array([-1.5, 0.0, 0.75, 2.0])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(4,))
    t = builder.tanh("t", xn)
    s = builder.sigmoid("s", xn)
    y = builder.add("y", t, s)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "diamond")
    actual = model.run(x)
    expected = interp_run(graph, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@requires_gpp
def test_e2e_two_input_add_dict_inputs(tmp_path):
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([0.25, 0.5, -4.0])
    builder = GraphBuilder()
    an = builder.input("a", shape=(3,))
    bn = builder.input("b", shape=(3,))
    y = builder.add("y", an, bn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "two_input_add")
    actual = model.run({"a": a, "b": b})
    expected = interp_run(graph, {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual, a + b, rtol=1e-9, atol=1e-9)


@requires_gpp
def test_e2e_matmul_two_2d_inputs(tmp_path):
    rng = np.random.default_rng(7)
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4))
    builder = GraphBuilder()
    an = builder.input("a", shape=(2, 3))
    bn = builder.input("b", shape=(3, 4))
    y = builder.matmul("y", an, bn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "matmul")
    actual = model.run({"a": a, "b": b})
    assert actual.shape == (2, 4)
    expected = interp_run(graph, {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual, a @ b, rtol=1e-9, atol=1e-9)


@requires_gpp
def test_e2e_linear_softmax_chain(tmp_path):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(4)
    w = rng.standard_normal((4, 3))
    b = rng.standard_normal(3)
    builder = GraphBuilder()
    xn = builder.input("x", shape=(4,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    y = builder.softmax("y", l0)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "softmax_chain")
    actual = model.run(x)
    expected = interp_run(graph, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.sum(actual), 1.0, rtol=1e-9)


@requires_gpp
def test_e2e_mixed_linear_tanh_linear_softmax(tmp_path):
    rng = np.random.default_rng(13)
    x = rng.standard_normal(5)
    w0 = rng.standard_normal((5, 6))
    b0 = rng.standard_normal(6)
    w1 = rng.standard_normal((6, 3))
    b1 = rng.standard_normal(3)
    builder = GraphBuilder()
    xn = builder.input("x", shape=(5,))
    l0 = builder.linear("linear_0", xn, weight=w0, bias=b0)
    t0 = builder.tanh("tanh_0", l0)
    l1 = builder.linear("linear_1", t0, weight=w1, bias=b1)
    y = builder.softmax("y", l1)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "mixed")
    actual = model.run(x)
    expected = interp_run(graph, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# CompiledModel.run compat / error cases
# --------------------------------------------------------------------------- #
@requires_gpp
def test_compiled_run_bare_array_rejected_for_two_inputs(tmp_path):
    builder = GraphBuilder()
    an = builder.input("a", shape=(2,))
    bn = builder.input("b", shape=(2,))
    y = builder.add("y", an, bn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "bare_two_inputs")
    with pytest.raises(ValueError, match="exactly one Input node"):
        model.run(np.array([1.0, 2.0]))


@requires_gpp
def test_compiled_run_missing_dict_input_raises(tmp_path):
    builder = GraphBuilder()
    an = builder.input("a", shape=(2,))
    bn = builder.input("b", shape=(2,))
    y = builder.add("y", an, bn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "missing_input")
    with pytest.raises(KeyError, match="b"):
        model.run({"a": np.array([1.0, 2.0])})


@requires_gpp
def test_compiled_run_wrong_size_input_raises(tmp_path):
    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    y = builder.tanh("y", xn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "wrong_size")
    with pytest.raises(ValueError, match="expected 3 values"):
        model.run(np.array([1.0, 2.0]))


@requires_gpp
def test_compiled_run_backward_compat_single_input_1d(tmp_path):
    # Single-input, 1D graphs keep the old contract: bare positional array
    # in, 1D array out.
    x = np.array([1.0, -1.0, 0.5])
    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    y = builder.relu("y", xn)
    graph = builder.build(output_node=y)

    model = compile_graph(graph, tmp_path / "compat")
    actual = model.run(x)
    assert actual.ndim == 1
    np.testing.assert_allclose(actual, np.maximum(x, 0.0), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# IR validation errors
# --------------------------------------------------------------------------- #
def test_add_mismatched_shapes_rejected_at_graph_build():
    nodes = (
        Node(name="a", op="Input", shape=(2,)),
        Node(name="b", op="Input", shape=(3,)),
        Node(name="y", op="Add", inputs=("a", "b"), shape=(2,)),
    )
    with pytest.raises(ValueError, match="equal shapes"):
        Graph(nodes=nodes, output_node="y")


def test_matmul_with_1d_input_rejected():
    nodes = (
        Node(name="a", op="Input", shape=(3,)),
        Node(name="b", op="Input", shape=(3, 4)),
        Node(name="y", op="MatMul", inputs=("a", "b"), shape=(3, 4)),
    )
    with pytest.raises(ValueError, match="2D"):
        Graph(nodes=nodes, output_node="y")


def test_softmax_on_2d_rejected():
    nodes = (
        Node(name="x", op="Input", shape=(2, 3)),
        Node(name="y", op="Softmax", inputs=("x",), shape=(2, 3)),
    )
    with pytest.raises(ValueError, match="1D"):
        Graph(nodes=nodes, output_node="y")


def test_add_with_one_input_rejected_at_node_level():
    with pytest.raises(ValueError, match="exactly two inputs"):
        Node(name="y", op="Add", inputs=("a",), shape=(2,))


# --------------------------------------------------------------------------- #
# Optimization pipeline stays safe alongside the new ops
# --------------------------------------------------------------------------- #
def test_default_pipeline_safe_with_new_ops():
    rng = np.random.default_rng(21)
    x = rng.standard_normal(4)
    w0 = rng.standard_normal((4, 5))
    b0 = rng.standard_normal(5)
    w1 = rng.standard_normal((5, 5))
    b1 = rng.standard_normal(5)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(4,))
    l0 = builder.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = builder.relu("relu_0", l0)  # fusable Linear -> ReLU pair
    l1 = builder.linear("linear_1", r0, weight=w1, bias=b1)
    t0 = builder.tanh("tanh_0", l1)
    s0 = builder.sigmoid("sigmoid_0", l1)
    a0 = builder.add("add_0", t0, s0)
    y = builder.softmax("y", a0)
    graph = builder.build(output_node=y)

    optimized = default_pipeline().run(graph)

    original_out = interp_run(graph, x)
    optimized_out = interp_run(optimized, x)
    np.testing.assert_allclose(optimized_out, original_out, rtol=1e-12, atol=1e-12)

    # Sanity: the Linear -> ReLU pair actually fused.
    ops = {n.op for n in optimized.nodes}
    assert "FusedLinearReLU" in ops
