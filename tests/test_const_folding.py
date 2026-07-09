"""Tests for the Tier 3 ``Const`` op and the constant-folding /
algebraic-simplification optimization passes (``tinynn.passes``).

Covers:
    * the ``Const`` op end-to-end: IR validation, interpreter, codegen,
      ``infer_shapes``;
    * :func:`tinynn.passes.constant_folding`: single-pass forward folding of
      fully-constant subgraphs into ``Const`` nodes, including chains;
    * :func:`tinynn.passes.simplify_algebraic`: add/sub-zero, mul-one, and
      redundant-ReLU aliasing, including stacked identities and the case
      where the eliminated node is ``graph.output_node``;
    * the two passes folded into ``default_pipeline()``.

The NumPy reference interpreter (``tinynn.interpreter.run``) is the
correctness oracle throughout. End-to-end compiled (C++) checks are skipped
when g++ is not available on PATH, matching ``tests/test_codegen.py``.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.analysis import infer_shapes
from tinynn.graph import Graph, GraphBuilder, Node
from tinynn.interpreter import run
from tinynn.ops import CONST, FUSED_LINEAR_RELU, RELU
from tinynn.passes import (
    PassManager,
    constant_folding,
    dead_code_elimination,
    default_pipeline,
    fuse_linear_relu,
    simplify_algebraic,
)

requires_gpp = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)


# --------------------------------------------------------------------------- #
# Const op: end-to-end (interpreter + codegen)
# --------------------------------------------------------------------------- #
def test_const_interpreter_add():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    c = b.const("c", np.array([10.0, 20.0, 30.0]))
    y = b.add("y", x, c)
    graph = b.build(output_node=y)

    x_val = np.array([1.0, 2.0, 3.0])
    actual = run(graph, {"x": x_val})
    expected = x_val + np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


@requires_gpp
def test_const_codegen_1d_add_matches_interpreter(tmp_path):
    from tinynn.codegen import compile_graph

    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    c = b.const("c", np.array([10.0, 20.0, 30.0]))
    y = b.add("y", x, c)
    graph = b.build(output_node=y)

    x_val = np.array([1.0, 2.0, 3.0])
    model = compile_graph(graph, tmp_path / "const_add")
    actual = model.run(x_val)
    expected = run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@requires_gpp
def test_const_codegen_2d_matmul_matches_interpreter(tmp_path):
    from tinynn.codegen import compile_graph

    x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    const_val = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])

    b = GraphBuilder()
    x = b.input("x", shape=(2, 2))
    c = b.const("c", const_val)
    y = b.matmul("y", x, c)
    graph = b.build(output_node=y)

    model = compile_graph(graph, tmp_path / "const_matmul")
    actual = model.run(x_val)
    expected = run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, x_val @ const_val, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Const op: validation
# --------------------------------------------------------------------------- #
def test_const_with_inputs_raises():
    with pytest.raises(ValueError, match="must have no inputs"):
        Node(name="c", op=CONST, inputs=("x",), shape=(3,), weight=np.zeros(3))


def test_const_without_weight_raises():
    with pytest.raises(ValueError, match="must have a weight"):
        Node(name="c", op=CONST, inputs=(), shape=(3,))


def test_const_with_bias_raises():
    with pytest.raises(ValueError, match="must not have a bias"):
        Node(
            name="c",
            op=CONST,
            inputs=(),
            shape=(3,),
            weight=np.zeros(3),
            bias=np.zeros(3),
        )


def test_const_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must equal"):
        Node(name="c", op=CONST, inputs=(), shape=(4,), weight=np.zeros(3))


# --------------------------------------------------------------------------- #
# infer_shapes on a graph containing a 2D Const (ordering-trap regression)
# --------------------------------------------------------------------------- #
def test_infer_shapes_with_2d_const():
    const_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = GraphBuilder()
    x = b.input("x", shape=(2, 3))
    c = b.const("c", const_val)
    y = b.matmul("y", x, c)
    graph = b.build(output_node=y)

    shapes = infer_shapes(graph)
    assert shapes["c"] == (3, 2)
    assert shapes["y"] == (2, 2)


# --------------------------------------------------------------------------- #
# constant_folding
# --------------------------------------------------------------------------- #
def test_constant_folding_add_chain_keeps_c1_c2_for_dce():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    c1 = b.const("c1", np.array([1.0, 2.0, 3.0]))
    c2 = b.const("c2", np.array([4.0, 5.0, 6.0]))
    s = b.add("s", c1, c2)
    y = b.mul("y", x, s)
    graph = b.build(output_node=y)

    folded = constant_folding(graph)

    assert [n.name for n in folded.nodes] == ["x", "c1", "c2", "s", "y"]
    s_node = folded.get_node("s")
    assert s_node.op == CONST
    np.testing.assert_allclose(s_node.weight, [5.0, 7.0, 9.0])
    # c1/c2 are still present -- constant_folding never removes nodes, only
    # replaces them; sweeping now-dead nodes is DCE's job.
    assert folded.get_node("c1").op == CONST
    assert folded.get_node("c2").op == CONST

    x_val = np.array([1.0, -1.0, 2.0])
    np.testing.assert_allclose(run(folded, {"x": x_val}), run(graph, {"x": x_val}))


def test_constant_folding_linear_chain_collapses_to_single_const():
    w0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b0 = np.array([0.5, -0.5])
    w1 = np.array([[2.0, 0.0], [0.0, 2.0]])
    b1 = np.array([1.0, 1.0])

    b = GraphBuilder()
    c0 = b.const("c0", np.array([1.0, 2.0]))
    l0 = b.linear("l0", c0, weight=w0, bias=b0)
    l1 = b.linear("l1", l0, weight=w1, bias=b1)
    graph = b.build(output_node=l1)

    folded = constant_folding(graph)

    assert folded.get_node("l1").op == CONST
    expected = (np.array([1.0, 2.0]) @ w0 + b0) @ w1 + b1
    np.testing.assert_allclose(folded.get_node("l1").weight, expected)
    np.testing.assert_allclose(run(folded, {}), run(graph, {}))


def test_constant_folding_tanh_of_const():
    b = GraphBuilder()
    c = b.const("c", np.array([0.0, 1.0, -1.0]))
    t = b.tanh("t", c)
    graph = b.build(output_node=t)

    folded = constant_folding(graph)

    assert folded.get_node("t").op == CONST
    np.testing.assert_allclose(folded.get_node("t").weight, np.tanh([0.0, 1.0, -1.0]))


def test_constant_folding_no_fold_with_runtime_operand():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    c = b.const("c", np.array([1.0, 2.0, 3.0]))
    y = b.add("y", x, c)
    graph = b.build(output_node=y)

    folded = constant_folding(graph)

    # x is a runtime Input -- y cannot fold.
    assert folded.get_node("y").op == "Add"
    assert folded.get_node("x").op == "Input"


# --------------------------------------------------------------------------- #
# simplify_algebraic
# --------------------------------------------------------------------------- #
def test_simplify_add_zero_both_orders():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    zeros = b.const("zeros", np.zeros(2))
    y1 = b.add("y1", x, zeros)
    y2 = b.add("y2", zeros, x)
    graph = b.build(output_node=y2)
    # Make y1 reachable too by making it the input of an extra op referencing it.
    graph = graph.with_node(Node(name="keep", op=RELU, inputs=(y1,), shape=(2,)))
    graph = graph.with_output("keep")

    simplified = simplify_algebraic(graph)

    keep = simplified.get_node("keep")
    assert keep.inputs == ("x",)
    assert simplified.output_node == "keep"

    x_val = np.array([1.0, -2.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


def test_simplify_mul_one_both_orders():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    ones = b.const("ones", np.ones(2))
    y1 = b.mul("y1", x, ones)
    y2 = b.mul("y2", ones, x)
    graph = b.build(output_node=y2)
    graph = graph.with_node(Node(name="keep", op=RELU, inputs=(y1,), shape=(2,)))
    graph = graph.with_output("keep")

    simplified = simplify_algebraic(graph)

    assert simplified.get_node("keep").inputs == ("x",)
    assert simplified.output_node == "keep"

    x_val = np.array([3.0, -4.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


def test_simplify_sub_zero():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    zeros = b.const("zeros", np.zeros(2))
    y = b.sub("y", x, zeros)
    graph = b.build(output_node=y)

    simplified = simplify_algebraic(graph)

    assert simplified.output_node == "x"
    assert [n.name for n in simplified.nodes] == ["x", "zeros"]

    x_val = np.array([5.0, -6.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


def test_simplify_sub_zero_does_not_fire_when_zero_is_first_operand():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    zeros = b.const("zeros", np.zeros(2))
    y = b.sub("y", zeros, x)  # zeros - x == -x, not an identity
    graph = b.build(output_node=y)

    simplified = simplify_algebraic(graph)

    assert simplified.get_node("y").op == "Sub"
    assert simplified.output_node == "y"


def test_simplify_relu_of_relu():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    r0 = b.relu("r0", x)
    r1 = b.relu("r1", r0)
    graph = b.build(output_node=r1)

    simplified = simplify_algebraic(graph)

    assert simplified.output_node == "r0"
    assert [n.name for n in simplified.nodes] == ["x", "r0"]

    x_val = np.array([1.0, -1.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


def test_simplify_eliminated_node_is_output_node():
    # The eliminated node (y, an add-zero identity) is itself graph.output_node
    # -- output_node must remap to its replacement.
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    zeros = b.const("zeros", np.zeros(2))
    y = b.add("y", x, zeros)
    graph = b.build(output_node=y)

    simplified = simplify_algebraic(graph)

    assert simplified.output_node == "x"
    assert "y" not in [n.name for n in simplified.nodes]

    x_val = np.array([2.0, 3.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


def test_simplify_chained_add_zero_collapses_via_fixpoint():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    zeros = b.const("zeros", np.zeros(2))
    a1 = b.add("a1", x, zeros)
    a2 = b.add("a2", a1, zeros)
    graph = b.build(output_node=a2)

    simplified = simplify_algebraic(graph)

    assert simplified.output_node == "x"
    assert [n.name for n in simplified.nodes] == ["x", "zeros"]

    x_val = np.array([7.0, -8.0])
    np.testing.assert_allclose(run(simplified, {"x": x_val}), run(graph, {"x": x_val}))


# --------------------------------------------------------------------------- #
# Pipeline: const subexpression + dead branch + Linear->ReLU pair
# --------------------------------------------------------------------------- #
def _pipeline_graph():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))

    # Constant subexpression: folds to a single Const, then a zero-add on top
    # of it should get eliminated by simplify_algebraic (after folding
    # produces the zero Const's sibling is already a Const, so this exercises
    # constant folding feeding into algebraic simplification).
    c1 = b.const("c1", np.array([1.0, 1.0]))
    c2 = b.const("c2", np.array([-1.0, -1.0]))
    s = b.add("s", c1, c2)  # folds to Const([0, 0])
    y0 = b.add("y0", x, s)  # Add(x, zero-Const) -> simplifies to x

    # Dead branch, unrelated to the output.
    b.linear("dead_linear", x, weight=np.ones((2, 3)), bias=np.zeros(3))

    # Linear -> ReLU pair that should fuse.
    l0 = b.linear("linear_0", y0, weight=np.array([[1.0, -1.0], [0.5, 2.0]]), bias=np.array([0.1, -0.2]))
    r0 = b.relu("relu_0", l0)

    graph = b.build(output_node=r0)
    x_val = np.array([1.0, -2.0])
    return graph, x_val


def test_default_pipeline_ops_and_semantics():
    graph, x_val = _pipeline_graph()

    optimized = default_pipeline().run(graph)

    ops = sorted(n.op for n in optimized.nodes)
    assert ops == sorted(["Input", FUSED_LINEAR_RELU])

    expected = run(graph, {"x": x_val})
    actual = run(optimized, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@requires_gpp
def test_default_pipeline_codegen_matches_original_interpreter(tmp_path):
    from tinynn.codegen import compile_graph

    graph, x_val = _pipeline_graph()
    optimized = default_pipeline().run(graph)

    model = compile_graph(optimized, tmp_path / "pipeline_case")
    actual = model.run(x_val)
    expected = run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_default_pipeline_pass_order():
    pm = default_pipeline()
    names = [name for name, _ in pm.passes]
    assert names == [
        "constant_folding",
        "simplify_algebraic",
        "dead_code_elimination",
        "fuse_linear_relu",
    ]
