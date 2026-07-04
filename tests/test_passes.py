"""Tests for the TinyNN Tier 3 optimizer passes (tinynn.passes).

These cover the ``FusedLinearReLU`` op end-to-end (IR validation, interpreter,
codegen) plus the optimization passes themselves: dead code elimination,
Linear+ReLU fusion, and the ``PassManager``/``default_pipeline`` composition.

The NumPy reference interpreter (``tinynn.interpreter.run``) is used
throughout as the correctness oracle: an optimized graph must produce
identical outputs to the original, un-optimized graph.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.graph import Graph, GraphBuilder, Node
from tinynn.interpreter import run
from tinynn.ops import FUSED_LINEAR_RELU, LINEAR, RELU
from tinynn.passes import (
    PassManager,
    dead_code_elimination,
    default_pipeline,
    fuse_linear_relu,
)


# --------------------------------------------------------------------------- #
# FusedLinearReLU op: direct construction + interpreter
# --------------------------------------------------------------------------- #
def test_fused_linear_relu_direct_interpreter():
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
    f0 = b.fused_linear_relu("fused_0", xn, weight=w, bias=bias)
    graph = b.build(output_node=f0)

    assert graph.get_node("fused_0").op == FUSED_LINEAR_RELU

    actual = run(graph, {"x": x})
    expected = np.maximum(x @ w + bias, 0.0)
    np.testing.assert_allclose(actual, expected)


def test_fused_linear_relu_shape_validation_matches_linear():
    # Bias/weight mismatch should fail exactly like Linear's rule.
    with pytest.raises(ValueError, match="does not match weight output dimension"):
        Node(
            name="fused_0",
            op=FUSED_LINEAR_RELU,
            inputs=("x",),
            shape=(2,),
            weight=np.zeros((3, 2)),
            bias=np.zeros((3,)),
        )


# --------------------------------------------------------------------------- #
# fuse_linear_relu: MLP chain fuses into two FusedLinearReLU nodes
# --------------------------------------------------------------------------- #
def _mlp_graph():
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

    b = GraphBuilder()
    xn = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=w1, bias=b1)
    r1 = b.relu("relu_1", l1)
    graph = b.build(output_node=r1)
    return graph, x


def test_fuse_linear_relu_mlp_chain():
    graph, x = _mlp_graph()

    fused = fuse_linear_relu(graph)

    assert len(fused.nodes) == 3  # x, fused_relu_0, fused_relu_1
    ops = [n.op for n in fused.nodes]
    assert ops == ["Input", FUSED_LINEAR_RELU, FUSED_LINEAR_RELU]
    # Fused nodes keep the ReLU names so downstream/output references work.
    assert [n.name for n in fused.nodes] == ["x", "relu_0", "relu_1"]
    assert fused.output_node == "relu_1"

    expected = run(graph, {"x": x})
    actual = run(fused, {"x": x})
    np.testing.assert_allclose(actual, expected)


# --------------------------------------------------------------------------- #
# fuse_linear_relu: negative cases
# --------------------------------------------------------------------------- #
def test_no_fusion_when_linear_is_the_graph_output():
    # linear_0 feeds relu_0, but graph.output_node points at linear_0 itself
    # (relu_0 is a dead/unused node hanging off it) -- fusing would discard
    # the pre-activation value the graph is declared to expose.
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.eye(3), bias=np.zeros(3))
    b.relu("relu_0", l0)
    graph = b.build(output_node=l0)

    fused = fuse_linear_relu(graph)

    assert [n.op for n in fused.nodes] == ["Input", "Linear", "ReLU"]
    assert fused.output_node == "linear_0"


def test_no_fusion_when_linear_has_multiple_consumers():
    # linear_0 is consumed by both relu_0 and the Output node "y" -- fusing
    # would discard the value "y" needs.
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.eye(3), bias=np.zeros(3))
    b.relu("relu_0", l0)
    b.output("y", l0)
    graph = b.build()  # last Output node ("y")

    fused = fuse_linear_relu(graph)

    assert [n.op for n in fused.nodes] == ["Input", "Linear", "ReLU", "Output"]
    assert fused.output_node == "y"

    x_val = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(run(fused, {"x": x_val}), run(graph, {"x": x_val}))


# --------------------------------------------------------------------------- #
# dead_code_elimination
# --------------------------------------------------------------------------- #
def test_dead_code_elimination_removes_unused_branch():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.eye(3), bias=np.zeros(3))
    r0 = b.relu("relu_0", l0)
    # Dead branch: nothing downstream references "linear_dead".
    b.linear("linear_dead", x, weight=np.ones((3, 5)), bias=np.zeros(5))
    graph = b.build(output_node=r0)

    assert [n.name for n in graph.nodes] == ["x", "linear_0", "relu_0", "linear_dead"]

    pruned = dead_code_elimination(graph)

    assert [n.name for n in pruned.nodes] == ["x", "linear_0", "relu_0"]
    assert pruned.output_node == "relu_0"

    x_val = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(run(pruned, {"x": x_val}), run(graph, {"x": x_val}))


def test_dead_code_elimination_removes_unused_input_node():
    # An entire Input node that nothing downstream consumes is dead code too.
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    b.input("unused", shape=(4,))
    l0 = b.linear("linear_0", x, weight=np.eye(2), bias=np.zeros(2))
    graph = b.build(output_node=l0)

    pruned = dead_code_elimination(graph)

    assert [n.name for n in pruned.nodes] == ["x", "linear_0"]


# --------------------------------------------------------------------------- #
# PassManager / default_pipeline
# --------------------------------------------------------------------------- #
def test_pass_manager_runs_passes_in_order():
    calls = []

    def mark_a(g: Graph) -> Graph:
        calls.append("a")
        return g

    def mark_b(g: Graph) -> Graph:
        calls.append("b")
        return g

    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    l0 = b.linear("linear_0", x, weight=np.eye(2), bias=np.zeros(2))
    graph = b.build(output_node=l0)

    pm = PassManager([mark_a, mark_b])
    result = pm.run(graph)

    assert calls == ["a", "b"]
    assert result is graph


def test_default_pipeline_combines_dce_and_fusion():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]), bias=np.array([0.1, -0.1]))
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=np.array([[2.0, -1.0], [0.5, 1.5]]), bias=np.array([0.0, 0.2]))
    r1 = b.relu("relu_1", l1)
    # Dead branch hanging off the input, unrelated to the fusable chain.
    b.linear("linear_dead", x, weight=np.ones((3, 3)), bias=np.zeros(3))
    graph = b.build(output_node=r1)

    pipeline = default_pipeline()
    optimized = pipeline.run(graph)

    assert [n.op for n in optimized.nodes] == ["Input", FUSED_LINEAR_RELU, FUSED_LINEAR_RELU]
    assert [n.name for n in optimized.nodes] == ["x", "relu_0", "relu_1"]

    x_val = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(run(optimized, {"x": x_val}), run(graph, {"x": x_val}))


# --------------------------------------------------------------------------- #
# End-to-end: fused graph compiled to C++ matches interpreter on the
# original, unfused graph.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not found on PATH")
def test_fused_graph_codegen_matches_original_interpreter(tmp_path):
    from tinynn.codegen import compile_graph

    graph, x = _mlp_graph()
    fused = fuse_linear_relu(graph)

    model = compile_graph(fused, tmp_path / "fused_case")
    actual = model.run(x)

    expected = run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
