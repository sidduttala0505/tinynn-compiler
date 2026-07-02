"""Tests for the TinyNN Graph IR (Phase 0, Step 1).

These cover *only* the IR: construction, validation, shape inference, and
serialization. No interpreter or codegen is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinynn.graph import Graph, GraphBuilder, Node


# --------------------------------------------------------------------------- #
# Test 1: single Linear graph
# --------------------------------------------------------------------------- #
def test_single_linear_graph():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.zeros((3, 2)), bias=np.zeros((2,)))
    graph = b.build(output_node=l0)

    assert len(graph.nodes) == 2
    assert graph.output_node == "linear_0"
    assert [n.name for n in graph.nodes] == ["x", "linear_0"]
    assert graph.get_node("x").shape == (3,)
    assert graph.get_node("linear_0").shape == (2,)


# --------------------------------------------------------------------------- #
# Test 2: MLP graph
# --------------------------------------------------------------------------- #
def test_mlp_graph():
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    l0 = b.linear("linear_0", x, weight=np.zeros((4, 5)), bias=np.zeros((5,)))
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=np.zeros((5, 2)), bias=np.zeros((2,)))
    graph = b.build(output_node=l1)

    assert [n.name for n in graph.nodes] == ["x", "linear_0", "relu_0", "linear_1"]
    assert graph.get_node("linear_0").shape == (5,)
    assert graph.get_node("relu_0").shape == (5,)
    assert graph.get_node("linear_1").shape == (2,)
    assert graph.output_node == "linear_1"


# --------------------------------------------------------------------------- #
# Test 3: explicit Output node
# --------------------------------------------------------------------------- #
def test_explicit_output_node():
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    l0 = b.linear("linear_0", x, weight=np.zeros((3, 2)), bias=np.zeros((2,)))
    out = b.output("out", l0)
    graph = b.build()  # uses last Output node

    assert graph.output_node == "out"
    assert graph.get_node(out).shape == graph.get_node(l0).shape == (2,)


# --------------------------------------------------------------------------- #
# Test 4: duplicate node names fail
# --------------------------------------------------------------------------- #
def test_duplicate_node_names_fail():
    nodes = (
        Node(name="x", op="Input", shape=(3,)),
        Node(name="x", op="Input", shape=(3,)),
    )
    with pytest.raises(ValueError, match="Duplicate node name: x"):
        Graph(nodes=nodes, output_node="x")


# --------------------------------------------------------------------------- #
# Test 5: missing input reference fails
# --------------------------------------------------------------------------- #
def test_missing_input_reference_fails():
    nodes = (
        Node(name="x", op="Input", shape=(3,)),
        Node(name="relu_0", op="ReLU", inputs=("linear_0",), shape=(3,)),
    )
    with pytest.raises(ValueError, match="references unknown input linear_0"):
        Graph(nodes=nodes, output_node="relu_0")


# --------------------------------------------------------------------------- #
# Test 6: non-topological order fails
# --------------------------------------------------------------------------- #
def test_non_topological_order_fails():
    # relu_0 references linear_0 but appears before it.
    nodes = (
        Node(name="x", op="Input", shape=(3,)),
        Node(name="relu_0", op="ReLU", inputs=("linear_0",), shape=(2,)),
        Node(
            name="linear_0",
            op="Linear",
            inputs=("x",),
            shape=(2,),
            weight=np.zeros((3, 2)),
            bias=np.zeros((2,)),
        ),
    )
    with pytest.raises(ValueError, match="topologically ordered"):
        Graph(nodes=nodes, output_node="linear_0")


# --------------------------------------------------------------------------- #
# Test 7: Linear shape mismatches fail
# --------------------------------------------------------------------------- #
def test_linear_input_shape_mismatch_fails():
    # input shape (3,) but weight input dimension 4
    b = GraphBuilder()
    x = b.input("x", shape=(3,))
    b.linear("linear_0", x, weight=np.zeros((4, 2)), bias=np.zeros((2,)))
    with pytest.raises(ValueError, match="weight has input dimension 4"):
        b.build(output_node="linear_0")


def test_linear_bias_shape_mismatch_fails():
    # weight (3, 2) but bias (3,)
    with pytest.raises(ValueError, match="does not match weight output dimension 2"):
        Node(
            name="linear_0",
            op="Linear",
            inputs=("x",),
            shape=(2,),
            weight=np.zeros((3, 2)),
            bias=np.zeros((3,)),
        )


# --------------------------------------------------------------------------- #
# Test 8: ReLU shape inference
# --------------------------------------------------------------------------- #
def test_relu_shape_inference():
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    l0 = b.linear("linear_0", x, weight=np.zeros((4, 7)), bias=np.zeros((7,)))
    r0 = b.relu("relu_0", l0)
    graph = b.build(output_node=r0)

    assert graph.get_node("relu_0").shape == graph.get_node("linear_0").shape == (7,)


# --------------------------------------------------------------------------- #
# Test 9: to_dict/from_dict round trip
# --------------------------------------------------------------------------- #
def test_serialization_round_trip():
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    l0 = b.linear("linear_0", x, weight=np.random.randn(4, 5), bias=np.random.randn(5))
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=np.random.randn(5, 2), bias=np.random.randn(2))
    graph = b.build(output_node=l1)

    restored = Graph.from_dict(graph.to_dict())

    assert restored.output_node == graph.output_node
    assert [n.name for n in restored.nodes] == [n.name for n in graph.nodes]
    for orig, new in zip(graph.nodes, restored.nodes):
        assert orig.name == new.name
        assert orig.op == new.op
        assert orig.inputs == new.inputs
        assert orig.shape == new.shape
        if orig.weight is None:
            assert new.weight is None
        else:
            assert np.array_equal(orig.weight, new.weight)
        if orig.bias is None:
            assert new.bias is None
        else:
            assert np.array_equal(orig.bias, new.bias)


# --------------------------------------------------------------------------- #
# A few extra sanity checks on the IR contract.
# --------------------------------------------------------------------------- #
def test_input_node_rejects_weight():
    with pytest.raises(ValueError, match="must not have a weight"):
        Node(name="x", op="Input", shape=(3,), weight=np.zeros((3, 2)))


def test_output_node_uses_last_output_on_build():
    b = GraphBuilder()
    x = b.input("x", shape=(2,))
    b.output("out", x)
    graph = b.build()
    assert graph.output_node == "out"


def test_build_without_output_raises():
    b = GraphBuilder()
    b.input("x", shape=(2,))
    with pytest.raises(ValueError, match="No output_node"):
        b.build()


def test_summary_lists_nodes_in_order():
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    l0 = b.linear("linear_0", x, weight=np.zeros((4, 5)), bias=np.zeros((5,)))
    r0 = b.relu("relu_0", l0)
    l1 = b.linear("linear_1", r0, weight=np.zeros((5, 2)), bias=np.zeros((2,)))
    graph = b.build(output_node=l1)

    text = graph.summary()
    assert "output_node='linear_1'" in text
    assert "x: Input inputs=[] shape=(4,)" in text
    assert "linear_1: Linear inputs=['relu_0'] shape=(2,)" in text
