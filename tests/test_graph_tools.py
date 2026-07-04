"""Tests for the Tier 1 graph-infrastructure utilities:

    - tinynn.analysis (topological_sort, infer_shapes)
    - tinynn.serialize (save_json, load_json)
    - tinynn.viz (to_dot, save_dot)

These are new, standalone modules; none of tinynn/graph.py, tinynn/ops.py,
tinynn/interpreter.py or tinynn/codegen.py are touched or exercised here
beyond their existing public API.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinynn.analysis import infer_shapes, topological_sort
from tinynn.graph import Graph, GraphBuilder, Node
from tinynn.serialize import load_json, save_json
from tinynn.viz import save_dot, to_dot


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_mlp() -> Graph:
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    l0 = b.linear(
        "linear_0",
        x,
        weight=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        ),
        bias=np.array([0.01, 0.02, 0.03]),
    )
    r0 = b.relu("relu_0", l0)
    l1 = b.linear(
        "linear_1",
        r0,
        weight=np.array([[1.0, -1.0], [0.5, 0.5], [-0.25, 0.75]]),
        bias=np.array([0.1, -0.1]),
    )
    out = b.output("out", l1)
    return b.build(output_node=out)


# --------------------------------------------------------------------------- #
# topological_sort
# --------------------------------------------------------------------------- #
def test_topological_sort_orders_shuffled_mlp():
    graph = _build_mlp()
    shuffled = (
        graph.get_node("relu_0"),
        graph.get_node("out"),
        graph.get_node("x"),
        graph.get_node("linear_1"),
        graph.get_node("linear_0"),
    )

    ordered = topological_sort(shuffled)

    # Must be a valid order that Graph() accepts.
    rebuilt = Graph(nodes=ordered, output_node="out")
    assert [n.name for n in rebuilt.nodes] == ["x", "linear_0", "relu_0", "linear_1", "out"]


def test_topological_sort_is_stable_on_already_sorted_input():
    graph = _build_mlp()
    ordered = topological_sort(graph.nodes)
    assert ordered == graph.nodes


def test_topological_sort_detects_cycle():
    # relu_a -> relu_b -> relu_a: a two-node cycle using single-input,
    # weight-less nodes (Node construction itself does not check ordering).
    nodes = (
        Node(name="relu_a", op="ReLU", inputs=("relu_b",), shape=(3,)),
        Node(name="relu_b", op="ReLU", inputs=("relu_a",), shape=(3,)),
    )
    with pytest.raises(ValueError, match="Cycle"):
        topological_sort(nodes)


def test_topological_sort_detects_unknown_reference():
    nodes = (
        Node(name="x", op="Input", shape=(3,)),
        Node(name="relu_0", op="ReLU", inputs=("linear_0",), shape=(3,)),
    )
    with pytest.raises(ValueError, match="unknown input"):
        topological_sort(nodes)


def test_topological_sort_detects_duplicate_names():
    nodes = (
        Node(name="x", op="Input", shape=(3,)),
        Node(name="x", op="Input", shape=(3,)),
    )
    with pytest.raises(ValueError, match="Duplicate node name"):
        topological_sort(nodes)


# --------------------------------------------------------------------------- #
# infer_shapes
# --------------------------------------------------------------------------- #
def test_infer_shapes_matches_declared_shapes():
    graph = _build_mlp()
    shapes = infer_shapes(graph)

    assert shapes == {
        "x": (4,),
        "linear_0": (3,),
        "relu_0": (3,),
        "linear_1": (2,),
        "out": (2,),
    }
    for node in graph.nodes:
        assert shapes[node.name] == node.shape


# --------------------------------------------------------------------------- #
# serialize
# --------------------------------------------------------------------------- #
def test_serialize_round_trip(tmp_path):
    graph = _build_mlp()
    path = save_json(graph, tmp_path / "nested" / "mlp.json")

    assert path.exists()
    restored = load_json(path)

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


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "does_not_exist.json")


def test_load_json_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ValueError):
        load_json(bad)


# --------------------------------------------------------------------------- #
# viz
# --------------------------------------------------------------------------- #
def test_to_dot_contains_header_nodes_and_edges():
    graph = _build_mlp()
    dot = to_dot(graph)

    assert dot.startswith("digraph tinynn {")
    assert dot.rstrip().endswith("}")

    for node in graph.nodes:
        assert f'"{node.name}"' in dot

    assert '"x" -> "linear_0"' in dot
    assert '"linear_0" -> "relu_0"' in dot
    assert '"relu_0" -> "linear_1"' in dot
    assert '"linear_1" -> "out"' in dot

    # Output node visually distinct.
    out_line = [line for line in dot.splitlines() if line.strip().startswith('"out"')][0]
    assert "penwidth=2" in out_line
    assert "style=bold" in out_line

    # Weighted (Linear) nodes are boxes; weight-less nodes are ellipses.
    linear0_line = [line for line in dot.splitlines() if line.strip().startswith('"linear_0"')][0]
    assert "shape=box" in linear0_line
    x_line = [line for line in dot.splitlines() if line.strip().startswith('"x"')][0]
    assert "shape=ellipse" in x_line


def test_save_dot_writes_file(tmp_path):
    graph = _build_mlp()
    path = save_dot(graph, tmp_path / "viz" / "mlp.dot")

    assert path.exists()
    content = path.read_text()
    assert content == to_dot(graph)
