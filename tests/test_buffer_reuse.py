"""Tests for liveness-based buffer reuse in the C++ codegen (Tier 4).

``CodegenOptions(reuse_buffers=True)`` replaces the one-vector-per-node decl
section of the generated C++ with a smaller set of exact-size physical slots
(``tinynn_buf_<k>``) plus one reference per node. These tests check both the
plan's structure (fewer physical slots than logical buffers, hazard-free
assignments) and end-to-end numerical correctness against the NumPy reference
interpreter.

The whole module is skipped if ``g++`` is not available on PATH.
"""

from __future__ import annotations

import re
import shutil

import numpy as np
import pytest

from tinynn.codegen import CodegenOptions, compile_graph, generate_cpp
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run as interp_run

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)

REUSE = CodegenOptions(reuse_buffers=True)

_SLOT_DECL_RE = re.compile(r"^\s*std::vector<double> tinynn_buf_\d+\(\d+\);$", re.M)
_PLAN_RE = re.compile(
    r"^// buffer plan: (\d+) logical buffers -> (\d+) physical slots "
    r"\(reuse_buffers=on\)$",
    re.M,
)
_REF_RE = re.compile(r"^\s*std::vector<double>& (\w+) = (tinynn_buf_\d+);", re.M)


def _mlp_chain_graph():
    """Input(8) -> Linear(8x8) -> ReLU -> Linear(8x8) -> ReLU -> Linear(8x8).

    All intermediate buffers have numel 8, so exact-size reuse must kick in.
    Deterministic weights via a seeded generator.
    """
    rng = np.random.default_rng(0)
    b = GraphBuilder()
    x = b.input("x", shape=(8,))
    h = b.linear("l0", x, rng.standard_normal((8, 8)), rng.standard_normal(8))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((8, 8)), rng.standard_normal(8))
    h = b.relu("r1", h)
    y = b.linear("l2", h, rng.standard_normal((8, 8)), rng.standard_normal(8))
    graph = b.build(output_node=y)
    x_val = rng.standard_normal(8)
    return graph, x_val


# --------------------------------------------------------------------------- #
# Chain MLP with repeated widths: correctness + actual reuse happened
# --------------------------------------------------------------------------- #
def test_mlp_chain_reuse_matches_interpreter_and_shares_slots(tmp_path):
    graph, x = _mlp_chain_graph()

    model = compile_graph(graph, tmp_path, options=REUSE)
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    source = model.cpp_path.read_text()

    # 5 non-Input nodes (l0, r0, l1, r1, l2) -> strictly fewer physical slots.
    non_input_nodes = sum(1 for n in graph.nodes if n.op != "Input")
    assert non_input_nodes == 5
    slot_decls = _SLOT_DECL_RE.findall(source)
    assert 0 < len(slot_decls) < non_input_nodes

    # The "buffer plan" header must report P < L, consistent with the decls.
    m = _PLAN_RE.search(source)
    assert m is not None, "missing '// buffer plan:' header line"
    logical, physical = int(m.group(1)), int(m.group(2))
    assert logical == non_input_nodes
    assert physical == len(slot_decls)
    assert physical < logical


# --------------------------------------------------------------------------- #
# Diamond / multi-consumer liveness
# --------------------------------------------------------------------------- #
def test_diamond_multi_consumer_matches_interpreter(tmp_path):
    rng = np.random.default_rng(1)
    b = GraphBuilder()
    x = b.input("x", shape=(6,))
    t = b.tanh("t", x)
    s = b.sigmoid("s", x)
    a = b.add("a", t, s)
    y = b.relu("y", a)
    graph = b.build(output_node=y)

    x_val = rng.standard_normal(6)
    model = compile_graph(graph, tmp_path, options=REUSE)
    actual = model.run(x_val)
    expected = interp_run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# 2D MatMul + never-alias-own-input hazard
# --------------------------------------------------------------------------- #
def test_matmul_2d_matches_interpreter(tmp_path):
    rng = np.random.default_rng(2)
    b = GraphBuilder()
    a = b.input("a", shape=(4, 5))
    c = b.input("c", shape=(5, 6))
    y = b.matmul("mm", a, c)
    graph = b.build(output_node=y)

    inputs = {
        "a": rng.standard_normal((4, 5)),
        "c": rng.standard_normal((5, 6)),
    }
    model = compile_graph(graph, tmp_path, options=REUSE)
    actual = model.run(inputs)
    expected = interp_run(graph, inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Softmax reads its input while writing its output: src must be a
# different physical slot by construction
# --------------------------------------------------------------------------- #
def test_softmax_chain_src_gets_distinct_slot(tmp_path):
    rng = np.random.default_rng(3)
    b = GraphBuilder()
    x = b.input("x", shape=(5,))
    h = b.linear("lin", x, rng.standard_normal((5, 7)), rng.standard_normal(7))
    y = b.softmax("sm", h)
    graph = b.build(output_node=y)

    x_val = rng.standard_normal(5)
    model = compile_graph(graph, tmp_path, options=REUSE)
    actual = model.run(x_val)
    expected = interp_run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    # Both buffers have numel 7 and the same lifetime boundary at the softmax
    # node -- the strictly-greater-than-last-use rule must still keep them in
    # different physical slots (softmax reads src while writing its output).
    source = model.cpp_path.read_text()
    slot_of = dict(_REF_RE.findall(source))
    assert slot_of["v_lin"] != slot_of["v_sm"]


# --------------------------------------------------------------------------- #
# Const operands live as globals, not slots
# --------------------------------------------------------------------------- #
def test_const_operand_not_in_slot_pool(tmp_path):
    rng = np.random.default_rng(4)
    c_val = rng.standard_normal(4)
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    c = b.const("c", c_val)
    a = b.add("a", x, c)
    y = b.relu("y", a)
    graph = b.build(output_node=y)

    x_val = rng.standard_normal(4)
    model = compile_graph(graph, tmp_path, options=REUSE)
    actual = model.run(x_val)
    expected = interp_run(graph, {"x": x_val})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    # The Const stays a global static array; only the two non-Input,
    # non-Const nodes (a, y) get slot references.
    source = model.cpp_path.read_text()
    assert "static const std::vector<double> v_c" in source
    refs = _REF_RE.findall(source)
    assert sorted(ident for ident, _ in refs) == ["v_a", "v_y"]


# --------------------------------------------------------------------------- #
# Benchmark path: repeated computes over reused slots stay correct
# --------------------------------------------------------------------------- #
def test_benchmark_path_with_reuse(tmp_path):
    graph, x = _mlp_chain_graph()
    model = compile_graph(graph, tmp_path, options=REUSE)

    secs = model.benchmark(x, iters=5, warmup=1)
    assert isinstance(secs, float)
    assert secs > 0.0

    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Combined with other codegen options (tiling)
# --------------------------------------------------------------------------- #
def test_reuse_combined_with_tiling(tmp_path):
    graph, x = _mlp_chain_graph()
    model = compile_graph(
        graph, tmp_path, options=CodegenOptions(tile_size=4, reuse_buffers=True)
    )
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Regression guard: default emission has no trace of the slot machinery
# --------------------------------------------------------------------------- #
def test_default_emission_has_no_buffer_slots():
    graph, _ = _mlp_chain_graph()
    assert "tinynn_buf_" not in generate_cpp(graph)
    assert "tinynn_buf_" not in generate_cpp(graph, CodegenOptions())
