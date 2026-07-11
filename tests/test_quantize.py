"""Tests for Tier 6 int8 quantized inference (``QuantizedLinear``).

Covers three layers:

    * ``_quantize_symmetric`` (interpreter helper) rounding semantics --
      round-half-away-from-zero, matching C++ ``std::lround``, NOT NumPy's
      round-half-to-even ``np.round``.
    * the :func:`tinynn.passes.quantize_linear` pass (Linear -> QuantizedLinear
      rewrite, and its composability with :func:`tinynn.passes.default_pipeline`).
    * end-to-end exactness between the NumPy reference interpreter and
      generated/compiled C++ for ``QuantizedLinear`` graphs (skipped if
      ``g++`` is not on PATH), plus a float-vs-quantized accuracy sanity
      check (no exact-value assertions -- just a small relative-error bound).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.codegen import CodegenOptions, compile_graph
from tinynn.graph import GraphBuilder
from tinynn.interpreter import _quantize_symmetric, run as interp_run
from tinynn.ops import (
    FUSED_LINEAR_RELU,
    LINEAR,
    QUANTIZED_FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
)
from tinynn.passes import (
    default_pipeline,
    fuse_linear_relu,
    quantize_fused_linear_relu,
    quantize_linear,
)

_needs_gxx = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)


# --------------------------------------------------------------------------- #
# Rounding semantics: round-half-away-from-zero, not NumPy's round-half-even
# --------------------------------------------------------------------------- #
def test_quantize_symmetric_rounding_ties_away_from_zero():
    # scale = 1.0 exactly by construction (max(abs(v)) == 127.0), so every
    # v_i / scale is an exact binary fraction -- no floating-point fuzz.
    v = np.array([127.0, 0.5, 1.5, 2.5, -0.5, -2.5, 0.0])
    q, scale = _quantize_symmetric(v)

    assert scale == 1.0
    # Round-half-AWAY-FROM-ZERO (std::lround): 0.5->1, 1.5->2, 2.5->3,
    # -0.5->-1, -2.5->-3. Round-half-to-EVEN (np.round) would instead give
    # 0.5->0, 1.5->2, 2.5->2, -0.5->0, -2.5->-2 -- deliberately different at
    # four of these five ties, so this pins down the correct rounding rule.
    expected = np.array([127, 1, 2, 3, -1, -3, 0])
    np.testing.assert_array_equal(q, expected)
    assert q.dtype == np.int64

    # Demonstrate this is NOT what np.round would produce (sanity check that
    # the test actually distinguishes the two rounding modes).
    banker_rounded = np.round(v / scale).astype(np.int64)
    assert not np.array_equal(banker_rounded, expected)


def test_quantize_symmetric_all_zero_falls_back_to_unit_scale():
    q, scale = _quantize_symmetric(np.zeros(5))
    assert scale == 1.0
    np.testing.assert_array_equal(q, np.zeros(5, dtype=np.int64))


def test_quantize_symmetric_clips_to_int8_range():
    # scale is derived from the max element, so no legitimate value should
    # ever need clipping in practice -- but assert the invariant holds.
    v = np.array([200.0, -50.0, 3.0])
    q, scale = _quantize_symmetric(v)
    assert np.all(q <= 127) and np.all(q >= -127)
    assert scale == 200.0 / 127.0


# --------------------------------------------------------------------------- #
# quantize_linear pass
# --------------------------------------------------------------------------- #
def _mlp_graph():
    rng = np.random.default_rng(0)
    b = GraphBuilder()
    x = b.input("x", shape=(6,))
    h = b.linear("l0", x, rng.standard_normal((6, 5)), rng.standard_normal(5))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((5, 4)), rng.standard_normal(4))
    y = b.output("y", h)
    return b.build(output_node=y)


def test_quantize_linear_replaces_all_linear_nodes():
    graph = _mlp_graph()
    original_by_name = graph.node_map()

    quantized = quantize_linear(graph)
    q_by_name = quantized.node_map()

    assert quantized.output_node == graph.output_node
    for name, node in original_by_name.items():
        q_node = q_by_name[name]
        assert q_node.name == node.name
        assert q_node.inputs == node.inputs
        assert q_node.shape == node.shape
        if node.op == LINEAR:
            assert q_node.op == QUANTIZED_LINEAR
            np.testing.assert_array_equal(q_node.weight, node.weight)
            np.testing.assert_array_equal(q_node.bias, node.bias)
        else:
            # Everything else (Input, ReLU, Output) is untouched.
            assert q_node.op == node.op

    # Sanity: the quantized graph actually runs.
    x = np.random.default_rng(1).standard_normal(6)
    out = interp_run(quantized, {"x": x})
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_quantize_linear_composes_with_default_pipeline():
    # Input -> Linear -> ReLU (fusable) -> Linear (final, not fused: it feeds
    # the graph output directly, with no trailing ReLU).
    rng = np.random.default_rng(2)
    b = GraphBuilder()
    x = b.input("x", shape=(6,))
    h = b.linear("l0", x, rng.standard_normal((6, 5)), rng.standard_normal(5))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((5, 4)), rng.standard_normal(4))
    graph = b.build(output_node=h)

    fused = default_pipeline().run(graph)
    ops_after_fusion = {n.name: n.op for n in fused.nodes}
    assert FUSED_LINEAR_RELU in ops_after_fusion.values()
    assert LINEAR in ops_after_fusion.values()  # l1 (final) stays a plain Linear

    quantized = quantize_linear(fused)
    ops_after_quantize = {n.name: n.op for n in quantized.nodes}

    # FusedLinearReLU nodes are left alone by quantize_linear...
    assert FUSED_LINEAR_RELU in ops_after_quantize.values()
    # ...while remaining Linear nodes are quantized, and no plain Linear
    # survives.
    assert LINEAR not in ops_after_quantize.values()
    assert QUANTIZED_LINEAR in ops_after_quantize.values()

    x_val = rng.standard_normal(6)
    out = interp_run(quantized, {"x": x_val})
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# Accuracy: quantized interpreter vs float Linear interpreter (no exact
# value assertions -- only a small relative-error bound).
# --------------------------------------------------------------------------- #
def test_quantized_accuracy_close_to_float_reference():
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((16, 8))  # unit-scale (standard normal) weights
    bias = rng.standard_normal(8)
    x = rng.standard_normal(16)

    bf = GraphBuilder()
    xf = bf.input("x", shape=(16,))
    yf = bf.linear("lin", xf, weight, bias)
    float_graph = bf.build(output_node=yf)

    bq = GraphBuilder()
    xq = bq.input("x", shape=(16,))
    yq = bq.quantized_linear("qlin", xq, weight, bias)
    quant_graph = bq.build(output_node=yq)

    f = interp_run(float_graph, {"x": x})
    q = interp_run(quant_graph, {"x": x})

    rel_err = np.max(np.abs(q - f)) / (np.max(np.abs(f)) + 1e-12)
    assert rel_err < 0.02


# --------------------------------------------------------------------------- #
# End-to-end exactness: compiled C++ vs NumPy interpreter (skipped if no g++)
# --------------------------------------------------------------------------- #
@_needs_gxx
def test_quantized_linear_cpp_matches_interpreter(tmp_path):
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((10, 6)) * 3.0
    bias = rng.standard_normal(6)
    x = rng.standard_normal(10) * 2.0

    b = GraphBuilder()
    xn = b.input("x", shape=(10,))
    y = b.quantized_linear("qlin", xn, weight, bias)
    graph = b.build(output_node=y)

    model = compile_graph(graph, tmp_path / "case_single")
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@_needs_gxx
def test_quantized_chain_cpp_matches_interpreter(tmp_path):
    rng = np.random.default_rng(8)
    w0 = rng.standard_normal((12, 9)) * 1.5
    b0 = rng.standard_normal(9)
    w1 = rng.standard_normal((9, 5)) * 2.0
    b1 = rng.standard_normal(5)
    x = rng.standard_normal(12)

    b = GraphBuilder()
    xn = b.input("x", shape=(12,))
    h = b.quantized_linear("qlin0", xn, w0, b0)
    h = b.relu("relu0", h)
    y = b.quantized_linear("qlin1", h, w1, b1)
    graph = b.build(output_node=y)

    model = compile_graph(graph, tmp_path / "case_chain")
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@_needs_gxx
def test_quantized_zero_weight_and_zero_input_edge_cases(tmp_path):
    rng = np.random.default_rng(9)

    # All-zero weight: s_w must fall back to 1.0, output must be exactly bias.
    zero_w = np.zeros((5, 4))
    bias0 = rng.standard_normal(4)
    x_nonzero = rng.standard_normal(5) * 3.0

    b1 = GraphBuilder()
    xn1 = b1.input("x", shape=(5,))
    y1 = b1.quantized_linear("qlin", xn1, zero_w, bias0)
    graph1 = b1.build(output_node=y1)

    model1 = compile_graph(graph1, tmp_path / "case_zero_weight")
    actual1 = model1.run(x_nonzero)
    expected1 = interp_run(graph1, {"x": x_nonzero})
    assert np.all(np.isfinite(actual1))
    np.testing.assert_allclose(actual1, expected1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual1, bias0, rtol=1e-12, atol=1e-12)

    # All-zero input: s_x must fall back to 1.0, output must be exactly bias.
    weight_nonzero = rng.standard_normal((5, 4)) * 2.0
    bias1 = rng.standard_normal(4)
    x_zero = np.zeros(5)

    b2 = GraphBuilder()
    xn2 = b2.input("x", shape=(5,))
    y2 = b2.quantized_linear("qlin", xn2, weight_nonzero, bias1)
    graph2 = b2.build(output_node=y2)

    model2 = compile_graph(graph2, tmp_path / "case_zero_input")
    actual2 = model2.run(x_zero)
    expected2 = interp_run(graph2, {"x": x_zero})
    assert np.all(np.isfinite(actual2))
    np.testing.assert_allclose(actual2, expected2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual2, bias1, rtol=1e-12, atol=1e-12)


@_needs_gxx
def test_quantized_chain_with_reuse_buffers_matches_interpreter(tmp_path):
    # Same numel (8) throughout so the buffer-reuse slot planner has a real
    # opportunity to alias buffers; the int8 scratch buffers must stay out
    # of that pool regardless.
    rng = np.random.default_rng(11)
    w0 = rng.standard_normal((8, 8))
    b0 = rng.standard_normal(8)
    w1 = rng.standard_normal((8, 8))
    b1 = rng.standard_normal(8)
    x = rng.standard_normal(8)

    b = GraphBuilder()
    xn = b.input("x", shape=(8,))
    h = b.quantized_linear("qlin0", xn, w0, b0)
    h = b.relu("relu0", h)
    y = b.quantized_linear("qlin1", h, w1, b1)
    graph = b.build(output_node=y)

    model = compile_graph(
        graph, tmp_path / "case_reuse", options=CodegenOptions(reuse_buffers=True)
    )
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    source = model.cpp_path.read_text()
    assert "_xq" in source
    assert "tinynn_buf_" in source


# --------------------------------------------------------------------------- #
# QuantizedFusedLinearReLU: pass, semantics, and end-to-end exactness
# --------------------------------------------------------------------------- #
def _linear_relu_chain_graph(seed=3):
    """Input -> Linear -> ReLU -> Linear -> ReLU -> Linear (final, unfused)."""
    rng = np.random.default_rng(seed)
    b = GraphBuilder()
    x = b.input("x", shape=(12,))
    h = b.linear("l0", x, rng.standard_normal((12, 10)), rng.standard_normal(10))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((10, 8)), rng.standard_normal(8))
    h = b.relu("r1", h)
    h = b.linear("l2", h, rng.standard_normal((8, 4)), rng.standard_normal(4))
    return b.build(output_node=h)


def test_quantize_fused_linear_relu_pass_replaces_fused_nodes():
    graph = fuse_linear_relu(_linear_relu_chain_graph())
    fused_names = {n.name for n in graph.nodes if n.op == FUSED_LINEAR_RELU}
    assert fused_names  # sanity: fusion actually happened

    quantized = quantize_fused_linear_relu(graph)
    q_by_name = quantized.node_map()

    assert quantized.output_node == graph.output_node
    for node in graph.nodes:
        q_node = q_by_name[node.name]
        assert q_node.inputs == node.inputs
        assert q_node.shape == node.shape
        if node.op == FUSED_LINEAR_RELU:
            assert q_node.op == QUANTIZED_FUSED_LINEAR_RELU
            np.testing.assert_array_equal(q_node.weight, node.weight)
            np.testing.assert_array_equal(q_node.bias, node.bias)
        else:
            # Everything else (Input, the final unfused Linear) is untouched.
            assert q_node.op == node.op


def test_fuse_then_quantize_lowers_linear_relu_to_quantized_fused():
    # The full opt-in int8 lowering: fuse Linear+ReLU pairs, quantize the
    # fused pairs, quantize the remaining bare Linears. No float Linear,
    # ReLU, or FusedLinearReLU survives.
    graph = _linear_relu_chain_graph()
    lowered = quantize_linear(quantize_fused_linear_relu(fuse_linear_relu(graph)))
    ops = [n.op for n in lowered.nodes]

    assert ops.count(QUANTIZED_FUSED_LINEAR_RELU) == 2
    assert ops.count(QUANTIZED_LINEAR) == 1
    assert LINEAR not in ops and FUSED_LINEAR_RELU not in ops
    assert "ReLU" not in ops

    x = np.random.default_rng(4).standard_normal(12)
    out = interp_run(lowered, {"x": x})
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_quantized_fused_equals_relu_of_quantized_linear():
    # Pins down the documented op order: the ReLU clamp is applied LAST,
    # after the rescale + bias add -- so QuantizedFusedLinearReLU must equal
    # max(0, QuantizedLinear) EXACTLY (same weights, same input), including
    # cases where the pre-activation is negative.
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((14, 9)) * 2.0
    bias = rng.standard_normal(9) - 1.0  # push some pre-activations negative
    x = rng.standard_normal(14)

    bq = GraphBuilder()
    xq = bq.input("x", shape=(14,))
    yq = bq.quantized_linear("qlin", xq, weight, bias)
    q = interp_run(bq.build(output_node=yq), {"x": x})

    bf = GraphBuilder()
    xf = bf.input("x", shape=(14,))
    yf = bf.quantized_fused_linear_relu("qflr", xf, weight, bias)
    f = interp_run(bf.build(output_node=yf), {"x": x})

    np.testing.assert_array_equal(f, np.maximum(q, 0.0))
    assert np.any(np.maximum(q, 0.0) == 0.0)  # sanity: the clamp actually bit


def test_quantized_fused_accuracy_close_to_float_fused():
    rng = np.random.default_rng(43)
    weight = rng.standard_normal((16, 8))
    bias = rng.standard_normal(8)
    x = rng.standard_normal(16)

    bf = GraphBuilder()
    xf = bf.input("x", shape=(16,))
    yf = bf.fused_linear_relu("flr", xf, weight, bias)
    f = interp_run(bf.build(output_node=yf), {"x": x})

    bq = GraphBuilder()
    xq = bq.input("x", shape=(16,))
    yq = bq.quantized_fused_linear_relu("qflr", xq, weight, bias)
    q = interp_run(bq.build(output_node=yq), {"x": x})

    rel_err = np.max(np.abs(q - f)) / (np.max(np.abs(f)) + 1e-12)
    assert rel_err < 0.02


@_needs_gxx
def test_quantized_fused_cpp_matches_interpreter(tmp_path):
    rng = np.random.default_rng(12)
    weight = rng.standard_normal((10, 6)) * 3.0
    bias = rng.standard_normal(6) - 0.5
    x = rng.standard_normal(10) * 2.0

    b = GraphBuilder()
    xn = b.input("x", shape=(10,))
    y = b.quantized_fused_linear_relu("qflr", xn, weight, bias)
    graph = b.build(output_node=y)

    model = compile_graph(graph, tmp_path / "case_fused_single")
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    assert np.any(expected == 0.0)  # sanity: the clamp actually bit
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@_needs_gxx
def test_lowered_linear_relu_chain_cpp_matches_interpreter(tmp_path):
    # Targeted end-to-end test for the Linear->ReLU quantized fused path:
    # several random graphs, each lowered via fuse -> quantize_fused ->
    # quantize_linear, compiled, and checked against both the quantized
    # interpreter (tight tolerance) and the float interpreter (loose bound).
    for seed in range(5):
        graph = _linear_relu_chain_graph(seed=seed)
        lowered = quantize_linear(
            quantize_fused_linear_relu(fuse_linear_relu(graph))
        )
        assert QUANTIZED_FUSED_LINEAR_RELU in {n.op for n in lowered.nodes}

        x = np.random.default_rng(100 + seed).standard_normal(12)
        expected = interp_run(lowered, {"x": x})
        model = compile_graph(lowered, tmp_path / f"case_lowered_{seed}")
        actual = model.run(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        f = interp_run(graph, {"x": x})
        rel_err = np.max(np.abs(expected - f)) / (np.max(np.abs(f)) + 1e-12)
        assert rel_err < 0.25  # multi-layer int8 error compounds; loose bound


@_needs_gxx
def test_quantized_fused_chain_with_reuse_buffers_matches_interpreter(tmp_path):
    rng = np.random.default_rng(13)
    w0 = rng.standard_normal((8, 8))
    b0 = rng.standard_normal(8)
    w1 = rng.standard_normal((8, 8))
    b1 = rng.standard_normal(8)
    x = rng.standard_normal(8)

    b = GraphBuilder()
    xn = b.input("x", shape=(8,))
    h = b.quantized_fused_linear_relu("qflr0", xn, w0, b0)
    y = b.quantized_fused_linear_relu("qflr1", h, w1, b1)
    graph = b.build(output_node=y)

    model = compile_graph(
        graph, tmp_path / "case_fused_reuse", options=CodegenOptions(reuse_buffers=True)
    )
    actual = model.run(x)
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
