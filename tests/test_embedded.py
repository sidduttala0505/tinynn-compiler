from __future__ import annotations

import shutil

import numpy as np
import pytest

from tinynn.codegen import compile_graph
from tinynn.embedded import compile_embedded, generate_embedded_c
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run as interp_run

_HAS_CC = shutil.which("cc") is not None
_HAS_GXX = shutil.which("g++") is not None

_skip_no_cc = pytest.mark.skipif(_HAS_CC is False, reason="cc not found on PATH")
_skip_no_gxx = pytest.mark.skipif(_HAS_GXX is False, reason="g++ not found on PATH")


# --------------------------------------------------------------------------- #
# Case 1: float MLP (Input -> Linear -> ReLU -> Linear -> Output)
# --------------------------------------------------------------------------- #
@_skip_no_cc
def test_float_mlp_matches_interpreter(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    w0 = rng.standard_normal((16, 12))
    b0 = rng.standard_normal(12)
    w1 = rng.standard_normal((12, 4))
    b1 = rng.standard_normal(4)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(16,))
    l0 = builder.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = builder.relu("relu_0", l0)
    l1 = builder.linear("linear_1", r0, weight=w1, bias=b1)
    out = builder.output("y", l1)
    graph = builder.build(output_node=out)

    model = compile_embedded(graph, tmp_path / "case1")
    actual = model.run(x)

    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 2: fused float model (FusedLinearReLU chain)
# --------------------------------------------------------------------------- #
@_skip_no_cc
def test_fused_linear_relu_matches_interpreter(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10)
    w0 = rng.standard_normal((10, 8))
    b0 = rng.standard_normal(8)
    w1 = rng.standard_normal((8, 3))
    b1 = rng.standard_normal(3)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(10,))
    f0 = builder.fused_linear_relu("fused_0", xn, weight=w0, bias=b0)
    f1 = builder.fused_linear_relu("fused_1", f0, weight=w1, bias=b1)
    out = builder.output("y", f1)
    graph = builder.build(output_node=out)

    model = compile_embedded(graph, tmp_path / "case2")
    actual = model.run(x)

    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 3: quantized model (QuantizedFusedLinearReLU -> QuantizedLinear)
# --------------------------------------------------------------------------- #
def _build_quantized_graph(rng):
    x = rng.standard_normal(20)
    w0 = rng.standard_normal((20, 14))
    b0 = rng.standard_normal(14)
    w1 = rng.standard_normal((14, 5))
    b1 = rng.standard_normal(5)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(20,))
    q0 = builder.quantized_fused_linear_relu("qfused_0", xn, weight=w0, bias=b0)
    q1 = builder.quantized_linear("qlinear_0", q0, weight=w1, bias=b1)
    out = builder.output("y", q1)
    graph = builder.build(output_node=out)
    return graph, x


@_skip_no_cc
def test_quantized_model_matches_interpreter(tmp_path):
    rng = np.random.default_rng(2)
    graph, x = _build_quantized_graph(rng)

    model = compile_embedded(graph, tmp_path / "case3")
    actual = model.run(x)

    # Both sides quantize identically (same deterministic weight-quantization
    # formula, same dynamic activation quantization), so agreement is
    # near-exact -- the C++ backend achieves ~1e-15, this C99 backend should
    # match at the same tolerance.
    expected = interp_run(graph, {"x": x})
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@_skip_no_cc
@_skip_no_gxx
def test_quantized_model_matches_cpp_backend(tmp_path):
    # The two backends (C99 static-memory vs C++ std::vector) must agree
    # with each other, not just with the interpreter independently.
    rng = np.random.default_rng(3)
    graph, x = _build_quantized_graph(rng)

    embedded_model = compile_embedded(graph, tmp_path / "case3_embedded")
    cpp_model = compile_graph(graph, tmp_path / "case3_cpp")

    embedded_actual = embedded_model.run(x)
    cpp_actual = cpp_model.run(x)

    np.testing.assert_allclose(embedded_actual, cpp_actual, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Case 4: source inspection -- static memory, no dynamic allocation, no C++
# --------------------------------------------------------------------------- #
def test_generate_embedded_c_source_has_no_dynamic_allocation_or_cpp():
    rng = np.random.default_rng(4)
    graph, _x = _build_quantized_graph(rng)

    source = generate_embedded_c(graph)

    assert "std::vector" not in source
    assert "malloc" not in source
    assert "new " not in source
    assert "#include <vector>" not in source

    assert "static double" in source
    assert "static const int8_t" in source
    assert "int8_t" in source
    assert "BENCH_NS" in source


# --------------------------------------------------------------------------- #
# Case 5: benchmark smoke test
# --------------------------------------------------------------------------- #
@_skip_no_cc
def test_benchmark_returns_positive_float(tmp_path):
    # Sized and iterated generously enough that the timed region reliably
    # clears the host's clock_gettime resolution -- a too-tiny workload can
    # legitimately round to 0ns on some platforms even though the code is
    # correct, which would make this smoke test flaky rather than wrong.
    rng = np.random.default_rng(5)
    x = rng.standard_normal(256)
    w0 = rng.standard_normal((256, 256))
    b0 = rng.standard_normal(256)
    w1 = rng.standard_normal((256, 64))
    b1 = rng.standard_normal(64)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(256,))
    l0 = builder.linear("linear_0", xn, weight=w0, bias=b0)
    r0 = builder.relu("relu_0", l0)
    l1 = builder.linear("linear_1", r0, weight=w1, bias=b1)
    out = builder.output("y", l1)
    graph = builder.build(output_node=out)

    model = compile_embedded(graph, tmp_path / "case5")
    seconds_per_iter = model.benchmark(x, iters=200, warmup=5)

    assert isinstance(seconds_per_iter, float)
    assert seconds_per_iter > 0.0


# --------------------------------------------------------------------------- #
# Case 6: unsupported op raises ValueError naming the op
# --------------------------------------------------------------------------- #
def test_unsupported_op_raises_value_error():
    builder = GraphBuilder()
    xn = builder.input("x", shape=(4,))
    sm = builder.softmax("softmax_0", xn)
    graph = builder.build(output_node=sm)

    with pytest.raises(ValueError, match="Softmax"):
        generate_embedded_c(graph)
