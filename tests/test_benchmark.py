"""Tests for CodegenOptions (Tier 4) and the benchmark harness.

Correctness assertions only: benchmark *timings* are machine-dependent, so
tests check structure and positivity, never speedup values.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from tinynn.benchmark import benchmark_graph
from tinynn.codegen import CodegenOptions, compile_graph, openmp_available
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not found on PATH"
)


def _mlp_graph():
    rng = np.random.default_rng(42)
    b = GraphBuilder()
    x = b.input("x", (8,))
    h = b.linear("l0", x, rng.standard_normal((8, 16)), rng.standard_normal(16))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((16, 4)), rng.standard_normal(4))
    return b.build(output_node=h), rng.standard_normal(8)


def _matmul_graph():
    rng = np.random.default_rng(7)
    b = GraphBuilder()
    a = b.input("a", (3, 4))
    c = b.input("c", (4, 5))
    mm = b.matmul("mm", a, c)
    graph = b.build(output_node=mm)
    inputs = {"a": rng.standard_normal((3, 4)), "c": rng.standard_normal((4, 5))}
    return graph, inputs


# --------------------------------------------------------------------------- #
# CodegenOptions correctness vs interpreter
# --------------------------------------------------------------------------- #
def test_fast_options_mlp_matches_interpreter(tmp_path):
    graph, x = _mlp_graph()
    model = compile_graph(graph, tmp_path, options=CodegenOptions.fast())
    np.testing.assert_allclose(model.run(x), run(graph, x), rtol=1e-9, atol=1e-9)


def test_fast_options_matmul_matches_interpreter(tmp_path):
    graph, inputs = _matmul_graph()
    model = compile_graph(graph, tmp_path, options=CodegenOptions.fast())
    np.testing.assert_allclose(
        model.run(inputs), run(graph, inputs), rtol=1e-9, atol=1e-9
    )


def test_tile_smaller_than_dims_matches_interpreter(tmp_path):
    # tile_size=2 is smaller than every dimension, exercising remainder tiles.
    graph, inputs = _matmul_graph()
    model = compile_graph(graph, tmp_path, options=CodegenOptions(tile_size=2))
    np.testing.assert_allclose(
        model.run(inputs), run(graph, inputs), rtol=1e-9, atol=1e-9
    )


@pytest.mark.skipif(not openmp_available(), reason="compiler lacks -fopenmp")
def test_openmp_options_match_interpreter(tmp_path):
    graph, x = _mlp_graph()
    model = compile_graph(
        graph, tmp_path, options=CodegenOptions(tile_size=8, openmp=True)
    )
    np.testing.assert_allclose(model.run(x), run(graph, x), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Repeat/bench mode of the compiled binary
# --------------------------------------------------------------------------- #
def test_compiled_model_benchmark_returns_positive_seconds(tmp_path):
    graph, x = _mlp_graph()
    model = compile_graph(graph, tmp_path)
    seconds = model.benchmark(x, iters=5, warmup=1)
    assert isinstance(seconds, float) and seconds > 0
    # Default invocation (run) still works and matches the oracle.
    np.testing.assert_allclose(model.run(x), run(graph, x), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# benchmark_graph harness
# --------------------------------------------------------------------------- #
def test_benchmark_graph_writes_structured_results(tmp_path):
    graph, x = _mlp_graph()
    result = benchmark_graph(
        graph, x, iters=5, warmup=1, results_dir=tmp_path, label="tiny"
    )

    assert result["correctness_verified"] is True
    timing = result["timing_s_per_iter"]
    for key in ("interpreter", "cpu_naive", "cpu_optimized"):
        assert timing[key] > 0
    assert set(result["speedup"]) == {
        "naive_vs_interpreter",
        "optimized_vs_naive",
        "optimized_vs_interpreter",
    }

    json_path = tmp_path / "tiny.json"
    md_path = tmp_path / "tiny.md"
    assert json_path.exists() and md_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["label"] == "tiny"
    assert set(on_disk) == set(result)
    assert "| variant | seconds/iter |" in md_path.read_text()
