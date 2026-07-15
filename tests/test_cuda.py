"""End-to-end tests for the minimal Tier 5 CUDA backend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tinynn.cuda import compile_graph_cuda, generate_cuda
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run as interp_run


def cuda_toolchain_available() -> bool:
    """Return True only if ``nvcc`` exists and runs."""
    if shutil.which("nvcc") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def cuda_runtime_available() -> bool:
    """Return True only if a compiled CUDA program can initialize the runtime."""
    if not cuda_toolchain_available():
        return False

    source = """
#include <cuda_runtime.h>
int main() {
    cudaError_t err = cudaFree(0);
    return err == cudaSuccess ? 0 : 1;
}
"""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cu_path = tmp_path / "cuda_probe.cu"
            bin_path = tmp_path / "cuda_probe"
            cu_path.write_text(source)
            compile_proc = subprocess.run(
                ["nvcc", str(cu_path), "-o", str(bin_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if compile_proc.returncode != 0:
                return False
            run_proc = subprocess.run(
                [str(bin_path)], capture_output=True, text=True, timeout=30
            )
            return run_proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not cuda_runtime_available(),
    reason="CUDA compiler/runtime/GPU not available",
)


def test_generate_cuda_contains_kernels():
    builder = GraphBuilder()
    x = builder.input("x", shape=(3,))
    y = builder.relu("relu_0", x)
    graph = builder.build(output_node=y)

    source = generate_cuda(graph)

    assert "__global__ void relu_kernel" in source
    assert "cudaMalloc" in source
    assert "cudaMemcpyDeviceToHost" in source


def test_cuda_input_linear_relu_matches_interpreter(tmp_path):
    x = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    w = np.array(
        [
            [0.5, -1.0, 2.0, 0.0],
            [1.5, 0.25, -0.5, 1.0],
            [-1.0, 0.75, 0.5, -2.0],
        ],
        dtype=np.float32,
    )
    b = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(3,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    r0 = builder.relu("relu_0", l0)
    graph = builder.build(output_node=r0)

    model = compile_graph_cuda(graph, tmp_path / "linear_relu")
    actual = model.run(x)
    expected = interp_run(graph, {"x": x.astype(np.float64)})

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_cuda_explicit_output_node_matches_interpreter(tmp_path):
    x = np.array([2.0, -1.0], dtype=np.float32)
    w = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float32)
    b = np.array([0.0, 0.25, -0.75], dtype=np.float32)

    builder = GraphBuilder()
    xn = builder.input("x", shape=(2,))
    l0 = builder.linear("linear_0", xn, weight=w, bias=b)
    out = builder.output("y", l0)
    graph = builder.build(output_node=out)

    model = compile_graph_cuda(graph, tmp_path / "output_node")
    actual = model.run(x)
    expected = interp_run(graph, x.astype(np.float64))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
