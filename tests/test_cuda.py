"""Source and hardware tests for the CUDA backend.

Source-generation tests deliberately run without CUDA. End-to-end execution
tests are marked individually so a machine without nvcc still tests the
optimized graph lowering and generated benchmark structure.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tinynn.cuda import CudaCompiledModel, compile_graph_cuda, generate_cuda
from tinynn.graph import GraphBuilder
from tinynn.interpreter import run as interp_run
from tinynn.ops import FUSED_LINEAR_RELU
from tinynn.passes import default_pipeline


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


CUDA_RUNTIME_AVAILABLE = cuda_runtime_available()
requires_cuda_runtime = pytest.mark.skipif(
    not CUDA_RUNTIME_AVAILABLE,
    reason="CUDA compiler/runtime/GPU not available",
)


def _two_layer_mlp():
    rng = np.random.default_rng(7)
    builder = GraphBuilder()
    x = builder.input("x", shape=(3,))
    h = builder.linear(
        "linear_0",
        x,
        weight=rng.standard_normal((3, 4)),
        bias=rng.standard_normal(4),
    )
    h = builder.relu("relu_0", h)
    y = builder.linear(
        "linear_1",
        h,
        weight=rng.standard_normal((4, 2)),
        bias=rng.standard_normal(2),
    )
    return builder.build(output_node=y), rng.standard_normal(3)


def test_generate_cuda_contains_baseline_kernels_and_io():
    builder = GraphBuilder()
    x = builder.input("x", shape=(3,))
    y = builder.relu("relu_0", x)

    source = generate_cuda(builder.build(output_node=y))

    assert "__global__ void linear_kernel" in source
    assert "__global__ void relu_kernel" in source
    assert "cudaMemcpyDeviceToHost" in source


def test_default_pipeline_mlp_generates_fused_cuda_without_intermediate_buffer():
    graph, _ = _two_layer_mlp()
    optimized = default_pipeline().run(graph)
    source = generate_cuda(optimized)

    assert [node.op for node in optimized.nodes] == [
        "Input",
        FUSED_LINEAR_RELU,
        "Linear",
    ]
    assert "__global__ void fused_linear_relu_kernel" in source
    assert "fused_linear_relu_kernel<<<blocks(4), 256>>>" in source
    assert "float* v_linear_0" not in source
    assert "cudaMalloc(&v_relu_0, 4 * sizeof(float))" in source
    assert source.count("cudaMalloc(&v_relu_0, 4 * sizeof(float))") == 1


def test_generated_benchmark_allocates_and_uploads_before_timing():
    graph, _ = _two_layer_mlp()
    source = generate_cuda(default_pipeline().run(graph))

    assert "cudaEventRecord(start)" in source
    assert "cudaEventElapsedTime" in source
    assert "CUDA_EVENT_SECONDS_PER_ITER=" in source
    assert "for (int i = 0; i < warmup; ++i)" in source
    assert "for (int i = 0; i < iters; ++i)" in source
    assert source.count("cudaMemcpy(d_v_relu_0_w") == 1
    assert source.index("cudaMemcpy(d_v_relu_0_w") < source.index("cudaEventRecord(start)")
    assert source.index("cudaMalloc(&v_relu_0") < source.index("cudaEventRecord(start)")


def test_generate_cuda_rejects_unsupported_optimized_graph_ops():
    builder = GraphBuilder()
    x = builder.input("x", shape=(3,))
    y = builder.tanh("tanh_0", x)

    with pytest.raises(ValueError, match="does not support op 'Tanh'"):
        generate_cuda(builder.build(output_node=y))


def test_cuda_benchmark_parses_device_timing_without_a_gpu(tmp_path, monkeypatch):
    model = CudaCompiledModel(
        cu_path=tmp_path / "model.cu",
        binary_path=tmp_path / "model",
        output_shape=(2,),
        input_specs=(("x", (3,)),),
        work_dir=tmp_path,
    )
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout="CUDA_EVENT_SECONDS_PER_ITER=1.25e-06\n", stderr=""
        )

    monkeypatch.setattr("tinynn.cuda.subprocess.run", fake_run)

    seconds = model.benchmark(np.array([1.0, 2.0, 3.0]), iters=20, warmup=3)

    assert seconds == 1.25e-06
    assert seen["command"][1] == "--benchmark"
    assert seen["command"][-2:] == ["20", "3"]


@requires_cuda_runtime
def test_cuda_default_pipeline_fused_mlp_matches_interpreter(tmp_path):
    graph, x = _two_layer_mlp()
    optimized = default_pipeline().run(graph)
    model = compile_graph_cuda(optimized, tmp_path / "fused_mlp")

    actual = model.run(x)
    expected = interp_run(graph, x)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@requires_cuda_runtime
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


@requires_cuda_runtime
def test_cuda_event_benchmark_returns_positive_time_without_speedup_assertion(tmp_path):
    graph, x = _two_layer_mlp()
    model = compile_graph_cuda(default_pipeline().run(graph), tmp_path / "benchmark")

    seconds = model.benchmark(x, iters=5, warmup=1)

    assert isinstance(seconds, float)
    assert seconds > 0.0
