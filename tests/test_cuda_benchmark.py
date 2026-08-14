"""CPU-only structural tests for the CUDA benchmark harness."""

from __future__ import annotations

import numpy as np

from tinynn.cuda import generate_cuda
from tinynn.cuda_benchmark import build_mlp_256
from tinynn.ops import FUSED_LINEAR_RELU
from tinynn.passes import default_pipeline


def test_mlp_256_is_deterministic_and_default_pipeline_fuses_hidden_layers():
    graph_a, input_a = build_mlp_256()
    graph_b, input_b = build_mlp_256()

    np.testing.assert_allclose(input_a, input_b)
    np.testing.assert_allclose(graph_a.nodes[1].weight, graph_b.nodes[1].weight)

    optimized = default_pipeline().run(graph_a)
    assert [node.op for node in optimized.nodes].count(FUSED_LINEAR_RELU) == 2
    source = generate_cuda(optimized)
    assert source.count("fused_linear_relu_kernel<<<") >= 2
    assert "cudaEventElapsedTime" in source
