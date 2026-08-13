"""Reproducible steady-state CUDA benchmarks for supported TinyNN MLPs.

Run on an NVIDIA host after installing the optional development dependencies:

    python -m tinynn.cuda_benchmark --results-dir results/cuda

The timed value is CUDA-event device time per inference. It intentionally
excludes process startup, file I/O, one-time allocations, and host/device
transfers. Each workload is first checked against the NumPy interpreter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .cuda import compile_graph_cuda
from .graph import Graph, GraphBuilder, INPUT
from .interpreter import run as interp_run
from .onnx_import import import_onnx
from .passes import default_pipeline


def build_mlp_256() -> Tuple[Graph, np.ndarray]:
    """Build a deterministic 256 -> 512 -> 512 -> 10 MLP workload."""
    rng = np.random.default_rng(20260813)
    builder = GraphBuilder()
    x = builder.input("x", (256,))
    h = builder.linear(
        "linear_0",
        x,
        rng.standard_normal((256, 512)),
        rng.standard_normal(512),
    )
    h = builder.relu("relu_0", h)
    h = builder.linear(
        "linear_1",
        h,
        rng.standard_normal((512, 512)),
        rng.standard_normal(512),
    )
    h = builder.relu("relu_1", h)
    y = builder.linear(
        "linear_2",
        h,
        rng.standard_normal((512, 10)),
        rng.standard_normal(10),
    )
    return builder.build(output_node=y), rng.standard_normal(256)


def _single_input(graph: Graph, seed: int) -> np.ndarray:
    input_nodes = [node for node in graph.nodes if node.op == INPUT]
    if len(input_nodes) != 1:
        raise ValueError(
            "CUDA benchmark harness requires exactly one Input node, got "
            f"{[node.name for node in input_nodes]}"
        )
    return np.random.default_rng(seed).standard_normal(input_nodes[0].shape)


def benchmark_cuda_graph(
    graph: Graph,
    inputs: np.ndarray,
    *,
    label: str,
    results_dir: Path,
    iters: int,
    warmup: int,
    source_model: str,
) -> Dict[str, object]:
    """Verify an optimized graph, time it, and write JSON/Markdown results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    optimized = default_pipeline().run(graph)
    model = compile_graph_cuda(optimized, results_dir / "build" / label)

    expected = interp_run(graph, inputs)
    actual = model.run(inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    seconds = model.benchmark(inputs, iters=iters, warmup=warmup)

    result: Dict[str, object] = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_model": source_model,
        "graph": {
            "num_nodes": len(optimized.nodes),
            "ops": dict(Counter(node.op for node in optimized.nodes)),
            "input_shape": list(inputs.shape),
            "output_shape": list(optimized.get_node(optimized.output_node).shape),
        },
        "correctness_verified_against_numpy": True,
        "oracle_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "iters": iters,
        "warmup": warmup,
        "timing": {
            "cuda_event_seconds_per_inference": seconds,
            "scope": "device kernel execution only",
            "excluded": [
                "process startup",
                "file I/O",
                "one-time device allocation",
                "host-to-device input transfer",
                "one-time parameter upload",
                "device-to-host output transfer",
            ],
        },
    }

    json_path = results_dir / f"{label}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path = results_dir / f"{label}.md"
    md_path.write_text(
        "\n".join(
            [
                f"# CUDA Benchmark: {label}",
                "",
                f"- when: {result['timestamp']}",
                f"- source model: {source_model}",
                f"- optimized graph: {result['graph']['num_nodes']} nodes, "
                f"ops {result['graph']['ops']}",
                f"- correctness: NumPy oracle passed at rtol=1e-5, atol=1e-6",
                f"- warmup: {warmup}; timed iterations: {iters}",
                "- timing: CUDA events around device kernel execution only",
                "- excluded: process startup, file I/O, one-time allocation, "
                "input/parameter upload, and output download",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| CUDA-event seconds/inference | {seconds:.9g} |",
                "",
                "This is not a cross-framework speedup claim. Compare only "
                "runs with the same workload, GPU, driver, compiler, and "
                "benchmark settings.",
                "",
            ]
        )
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify and benchmark TinyNN CUDA MLPs with CUDA events."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cuda"),
        help="directory for generated source, binaries, and reports",
    )
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument(
        "--digits-model",
        type=Path,
        default=Path("examples/models/digits_mlp.onnx"),
        help="existing MLP-style ONNX model to benchmark after mlp_256",
    )
    parser.add_argument(
        "--skip-digits",
        action="store_true",
        help="benchmark only the deterministic mlp_256 workload",
    )
    args = parser.parse_args(argv)
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")

    graph, inputs = build_mlp_256()
    workloads = [("mlp_256", graph, inputs, "deterministic 256->512->512->10 MLP")]
    if not args.skip_digits:
        if not args.digits_model.exists():
            parser.error(f"digits model does not exist: {args.digits_model}")
        digits_graph = import_onnx(args.digits_model)
        workloads.append(
            (
                "digits_mlp",
                digits_graph,
                _single_input(digits_graph, seed=20260814),
                str(args.digits_model),
            )
        )

    for label, workload_graph, workload_inputs, source_model in workloads:
        result = benchmark_cuda_graph(
            workload_graph,
            workload_inputs,
            label=label,
            results_dir=args.results_dir,
            iters=args.iters,
            warmup=args.warmup,
            source_model=source_model,
        )
        seconds = result["timing"]["cuda_event_seconds_per_inference"]
        print(f"{label}: {seconds:.9g} CUDA-event seconds/inference")


if __name__ == "__main__":
    main()
