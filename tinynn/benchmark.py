"""Benchmarking: interpreter vs naive C++ vs optimized C++.

We check every compiled version against the interpreter first and bail if they
disagree - no point timing a wrong answer. The C++ timing happens inside the
binary (so we're not measuring process startup); the interpreter is timed here
with perf_counter. Numbers depend a lot on the machine, so don't read too much
into them.

Run it with:  ./venv/bin/python -m tinynn.benchmark
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .codegen import CodegenOptions, compile_graph, openmp_available
from .graph import Graph, GraphBuilder, INPUT
from .interpreter import run
from .passes import default_pipeline

__all__ = ["benchmark_graph"]

Inputs = Union[Dict[str, np.ndarray], np.ndarray]

# how close the compiled output has to be to the interpreter before we time it
_RTOL = _ATOL = 1e-9


def _verify(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
        raise AssertionError(
            f"{label} output disagrees with the interpreter oracle "
            f"(max abs diff {np.max(np.abs(actual - expected))}); "
            "refusing to benchmark an incorrect binary"
        )


def _time_interpreter(graph: Graph, inputs: Inputs, iters: int) -> float:
    for _ in range(3):  # warm up first
        run(graph, inputs)
    start = time.perf_counter()
    for _ in range(iters):
        run(graph, inputs)
    return (time.perf_counter() - start) / iters


def benchmark_graph(
    graph: Graph,
    inputs: Inputs,
    iters: int = 200,
    warmup: int = 20,
    results_dir: Union[str, Path] = "results",
    label: str = "benchmark",
) -> Dict[str, object]:
    """Benchmark a graph three ways: interpreter, plain C++, optimized C++.

    "Optimized" = run the default pipeline first, then compile with
    CodegenOptions.fast() (and OpenMP if we have it). Checks correctness first,
    writes <label>.json and <label>.md into results_dir, and returns the dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_ref = run(graph, inputs)

    naive = compile_graph(graph, results_dir / "build" / f"{label}_naive")
    _verify("naive compiled", naive.run(inputs), y_ref)

    optimized_graph = default_pipeline().run(graph)
    use_openmp = openmp_available()
    opt = compile_graph(
        optimized_graph,
        results_dir / "build" / f"{label}_opt",
        options=CodegenOptions.fast(openmp=use_openmp),
    )
    _verify("optimized compiled", opt.run(inputs), y_ref)

    interp_s = _time_interpreter(graph, inputs, min(iters, 50))
    naive_s = naive.benchmark(inputs, iters=iters, warmup=warmup)
    opt_s = opt.benchmark(inputs, iters=iters, warmup=warmup)

    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    result: Dict[str, object] = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "graph": {
            "num_nodes": len(graph.nodes),
            "ops": dict(Counter(n.op for n in graph.nodes)),
            "input_shapes": {n.name: list(n.shape) for n in input_nodes},
            "output_shape": list(graph.get_node(graph.output_node).shape),
        },
        "iters": iters,
        "warmup": warmup,
        "openmp_used": use_openmp,
        "correctness_verified": True,
        "timing_s_per_iter": {
            "interpreter": interp_s,
            "cpu_naive": naive_s,
            "cpu_optimized": opt_s,
        },
        "speedup": {
            "naive_vs_interpreter": interp_s / naive_s,
            "optimized_vs_naive": naive_s / opt_s,
            "optimized_vs_interpreter": interp_s / opt_s,
        },
    }

    json_path = results_dir / f"{label}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    md_lines = [
        f"# Benchmark: {label}",
        "",
        f"- when: {result['timestamp']}",
        f"- graph: {result['graph']['num_nodes']} nodes, ops {result['graph']['ops']}",
        f"- iters: {iters} (warmup {warmup}), OpenMP: {use_openmp}",
        "- correctness: all compiled variants verified against the interpreter"
        f" (rtol/atol {_RTOL:g}) before timing",
        "",
        "| variant | seconds/iter | vs interpreter |",
        "|---|---|---|",
        f"| interpreter (NumPy) | {interp_s:.3e} | 1.0x |",
        f"| C++ naive (-O2) | {naive_s:.3e} | {interp_s / naive_s:.2f}x |",
        f"| C++ optimized (passes + fast codegen) | {opt_s:.3e} "
        f"| {interp_s / opt_s:.2f}x |",
        "",
        f"Optimized vs naive: {naive_s / opt_s:.2f}x. Numbers are machine- and"
        " load-dependent; treat as indicative only. Note that the NumPy"
        " interpreter delegates matmul-heavy work to an optimized BLAS"
        " (e.g. Apple Accelerate), so it can outrun TinyNN's scalar C++"
        " loops; the meaningful codegen comparison is optimized vs naive.",
        "",
    ]
    (results_dir / f"{label}.md").write_text("\n".join(md_lines))

    return result


# a couple of example workloads to run when called directly
def _mlp_256() -> Dict[str, object]:
    rng = np.random.default_rng(0)
    b = GraphBuilder()
    x = b.input("x", (256,))
    h = b.linear("l0", x, rng.standard_normal((256, 512)), rng.standard_normal(512))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((512, 512)), rng.standard_normal(512))
    h = b.relu("r1", h)
    h = b.linear("l2", h, rng.standard_normal((512, 10)), rng.standard_normal(10))
    graph = b.build(output_node=h)
    return benchmark_graph(graph, rng.standard_normal(256), label="mlp_256")


def _matmul_128() -> Dict[str, object]:
    rng = np.random.default_rng(1)
    b = GraphBuilder()
    a = b.input("a", (128, 128))
    c = b.input("c", (128, 128))
    mm = b.matmul("mm", a, c)
    graph = b.build(output_node=mm)
    inputs = {
        "a": rng.standard_normal((128, 128)),
        "c": rng.standard_normal((128, 128)),
    }
    return benchmark_graph(graph, inputs, label="matmul_128")


def main() -> None:
    for result in (_mlp_256(), _matmul_128()):
        label = result["label"]
        timing = result["timing_s_per_iter"]
        speedup = result["speedup"]
        print(f"{label}:")
        print(f"  interpreter   {timing['interpreter']:.3e} s/iter")
        print(f"  cpu naive     {timing['cpu_naive']:.3e} s/iter")
        print(f"  cpu optimized {timing['cpu_optimized']:.3e} s/iter")
        print(
            f"  speedups: naive {speedup['naive_vs_interpreter']:.2f}x, "
            f"optimized-vs-naive {speedup['optimized_vs_naive']:.2f}x"
        )
    print("\nResults written to results/*.json and results/*.md")


if __name__ == "__main__":
    main()
