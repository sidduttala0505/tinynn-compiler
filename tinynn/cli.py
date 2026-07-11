"""Command-line interface for TinyNN.

Wraps the library API (graph loading, optimization passes, the NumPy
interpreter, C++ codegen/compilation, benchmarking, and DOT visualization)
in a small ``argparse``-based CLI, so common workflows don't require writing
a Python script. Every subcommand is a thin driver over functions already
exported from the ``tinynn`` package -- this module contains no modeling or
compilation logic of its own.

Subcommands
-----------
``compile MODEL --out DIR``
    Load a model, optionally run the default optimization pipeline and/or
    int8 quantization, then compile it to a standalone C++ binary (or, with
    ``--backend embedded_c``, the experimental embedded-C backend).
``run MODEL``
    Load a model and evaluate it with the NumPy reference interpreter,
    either on a provided ``.npy`` input or on random inputs.
``benchmark MODEL --out DIR``
    Run the interpreter-vs-naive-C++-vs-optimized-C++ benchmark harness
    (:func:`tinynn.benchmark_graph`) and report where the results landed.
``visualize MODEL --out PATH.dot``
    Render a graph to Graphviz DOT text.

Exit codes: ``0`` on success, ``2`` for clean, user-facing errors (bad
model path/extension, missing optional dependency, shape mismatch, ...).
Uncaught exceptions from the underlying library are allowed to propagate as
tracebacks -- they indicate a bug, not a usage error.

Entry points: ``python -m tinynn ...`` (see :mod:`tinynn.__main__`) or
``python -m tinynn.cli ...`` both call :func:`main`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .benchmark import benchmark_graph
from .codegen import CodegenOptions, compile_graph, openmp_available
from .graph import Graph
from .interpreter import run as run_graph
from .onnx_import import import_onnx
from .passes import (
    default_pipeline,
    fuse_linear_relu,
    quantize_fused_linear_relu,
    quantize_linear,
)
from .serialize import load_json
from .viz import save_dot

__all__ = ["main"]


def _fail(msg: str) -> int:
    """Print ``msg`` to stderr (prefixed ``"error: "``) and return exit code 2.

    Callers write ``return _fail(...)`` from within a subcommand handler so
    the message reaches the user without a traceback, and the process exits
    cleanly with code 2 (the convention this CLI uses for usage/data errors,
    as distinct from unexpected bugs which are allowed to raise).
    """
    print(f"error: {msg}", file=sys.stderr)
    return 2


class _CleanExit(Exception):
    """Internal control-flow exception carrying a clean exit code.

    Raised by helpers that are called from deep inside a subcommand handler
    (where a bare ``return`` would only unwind one frame) and caught once in
    :func:`main` so every subcommand still reports a clean error via
    :func:`_fail` instead of a traceback.
    """

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _load_graph(path: str) -> Graph:
    """Load a ``Graph`` from ``path`` based on its file suffix.

    ``.onnx`` files are imported with :func:`tinynn.import_onnx`; ``.json``
    files (as written by :func:`tinynn.save_json`) are loaded with
    :func:`tinynn.load_json`. Any other suffix, a missing file, a missing
    ``onnx`` package, or a semantically unsupported ONNX model all raise
    :class:`_CleanExit` after printing a clean, traceback-free message to
    stderr -- callers should invoke this only from within ``main``'s
    top-level dispatch (or a handler called directly from it) so the
    exception is caught in one place.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise _CleanExit(_fail(f"model file not found: {model_path}"))

    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        try:
            return import_onnx(model_path)
        except ImportError:
            raise _CleanExit(
                _fail(
                    "loading an .onnx model requires the 'onnx' package; "
                    "install it with `pip install onnx`"
                )
            )
        except ValueError as exc:
            raise _CleanExit(_fail(f"could not import ONNX model: {exc}"))
    elif suffix == ".json":
        return load_json(model_path)
    else:
        raise _CleanExit(
            _fail(
                f"unsupported model file extension {suffix!r} (expected "
                "'.onnx' or '.json')"
            )
        )


def _random_inputs(graph: Graph, seed: int) -> Dict[str, np.ndarray]:
    """Generate standard-normal random inputs for every ``Input`` node.

    Uses ``numpy.random.default_rng(seed)`` so runs are reproducible given
    the same seed.
    """
    rng = np.random.default_rng(seed)
    return {n.name: rng.standard_normal(n.shape) for n in graph.input_nodes()}


def _apply_pipeline(graph: Graph, pipeline: str) -> Graph:
    """Apply the named pipeline ('default' or 'none') to ``graph``."""
    if pipeline == "none":
        return graph
    return default_pipeline().run(graph)


def _apply_quantize(graph: Graph) -> Graph:
    """Apply the opt-in int8 quantization passes and print a numerics note."""
    graph = quantize_fused_linear_relu(graph)
    graph = quantize_linear(graph)
    print("note: int8 quantization changes numerics (approximate results).")
    return graph


def _graph_stats(graph: Graph) -> Dict[str, object]:
    """Return small Markdown-friendly graph statistics for reports."""
    input_nodes = graph.input_nodes()
    output = graph.get_node(graph.output_node)
    parameter_count = 0
    for node in graph.nodes:
        if node.weight is not None:
            parameter_count += int(node.weight.size)
        if node.bias is not None:
            parameter_count += int(node.bias.size)

    return {
        "nodes": len(graph.nodes),
        "ops": dict(Counter(node.op for node in graph.nodes)),
        "inputs": {node.name: list(node.shape) for node in input_nodes},
        "output_node": graph.output_node,
        "output_shape": list(output.shape),
        "parameters": parameter_count,
    }


def _format_ops(ops: Dict[str, int]) -> str:
    if not ops:
        return "(none)"
    return ", ".join(f"{op}: {count}" for op, count in sorted(ops.items()))


def _report_graph_section(title: str, stats: Dict[str, object]) -> list[str]:
    inputs = stats["inputs"]
    input_text = ", ".join(f"{name}{tuple(shape)}" for name, shape in inputs.items())
    if not input_text:
        input_text = "(none)"
    return [
        f"## {title}",
        "",
        f"- nodes: {stats['nodes']}",
        f"- ops: {_format_ops(stats['ops'])}",
        f"- inputs: {input_text}",
        f"- output: {stats['output_node']}{tuple(stats['output_shape'])}",
        f"- parameters: {stats['parameters']}",
        "",
    ]


def _write_compile_report(
    path: Path,
    *,
    model_path: Path,
    original_graph: Graph,
    compiled_graph: Graph,
    backend: str,
    quantized: bool,
    static_memory: bool,
    artifacts: Sequence[Path],
    benchmark: Optional[Dict[str, object]],
) -> Path:
    """Write a concise Markdown compile report and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# TinyNN Compile Report",
        "",
        f"- model: `{model_path}`",
        f"- backend: `{backend}`",
        f"- quantization: {'enabled' if quantized else 'disabled'}",
        f"- embedded/static memory: {'yes' if static_memory else 'no'}",
        "",
    ]
    lines.extend(_report_graph_section("Original Graph", _graph_stats(original_graph)))
    lines.extend(
        _report_graph_section(
            "Optimized Graph",
            _graph_stats(compiled_graph),
        )
    )

    if benchmark is not None:
        timing = benchmark["timing_s_per_iter"]
        speedup = benchmark["speedup"]
        lines.extend(
            [
                "## Benchmark",
                "",
                f"- iters: {benchmark['iters']}",
                f"- warmup: {benchmark['warmup']}",
                f"- OpenMP used: {benchmark['openmp_used']}",
                f"- correctness verified: {benchmark['correctness_verified']}",
                "",
                "| variant | seconds/iter |",
                "|---|---:|",
                f"| interpreter | {timing['interpreter']:.6e} |",
                f"| cpu naive | {timing['cpu_naive']:.6e} |",
                f"| cpu optimized | {timing['cpu_optimized']:.6e} |",
                "",
                "| speedup | value |",
                "|---|---:|",
                f"| naive vs interpreter | {speedup['naive_vs_interpreter']:.2f}x |",
                f"| optimized vs naive | {speedup['optimized_vs_naive']:.2f}x |",
                f"| optimized vs interpreter | {speedup['optimized_vs_interpreter']:.2f}x |",
                "",
            ]
        )

    lines.extend(["## Output Artifacts", ""])
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    lines.append("")

    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Subcommand: compile
# --------------------------------------------------------------------------- #
def _cmd_compile(args: argparse.Namespace) -> int:
    try:
        original_graph = _load_graph(args.model)
    except _CleanExit as exc:
        return exc.code

    graph = _apply_pipeline(original_graph, args.pipeline)
    if args.quantize:
        graph = _apply_quantize(graph)

    out_dir = Path(args.out)
    artifacts = []

    if args.backend == "embedded_c":
        try:
            from .embedded import compile_embedded
        except ImportError:
            return _fail(
                "the experimental embedded_c backend is unavailable "
                "(tinynn.embedded could not be imported)"
            )
        model = compile_embedded(graph, out_dir)
    else:
        options = None
        if args.fast:
            options = CodegenOptions.fast(openmp=openmp_available())
        model = compile_graph(graph, out_dir, options=options)
    artifacts.extend([model.cpp_path, model.binary_path])

    if args.emit_dot:
        dot_path = save_dot(graph, out_dir / "graph.dot")
        artifacts.append(dot_path)
        print(f"Wrote DOT graph to {dot_path}")

    lang = "C" if args.backend == "embedded_c" else "C++"
    print(f"Wrote generated {lang} source to {model.cpp_path}")
    print(f"Wrote compiled binary to {model.binary_path}")

    benchmark_result = None
    if args.benchmark:
        label = Path(args.model).stem
        inputs = _random_inputs(original_graph, args.seed)
        benchmark_result = benchmark_graph(
            original_graph,
            inputs,
            iters=args.iters,
            warmup=args.warmup,
            results_dir=out_dir,
            label=label,
        )
        artifacts.extend([out_dir / f"{label}.json", out_dir / f"{label}.md"])
        _print_benchmark_summary(benchmark_result)

    if args.report is not None:
        report_path = _write_compile_report(
            Path(args.report),
            model_path=Path(args.model),
            original_graph=original_graph,
            compiled_graph=graph,
            backend=args.backend,
            quantized=args.quantize,
            static_memory=args.backend == "embedded_c",
            artifacts=artifacts,
            benchmark=benchmark_result,
        )
        print(f"Wrote compile report to {report_path}")

    return 0


# --------------------------------------------------------------------------- #
# Subcommand: run
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    try:
        graph = _load_graph(args.model)
    except _CleanExit as exc:
        return exc.code

    graph = _apply_pipeline(graph, args.pipeline)
    if args.quantize:
        graph = _apply_quantize(graph)

    input_nodes = graph.input_nodes()

    if args.input is not None:
        if len(input_nodes) != 1:
            return _fail(
                "--input is only supported for models with exactly one "
                f"Input node; this model has {len(input_nodes)} "
                f"({[n.name for n in input_nodes]})"
            )
        input_path = Path(args.input)
        if not input_path.exists():
            return _fail(f"input file not found: {input_path}")
        try:
            array = np.load(input_path)
        except (OSError, ValueError) as exc:
            return _fail(f"could not load {input_path} as a NumPy .npy file: {exc}")

        expected = 1
        for d in input_nodes[0].shape:
            expected *= d
        if array.size != expected:
            return _fail(
                f"input file {input_path} has {array.size} element(s), but "
                f"Input node {input_nodes[0].name!r} expects {expected} "
                f"(shape {input_nodes[0].shape})"
            )
        inputs = {input_nodes[0].name: array}
    else:
        inputs = _random_inputs(graph, args.seed)
        print("note: no --input given, using random inputs.")

    output = run_graph(graph, inputs)
    print(output)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: benchmark
# --------------------------------------------------------------------------- #
def _cmd_benchmark(args: argparse.Namespace) -> int:
    try:
        graph = _load_graph(args.model)
    except _CleanExit as exc:
        return exc.code

    inputs = _random_inputs(graph, args.seed)
    label = Path(args.model).stem
    out_dir = Path(args.out)

    result = benchmark_graph(
        graph,
        inputs,
        iters=args.iters,
        warmup=args.warmup,
        results_dir=out_dir,
        label=label,
    )
    _print_benchmark_summary(result)
    print(f"Wrote {out_dir / (label + '.json')}")
    print(f"Wrote {out_dir / (label + '.md')}")
    return 0


def _print_benchmark_summary(result: Dict[str, object]) -> None:
    """Print per-variant seconds/iter and speedups from a benchmark_graph() result."""
    timing = result["timing_s_per_iter"]
    speedup = result["speedup"]
    print(f"benchmark: {result['label']}")
    print(f"  interpreter   {timing['interpreter']:.3e} s/iter")
    print(f"  cpu naive     {timing['cpu_naive']:.3e} s/iter")
    print(f"  cpu optimized {timing['cpu_optimized']:.3e} s/iter")
    print(
        f"  speedups: naive-vs-interpreter {speedup['naive_vs_interpreter']:.2f}x, "
        f"optimized-vs-naive {speedup['optimized_vs_naive']:.2f}x, "
        f"optimized-vs-interpreter {speedup['optimized_vs_interpreter']:.2f}x"
    )


# --------------------------------------------------------------------------- #
# Subcommand: visualize
# --------------------------------------------------------------------------- #
def _cmd_visualize(args: argparse.Namespace) -> int:
    try:
        graph = _load_graph(args.model)
    except _CleanExit as exc:
        return exc.code

    graph = _apply_pipeline(graph, args.pipeline)
    dot_path = save_dot(graph, args.out)
    print(f"Wrote DOT graph to {dot_path}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinynn",
        description="TinyNN: a tiny compiler for simple neural networks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- compile ------------------------------------------------------- #
    p_compile = subparsers.add_parser(
        "compile", help="Compile a model to a standalone C++ binary."
    )
    p_compile.add_argument("model", help="Path to a .onnx or .json model file")
    p_compile.add_argument("--out", required=True, help="Output directory")
    p_compile.add_argument(
        "--pipeline", choices=["default", "none"], default="default",
        help="Optimization pipeline to apply before compiling (default: default)",
    )
    p_compile.add_argument(
        "--backend", choices=["cpp", "embedded_c"], default="cpp",
        help="Codegen backend to use (default: cpp)",
    )
    p_compile.add_argument(
        "--quantize", action="store_true",
        help="Apply int8 quantization passes after the pipeline (changes numerics)",
    )
    p_compile.add_argument(
        "--fast", action="store_true",
        help="Use CodegenOptions.fast() (-O3 -march=native, tiling, OpenMP if available)",
    )
    p_compile.add_argument(
        "--benchmark", action="store_true",
        help="Also run the interpreter/naive/optimized benchmark harness",
    )
    p_compile.add_argument(
        "--emit-dot", action="store_true",
        help="Also write the (post-pass) graph as out/graph.dot",
    )
    p_compile.add_argument(
        "--report",
        default=None,
        help="Write a Markdown compile report to the given path",
    )
    p_compile.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for --benchmark's random inputs (default: 0)",
    )
    p_compile.add_argument(
        "--iters", type=int, default=200,
        help="Timed iterations for --benchmark (default: 200)",
    )
    p_compile.add_argument(
        "--warmup", type=int, default=20,
        help="Untimed warmup iterations for --benchmark (default: 20)",
    )
    p_compile.set_defaults(func=_cmd_compile)

    # -- run ------------------------------------------------------------ #
    p_run = subparsers.add_parser(
        "run", help="Evaluate a model with the NumPy reference interpreter."
    )
    p_run.add_argument("model", help="Path to a .onnx or .json model file")
    p_run.add_argument(
        "--input", default=None,
        help="Path to a .npy file with input values (single-Input models only)",
    )
    p_run.add_argument(
        "--pipeline", choices=["default", "none"], default="default",
        help="Optimization pipeline to apply before running (default: default)",
    )
    p_run.add_argument(
        "--quantize", action="store_true",
        help="Apply int8 quantization passes after the pipeline (changes numerics)",
    )
    p_run.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for random inputs when --input is not given (default: 0)",
    )
    p_run.set_defaults(func=_cmd_run)

    # -- benchmark -------------------------------------------------------- #
    p_bench = subparsers.add_parser(
        "benchmark", help="Benchmark interpreter vs naive C++ vs optimized C++."
    )
    p_bench.add_argument("model", help="Path to a .onnx or .json model file")
    p_bench.add_argument("--out", required=True, help="Results directory")
    p_bench.add_argument(
        "--iters", type=int, default=200, help="Timed iterations (default: 200)"
    )
    p_bench.add_argument(
        "--warmup", type=int, default=20, help="Untimed warmup iterations (default: 20)"
    )
    p_bench.add_argument(
        "--seed", type=int, default=0, help="RNG seed for random inputs (default: 0)"
    )
    p_bench.set_defaults(func=_cmd_benchmark)

    # -- visualize --------------------------------------------------------- #
    p_viz = subparsers.add_parser(
        "visualize", help="Render a model as a Graphviz DOT file."
    )
    p_viz.add_argument("model", help="Path to a .onnx or .json model file")
    p_viz.add_argument("--out", required=True, help="Output .dot file path")
    p_viz.add_argument(
        "--pipeline", choices=["default", "none"], default="none",
        help="Optimization pipeline to apply before visualizing (default: none)",
    )
    p_viz.set_defaults(func=_cmd_visualize)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse ``argv`` (or ``sys.argv[1:]`` if ``None``) and run the selected subcommand.

    Returns the process exit code: ``0`` on success, ``2`` for clean,
    user-facing errors reported via :func:`_fail`. ``argparse`` itself exits
    the process (via ``SystemExit``) for ``--help`` and argument-parsing
    errors, which is standard ``argparse`` behavior and is left unchanged.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
