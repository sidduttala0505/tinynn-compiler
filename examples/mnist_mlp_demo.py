"""End-to-end "MNIST-style" demo: NumPy training -> ONNX export -> TinyNN
import -> interpret -> optimize -> compile to C++ -> quantize to int8 ->
verify every stage against the interpreter oracle -> benchmark -> report.

HONESTY NOTE: the dataset is scikit-learn's ``load_digits`` (8x8 grayscale
digit images, 64 features, 10 classes, 1797 samples) -- an MNIST-style
digit-classification dataset, but NOT the real MNIST dataset (which is
28x28 with 70000 samples). Every printed line and report section below
calls it "sklearn digits (8x8)". No PyTorch is used anywhere: the "NumPy
baseline" is the pure-NumPy training forward pass from
``export_mnist_mlp.py``. Every number in the generated report is computed
during this run -- nothing is hardcoded.

Run from the repo root (one command runs everything):

    ./venv/bin/python examples/mnist_mlp_demo.py --out results/mnist

Re-running is idempotent: it retrains from the same seed, re-exports the
same ONNX weights, and overwrites the output artifacts with the same
(deterministic) numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np

# Allow running as `python examples/mnist_mlp_demo.py` (repo root not on
# sys.path by default).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_mnist_mlp import (  # noqa: E402
    accuracy,
    export_onnx,
    load_dataset,
    train_mlp,
)

import tinynn  # noqa: E402
from tinynn import (  # noqa: E402
    CodegenOptions,
    compile_graph,
    default_pipeline,
    fuse_linear_relu,
    import_onnx,
    openmp_available,
    quantize_fused_linear_relu,
    quantize_linear,
)

MODEL_PATH = _REPO_ROOT / "examples" / "models" / "digits_mlp.onnx"

_VERIFY_RTOL = _VERIFY_ATOL = 1e-9
_LOGIT_TOL = 1e-6


def _interp_accuracy(graph, X: np.ndarray, y: np.ndarray) -> float:
    """TinyNN interpreter accuracy over a test set, one sample at a time."""
    correct = 0
    for xi, yi in zip(X, y):
        out = tinynn.run(graph, xi.astype(np.float64))
        if int(np.argmax(out)) == int(yi):
            correct += 1
    return correct / len(y)


def _compiled_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    """Compiled C++ binary accuracy over a test set (one subprocess call per
    sample)."""
    correct = 0
    for xi, yi in zip(X, y):
        out = model.run(xi.astype(np.float64))
        if int(np.argmax(out)) == int(yi):
            correct += 1
    return correct / len(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/mnist", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed (drives everything)")
    parser.add_argument("--epochs", type=int, default=30, help="NumPy MLP training epochs")
    parser.add_argument("--iters", type=int, default=2000, help="benchmark iterations for compiled models")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = out_dir / "build"

    t_start = time.perf_counter()

    # ----------------------------------------------------------------- #
    # 1. Load dataset + train the NumPy MLP.
    # ----------------------------------------------------------------- #
    print("=" * 72)
    print("Step 1/8: load dataset + train pure-NumPy MLP")
    print("=" * 72)
    X_train, y_train, X_test, y_test, dataset_name = load_dataset(args.seed)
    print(f"Dataset: {dataset_name}")
    print(f"  train samples: {X_train.shape[0]}, test samples: {X_test.shape[0]}, features: {X_train.shape[1]}")

    params = train_mlp(X_train, y_train, seed=args.seed, epochs=args.epochs)
    numpy_train_acc = accuracy(params, X_train, y_train)
    numpy_test_acc = accuracy(params, X_test, y_test)
    print(f"NumPy baseline train accuracy: {numpy_train_acc:.4f}")
    print(f"NumPy baseline test accuracy:  {numpy_test_acc:.4f}")

    # ----------------------------------------------------------------- #
    # 2. Export ONNX.
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 2/8: export trained weights to ONNX")
    print("=" * 72)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(params, MODEL_PATH)
    print(f"Wrote ONNX model to {MODEL_PATH}")

    # ----------------------------------------------------------------- #
    # 3. Import + interpret.
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 3/8: import ONNX into TinyNN Graph IR, run interpreter")
    print("=" * 72)
    graph = import_onnx(MODEL_PATH)
    print(f"Imported graph: {len(graph.nodes)} nodes, ops = "
          f"{sorted({n.op for n in graph.nodes})}")

    interp_test_acc = _interp_accuracy(graph, X_test, y_test)
    print(f"TinyNN interpreter test accuracy: {interp_test_acc:.4f}")

    # Compare interpreter logits vs the NumPy forward pass on a subset, to
    # bound the float32-rounding error introduced by the ONNX export
    # (weights are stored as float32 initializers).
    subset = min(25, X_test.shape[0])
    max_logit_diff = 0.0
    for xi in X_test[:subset]:
        interp_logits = tinynn.run(graph, xi.astype(np.float64))
        W0, b0, W1, b1 = params["W0"], params["b0"], params["W1"], params["b1"]
        numpy_logits = np.maximum(xi @ W0 + b0, 0.0) @ W1 + b1
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(interp_logits - numpy_logits))))
    assert max_logit_diff < _LOGIT_TOL, (
        f"interpreter vs NumPy forward logits differ by {max_logit_diff}, "
        f"expected < {_LOGIT_TOL} (float32 weight rounding bound)"
    )
    print(f"Max |interpreter logits - NumPy forward logits| over {subset} samples: "
          f"{max_logit_diff:.3e} (< {_LOGIT_TOL:g}: OK)")

    # ----------------------------------------------------------------- #
    # 4. Optimize (default pipeline: const folding, algebraic simplify,
    #    DCE, Linear/ReLU fusion).
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 4/8: optimize with tinynn.default_pipeline()")
    print("=" * 72)
    opt_graph = default_pipeline().run(graph)
    opt_ops = {n.op for n in opt_graph.nodes}
    assert "FusedLinearReLU" in opt_ops, f"expected FusedLinearReLU in optimized graph ops, got {opt_ops}"
    print(f"Optimized graph: {len(opt_graph.nodes)} nodes, ops = {sorted(opt_ops)}")

    opt_interp_test_acc = _interp_accuracy(opt_graph, X_test, y_test)
    assert opt_interp_test_acc == interp_test_acc, (
        f"optimized-graph interpreter accuracy ({opt_interp_test_acc}) != "
        f"original-graph interpreter accuracy ({interp_test_acc}); same numerics expected"
    )
    print(f"Optimized-graph interpreter test accuracy: {opt_interp_test_acc:.4f} "
          f"(matches original graph exactly: OK)")

    # ----------------------------------------------------------------- #
    # 5. Compile to C++: naive (from original graph) and optimized (from
    #    opt_graph, CodegenOptions.fast()).
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 5/8: compile to native C++ (naive + optimized)")
    print("=" * 72)
    use_openmp = openmp_available()
    print(f"OpenMP available: {use_openmp}")

    naive = compile_graph(graph, build_dir / "naive", options=None)
    optimized = compile_graph(
        opt_graph, build_dir / "optimized",
        options=CodegenOptions.fast(openmp=use_openmp),
    )
    print(f"Compiled naive binary:     {naive.binary_path}")
    print(f"Compiled optimized binary: {optimized.binary_path}")

    verify_n = min(50, X_test.shape[0])
    for xi in X_test[:verify_n]:
        xi64 = xi.astype(np.float64)
        interp_out = tinynn.run(graph, xi64)
        naive_out = naive.run(xi64)
        opt_out = optimized.run(xi64)
        assert np.allclose(naive_out, interp_out, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL), (
            f"naive compiled output disagrees with interpreter oracle "
            f"(max abs diff {np.max(np.abs(naive_out - interp_out))})"
        )
        assert np.allclose(opt_out, interp_out, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL), (
            f"optimized compiled output disagrees with interpreter oracle "
            f"(max abs diff {np.max(np.abs(opt_out - interp_out))})"
        )
    print(f"Verified naive + optimized compiled outputs against the interpreter "
          f"on {verify_n} samples (rtol=atol={_VERIFY_RTOL:g}): OK")

    compiled_opt_test_acc = _compiled_accuracy(optimized, X_test, y_test)
    print(f"Compiled C++ (optimized) test accuracy: {compiled_opt_test_acc:.4f}")

    # ----------------------------------------------------------------- #
    # 6. int8 quantization path.
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 6/8: int8 quantization (fuse -> quantize fused -> quantize linear)")
    print("=" * 72)
    q_graph = quantize_linear(quantize_fused_linear_relu(fuse_linear_relu(graph)))
    q_ops = {n.op for n in q_graph.nodes}
    assert "QuantizedFusedLinearReLU" in q_ops, (
        f"expected QuantizedFusedLinearReLU in quantized graph ops, got {q_ops}"
    )
    print(f"Quantized graph: {len(q_graph.nodes)} nodes, ops = {sorted(q_ops)}")

    q_interp_test_acc = _interp_accuracy(q_graph, X_test, y_test)
    print(f"int8 interpreter test accuracy: {q_interp_test_acc:.4f}")

    q_compiled = compile_graph(q_graph, build_dir / "int8", options=None)
    print(f"Compiled int8 binary: {q_compiled.binary_path}")

    for xi in X_test[:verify_n]:
        xi64 = xi.astype(np.float64)
        q_interp_out = tinynn.run(q_graph, xi64)
        q_compiled_out = q_compiled.run(xi64)
        assert np.allclose(q_compiled_out, q_interp_out, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL), (
            f"int8 compiled output disagrees with int8 interpreter oracle "
            f"(max abs diff {np.max(np.abs(q_compiled_out - q_interp_out))})"
        )
    print(f"Verified int8 compiled output against the int8 interpreter on "
          f"{verify_n} samples (rtol=atol={_VERIFY_RTOL:g}): OK")

    compiled_q_test_acc = _compiled_accuracy(q_compiled, X_test, y_test)
    print(f"Compiled int8 test accuracy: {compiled_q_test_acc:.4f}")

    # ----------------------------------------------------------------- #
    # 7. Latency (only after all verification above has passed).
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 7/8: latency benchmarks")
    print("=" * 72)
    x0 = X_test[0].astype(np.float64)

    interp_iters = 200
    for _ in range(10):  # warmup
        tinynn.run(graph, x0)
    t0 = time.perf_counter()
    for _ in range(interp_iters):
        tinynn.run(graph, x0)
    interp_s_per_iter = (time.perf_counter() - t0) / interp_iters

    warmup = max(1, args.iters // 10)
    naive_s_per_iter = naive.benchmark(x0, iters=args.iters, warmup=warmup)
    optimized_s_per_iter = optimized.benchmark(x0, iters=args.iters, warmup=warmup)
    q_s_per_iter = q_compiled.benchmark(x0, iters=args.iters, warmup=warmup)

    print(f"interpreter (NumPy, per-sample):        {interp_s_per_iter:.3e} s/iter")
    print(f"C++ naive:                               {naive_s_per_iter:.3e} s/iter")
    print(f"C++ optimized (FusedLinearReLU, float):  {optimized_s_per_iter:.3e} s/iter")
    print(f"C++ int8 (QuantizedFusedLinearReLU):     {q_s_per_iter:.3e} s/iter")
    print(
        "FusedLinearReLU (float, optimized) vs QuantizedFusedLinearReLU (int8): "
        f"{optimized_s_per_iter / q_s_per_iter:.2f}x"
        if q_s_per_iter > 0 else "n/a"
    )

    # ----------------------------------------------------------------- #
    # 8. Report.
    # ----------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("Step 8/8: write report")
    print("=" * 72)

    timestamp = datetime.now(timezone.utc).isoformat()
    total_wall_s = time.perf_counter() - t_start

    accuracy_table = [
        ("NumPy baseline (train)", numpy_train_acc),
        ("NumPy baseline (test)", numpy_test_acc),
        ("TinyNN interpreter (test)", interp_test_acc),
        ("TinyNN interpreter, optimized graph (test)", opt_interp_test_acc),
        ("TinyNN compiled C++ optimized (test)", compiled_opt_test_acc),
        ("int8 interpreter (test)", q_interp_test_acc),
        ("int8 compiled C++ (test)", compiled_q_test_acc),
    ]

    latency_table = [
        ("interpreter (NumPy)", interp_s_per_iter, naive_s_per_iter / interp_s_per_iter if interp_s_per_iter else float("nan")),
        ("C++ naive", naive_s_per_iter, 1.0),
        ("C++ optimized (float, FusedLinearReLU)", optimized_s_per_iter, naive_s_per_iter / optimized_s_per_iter),
        ("C++ int8 (QuantizedFusedLinearReLU)", q_s_per_iter, naive_s_per_iter / q_s_per_iter),
    ]

    reproduction_cmd = (
        f"./venv/bin/python examples/mnist_mlp_demo.py --out {args.out} "
        f"--seed {args.seed} --epochs {args.epochs} --iters {args.iters}"
    )

    report_json: Dict[str, object] = {
        "timestamp": timestamp,
        "dataset": {
            "name": dataset_name,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "features": int(X_train.shape[1]),
            "classes": int(len(np.unique(y_train))),
        },
        "model_architecture": "64 -> 32 (ReLU) -> 10, trained in pure NumPy (no PyTorch)",
        "seed": args.seed,
        "epochs": args.epochs,
        "bench_iters": args.iters,
        "openmp_used": use_openmp,
        "accuracy": {name: acc for name, acc in accuracy_table},
        "latency_s_per_iter": {
            "interpreter": interp_s_per_iter,
            "cpu_naive": naive_s_per_iter,
            "cpu_optimized_float": optimized_s_per_iter,
            "cpu_int8_quantized": q_s_per_iter,
        },
        "speedup_vs_naive": {
            "cpu_optimized_float": naive_s_per_iter / optimized_s_per_iter,
            "cpu_int8_quantized": naive_s_per_iter / q_s_per_iter,
        },
        "float_fused_vs_quantized_fused_speedup": optimized_s_per_iter / q_s_per_iter,
        "max_logit_diff_interpreter_vs_numpy": max_logit_diff,
        "verification": {
            "rtol": _VERIFY_RTOL,
            "atol": _VERIFY_ATOL,
            "samples_checked": verify_n,
            "all_compiled_variants_verified_before_timing": True,
        },
        "total_wall_time_s": total_wall_s,
        "reproduction_command": reproduction_cmd,
    }
    (out_dir / "report.json").write_text(json.dumps(report_json, indent=2) + "\n")

    md_lines = []
    md_lines.append("# MNIST-style MLP demo report (TinyNN)")
    md_lines.append("")
    md_lines.append(f"- generated: {timestamp}")
    md_lines.append(f"- total wall time: {total_wall_s:.1f}s")
    md_lines.append("")

    md_lines.append("## Model architecture")
    md_lines.append("")
    md_lines.append("64 -> 32 (ReLU) -> 10, trained in **pure NumPy** (forward pass, manual")
    md_lines.append("backprop, mini-batch SGD, softmax cross-entropy). No PyTorch was used")
    md_lines.append("anywhere in this demo -- the \"NumPy baseline\" below is that same NumPy")
    md_lines.append("training forward pass, evaluated directly (not re-derived from ONNX).")
    md_lines.append("")

    md_lines.append("## Dataset")
    md_lines.append("")
    md_lines.append(f"**{dataset_name}** -- this is NOT the real MNIST dataset (which is")
    md_lines.append("28x28, 70000 samples); it is scikit-learn's 8x8 handwritten-digit")
    md_lines.append("dataset, used here as a small, fast, MNIST-style stand-in.")
    md_lines.append("")
    md_lines.append(f"- train samples: {X_train.shape[0]}")
    md_lines.append(f"- test samples: {X_test.shape[0]}")
    md_lines.append(f"- features: {X_train.shape[1]}")
    md_lines.append(f"- classes: {len(np.unique(y_train))}")
    md_lines.append(f"- seed: {args.seed}, epochs: {args.epochs}")
    md_lines.append("")

    md_lines.append("## Accuracy")
    md_lines.append("")
    md_lines.append("| stage | test accuracy |")
    md_lines.append("|---|---|")
    for name, acc in accuracy_table:
        md_lines.append(f"| {name} | {acc:.4f} |")
    md_lines.append("")
    md_lines.append(
        f"Max |interpreter logits - NumPy forward logits| over {subset} samples: "
        f"{max_logit_diff:.3e} (bound: float32 ONNX weight rounding, < {_LOGIT_TOL:g})."
    )
    md_lines.append("")

    md_lines.append("## Latency")
    md_lines.append("")
    md_lines.append(f"Benchmark iterations: {args.iters} (warmup {warmup}). OpenMP used: {use_openmp}.")
    md_lines.append("")
    md_lines.append("| variant | s/iter | speedup vs C++ naive |")
    md_lines.append("|---|---|---|")
    for name, s_per_iter, speedup in latency_table:
        md_lines.append(f"| {name} | {s_per_iter:.3e} | {speedup:.2f}x |")
    md_lines.append("")
    md_lines.append(
        "**FusedLinearReLU (float, optimized) vs QuantizedFusedLinearReLU (int8)**: "
        f"{optimized_s_per_iter / q_s_per_iter:.2f}x "
        f"({'int8 faster' if q_s_per_iter < optimized_s_per_iter else 'float optimized faster or equal'})."
    )
    md_lines.append("")

    md_lines.append("## Verification note")
    md_lines.append("")
    md_lines.append(
        f"All compiled C++ variants (naive, optimized float, int8 quantized) were "
        f"verified against the NumPy interpreter oracle on {verify_n} test samples "
        f"with `np.allclose(rtol={_VERIFY_RTOL:g}, atol={_VERIFY_ATOL:g})` **before** "
        "any latency measurement was taken. The optimized-graph interpreter accuracy "
        "was also checked to match the original-graph interpreter accuracy exactly "
        "(same float64 numerics; fusion is not lossy)."
    )
    md_lines.append("")

    md_lines.append("## Reproduction")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(reproduction_cmd)
    md_lines.append("```")
    md_lines.append("")

    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n")

    print(f"Wrote {out_dir / 'report.json'}")
    print(f"Wrote {out_dir / 'report.md'}")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Dataset: {dataset_name} ({X_train.shape[0]} train / {X_test.shape[0]} test)")
    print(f"NumPy baseline test accuracy:        {numpy_test_acc:.4f}")
    print(f"TinyNN interpreter test accuracy:    {interp_test_acc:.4f}")
    print(f"Compiled C++ optimized test accuracy:{compiled_opt_test_acc:.4f}")
    print(f"int8 interpreter test accuracy:      {q_interp_test_acc:.4f}")
    print(f"int8 compiled test accuracy:         {compiled_q_test_acc:.4f}")
    print(f"C++ optimized speedup vs naive:      {naive_s_per_iter / optimized_s_per_iter:.2f}x")
    print(f"C++ int8 speedup vs naive:            {naive_s_per_iter / q_s_per_iter:.2f}x")
    print(f"Total wall time: {total_wall_s:.1f}s")


if __name__ == "__main__":
    main()
