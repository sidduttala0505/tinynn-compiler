# TinyNN Compiler

TinyNN is a small ahead-of-time compiler for neural networks. It takes a model — from
ONNX, a JSON graph, or a Python builder API — lowers it through a shared graph IR and a
set of optimization passes, and generates standalone C++, embedded C, or CUDA code.

A NumPy reference interpreter acts as the correctness oracle. Every optimization pass and
every generated backend is checked against it, so the compiled output is guaranteed to
match the reference implementation. The project was built to work through compiler
construction end to end — from graph frontend to generated, compilable code — in a single,
well-tested codebase.

## What it covers

- **Compiler construction** — a typed, immutable graph IR, a pass manager, constant
  folding, algebraic simplification, dead-code elimination, and operator fusion.
- **Code generation** — three independent backends that emit and compile real code:
  portable C++, static-memory embedded C99, and a CUDA GPU slice.
- **Performance** — cache-blocked loop tiling, loop interchange, `-O3 -march=native`
  vectorization, optional OpenMP, and liveness-based memory planning (buffer reuse).
- **ML infrastructure** — ONNX import and int8 quantization with bit-exact rounding parity
  between the interpreter and the generated C++.
- **Engineering** — differential testing against the oracle, seeded graph fuzzing, a real
  CLI, packaging, and continuous integration.

## How it works

The compilation pipeline runs frontend → graph IR → optimization passes → backend. A model
enters from ONNX, JSON, or the builder API; it is lowered to a shared IR, optimized, and
then either interpreted with NumPy or compiled to C++, embedded C, or CUDA.

The design rule the whole project is built around is that the NumPy interpreter is the
single source of truth. For any graph and any input, the optimized graph must produce the
same result as the original, and every compiled binary must match the interpreter — checked
numerically before any timing is ever recorded. That discipline is what makes the optimizer
and all three backends trustworthy.

## Quickstart

```bash
git clone https://github.com/sidduttala0505/tinynn-compiler.git
cd tinynn-compiler
python -m pip install -e ".[dev]"

# Run the CPU test suite
python -m pytest tests -k "not cuda"
```

TinyNN requires Python 3.9+ and NumPy. `onnx` and `scikit-learn` are optional extras,
needed only for ONNX import and the example dataset.

### Build and compile a model in Python

```python
import numpy as np
from tinynn import GraphBuilder, run, default_pipeline, compile_graph

# A small 4 -> 8 -> 3 MLP
b = GraphBuilder()
x = b.input("x", (4,))
h = b.linear("fc0", x, weight=np.random.randn(4, 8), bias=np.random.randn(8))
h = b.relu("act0", h)
y = b.linear("fc1", h, weight=np.random.randn(8, 3), bias=np.random.randn(3))
b.output("y", y)
graph = b.build()

x_in = np.array([0.5, -1.0, 2.0, 0.25])

reference = run(graph, x_in)                 # NumPy interpreter (the oracle)
optimized = default_pipeline().run(graph)    # fold -> simplify -> DCE -> fuse Linear+ReLU
model = compile_graph(optimized, "build/")   # emit + compile standalone C++
compiled = model.run(x_in)                   # execute the native binary

assert np.allclose(reference, compiled)      # backend matches the oracle
```

### Or use the command line

```bash
# Compile an ONNX model to an optimized native binary, with a benchmark and graph dump
python -m tinynn compile model.onnx --out build/ --fast --benchmark --emit-dot

# Lower the same model fully to int8 on the embedded C backend
python -m tinynn compile model.onnx --out build/ --backend embedded_c --quantize

# Evaluate with the interpreter, or render the graph to Graphviz DOT
python -m tinynn run model.onnx
python -m tinynn visualize model.onnx --out graph.dot
```

The CLI (`compile`, `run`, `benchmark`, `visualize`) accepts both `.onnx` and `.json`
models and fails with clean, traceback-free messages on bad input or a missing optional
dependency.

## Features in detail

**Graph IR.** `Node`, `Graph`, and `GraphBuilder` are frozen dataclasses. Nodes reference
each other by name rather than by pointer, graphs are stored in topological order, and
validation runs eagerly at construction so malformed graphs fail immediately. Passes never
mutate — they return fresh graphs — which keeps transformations composable.

**Operators.** `Input`, `Linear`, `ReLU`, `Output`, elementwise `Add` / `Sub` / `Mul`, 2D
`MatMul`, numerically stable `Softmax`, `Tanh`, `Sigmoid`, and compile-time `Const`, plus
the fused and quantized ops `FusedLinearReLU`, `QuantizedLinear`, and
`QuantizedFusedLinearReLU`.

**Optimization passes.** Constant folding, algebraic simplification (add/sub-zero, mul-one,
ReLU idempotence), dead-code elimination, and `Linear + ReLU → FusedLinearReLU` fusion, all
orchestrated by a pass manager. The default pipeline is fold → simplify → DCE → fuse.

**CPU performance codegen.** Loop interchange and cache-blocked tiling for `Linear`,
`MatMul`, and fused ops; vectorization-friendly row-major streaming under
`-O3 -march=native`; option-gated OpenMP with a runtime availability probe; and
liveness-based memory planning that reuses buffers — for example, collapsing five logical
buffers to two physical slots on an MLP chain.

**Int8 quantization.** Symmetric int8 weight quantization with dynamic per-call activation
quantization, int64 accumulation, and a single float rescale. Rounding is
round-half-away-from-zero to match C++ `std::lround`, so the quantized interpreter path and
the quantized C++ path produce bit-identical results rather than merely close ones.

**ONNX import.** Imports MLP-style ONNX models (`Gemm`, `MatMul`, `Relu`, `Tanh`,
`Sigmoid`, `Softmax`, `Add`, `Sub`, `Mul`, `Identity`), with clear errors on anything
outside the supported subset.

**Embedded C backend.** An alternate code generator for constrained devices: plain C99 with
file-scope `static` arrays only — no `malloc`, no `new`, no `std::vector`, enforced by a
source-inspection test — and int8 weights baked in as `static const int8_t`.

**CUDA backend.** A deliberately narrow, honest vertical slice rather than full GPU support.
It generates CUDA C++, compiles it with `nvcc`, runs real kernels on an NVIDIA GPU, and
compares the output against the interpreter oracle. It covers 1D `Input` / `Linear` /
`FusedLinearReLU` / `ReLU` / `Output`. A fused node emits one linear-plus-ReLU kernel
and does not materialize a pre-ReLU activation buffer. Its tests skip cleanly when no
compiler, runtime, or GPU is present.

## Testing

Correctness is the point of the project, so the test story is substantial. There are 16
test suites and over 175 test functions (237 cases collected and passing on a CUDA-capable
machine; CPU-only runs skip the GPU tests cleanly). Testing is differential: every op,
every pass, and every backend is compared against the NumPy oracle. Seeded graph fuzzing
generates random graphs — including branches, diamonds, multi-consumer nodes, dead
branches, and 2D tensors — and compares them four ways: interpreter versus compiled C++,
and optimized versus unoptimized.

Continuous integration runs on GitHub Actions (Ubuntu, Python 3.11). It installs the
package, runs the full suite, checks the CLI help, and compiles both a float and a
quantized embedded-C demo end to end on every push.

```bash
python -m pytest tests            # full suite (needs g++; CUDA and ONNX tests are conditional)
python -m pytest tests -k "not cuda"   # CPU-only
```

Backend tests are conditional on your toolchain: C++ tests need `g++`, embedded-C tests
need a C/C++ compiler, ONNX tests need the optional `onnx` package, and CUDA tests need
`nvcc` plus a working GPU.

### CUDA steady-state benchmark

On an NVIDIA host, run:

```bash
python -m tinynn.cuda_benchmark --results-dir results/cuda --iters 10000 --warmup 1000
```

The harness verifies each workload against the NumPy interpreter before timing a deterministic
`256 -> 512 -> 512 -> 10` MLP and the repository's `examples/models/digits_mlp.onnx`.
It performs parameter/input uploads and device allocation once, warms up, then reports
CUDA-event seconds per inference for device kernel execution only. It deliberately excludes
process startup, file I/O, transfer time, and one-time setup, and makes no cross-framework
speedup claim.

## Benchmarks

End-to-end demos write benchmark reports to `results/`. On a three-layer MLP, the optimized
C++ (passes plus tiled, vectorized codegen) runs roughly 8x faster than the naive `-O2`
C++. One honest caveat, baked into the reports: the NumPy interpreter delegates
matmul-heavy work to an optimized BLAS, so it can outrun TinyNN's scalar C++ loops — the
meaningful codegen comparison is optimized versus naive C++, and the numbers are
machine-dependent.

The MNIST-style demo shows quantization is lossless in practice here. An MLP trained in pure
NumPy (manual backprop, no PyTorch) on the scikit-learn 8x8 digits set holds about 97% test
accuracy identically across the NumPy baseline, the interpreter, the optimized graph, the
compiled C++, and both int8 paths. Note that this demo uses scikit-learn's 8x8 digits as a
small, fast stand-in — it is not the real 28x28 MNIST.

## Repository layout

```
tinynn/
├── graph.py         # Graph IR: Node / Graph / GraphBuilder (frozen, validated)
├── ops.py           # Canonical operator names and quantization semantics
├── interpreter.py   # NumPy reference interpreter — the correctness oracle
├── analysis.py      # Topological sort and shape inference
├── passes.py        # Optimization passes, pass manager, default pipeline
├── codegen.py       # C++ code generator and g++ compile/run driver
├── embedded.py      # Static-memory embedded C99 backend
├── cuda.py          # Minimal CUDA backend vertical slice
├── onnx_import.py   # ONNX -> Graph IR importer (optional onnx dependency)
├── serialize.py     # Graph <-> JSON
├── viz.py           # Graphviz DOT export
├── benchmark.py     # Verify-then-time benchmark harness
└── cli.py           # argparse CLI: compile / run / benchmark / visualize

examples/            # End-to-end demos and sample models
results/             # Committed benchmark and accuracy reports
tests/               # 16 test suites, including fuzzing and per-backend tests
ROADMAP.md           # Tiered development plan and status
```

The project is organized into tiers in `ROADMAP.md`, from the end-to-end foundation through
graph infrastructure, more operators, the optimizer, CPU performance, the GPU slice,
quantization, and correctness rigor.

## Design principles

- The interpreter is the oracle; every backend and every pass is verified against it.
- Correctness comes before speed: compiled variants are checked for numerical agreement
  before any latency is measured.
- The IR is immutable and passes are pure `Graph -> Graph` functions.
- Scope is stated honestly: where a feature is a slice rather than full support, such as the
  CUDA backend or the digits dataset, the docs say so.

## License

This is a personal portfolio project. If you would like to reuse it, please open an issue or
add a `LICENSE` file specifying your terms.
