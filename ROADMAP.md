# TinyNN Compiler — Tier Roadmap

> NOTE: This file was reconstructed from the project owner's prompt on
> 2026-07-06 because no roadmap file existed in the repo. Treat it as the
> source of truth going forward and edit it if the intended roadmap differs.

## Tier 0 — End-to-end foundation ✅ (complete)

- Shared Graph IR (`tinynn/graph.py`, `tinynn/ops.py`): Input / Linear / ReLU / Output
- NumPy reference interpreter as the correctness oracle (`tinynn/interpreter.py`)
- Basic C++ codegen + g++ compile/run wrapper (`tinynn/codegen.py`)
- Correctness tests comparing compiled C++ against the interpreter

## Tier 1 — Graph infrastructure ✅ (complete)

- [x] Topological sort (accept unordered node lists, detect cycles) — `tinynn/analysis.py`
- [x] Shape inference (standalone shape-rule reference / consistency check) — `tinynn/analysis.py`
- [x] Graph serialization to/from JSON files — `tinynn/serialize.py`
- [x] Graph visualization (Graphviz DOT export) — `tinynn/viz.py`

## Tier 2 — More operators ✅ (complete)

- [x] Elementwise Add / Sub / Mul (multi-input node semantics, same-shape,
      1D or 2D)
- [x] MatMul (2D x 2D; 2D tensors + multiple Input nodes supported end-to-end
      in IR, interpreter, and codegen)
- [x] Softmax (numerically stable), plus Tanh and Sigmoid
- [ ] LayerNorm (deferred)

## Tier 3 — Optimizer structure ✅ (complete)

- [x] Pass manager (passes are pure `Graph -> Graph` functions) — `tinynn/passes.py`
- [x] Dead code elimination — `tinynn/passes.py`
- [x] Linear + ReLU -> FusedLinearReLU fusion (op supported in IR,
      interpreter, and codegen)
- [x] Const op (value stored in `weight`) + constant folding + algebraic
      simplification (add/sub-zero, mul-one, ReLU idempotence); default
      pipeline is now fold -> simplify -> DCE -> fuse

## Tier 4 — CPU codegen performance ✅ (complete)

- [x] Loop interchange + cache-blocked tiling for Linear / FusedLinearReLU /
      MatMul (`CodegenOptions(tile_size=...)`, `CodegenOptions.fast()`)
- [x] Vectorization-friendly codegen (`-O3 -march=native` via options; the
      interchanged loops stream memory row-major)
- [x] OpenMP multithreading (option-gated, race-free pragma placement,
      `openmp_available()` probe; unavailable on the current dev machine's
      Apple clang, so it is tested via a cleanly-skipped test)
- [x] Repeat/warmup timing mode inside generated binaries + benchmark
      harness (`tinynn/benchmark.py`, results in `results/`)
- [x] Memory planning / buffer reuse (`CodegenOptions(reuse_buffers=True)`:
      liveness-based exact-size slot planner; e.g. 5 logical buffers -> 2
      physical slots on an MLP chain)

## Tier 5 — GPU (blocked on environment)

- [ ] CUDA backend — BLOCKED: the development machine (arm64 macOS) has no
      nvcc/CUDA toolchain, so a GPU backend cannot be compiled or verified
      against the interpreter oracle here. `tests/test_cuda.py` skips cleanly
      without nvcc and fails loudly on a CUDA-capable machine until a real
      backend exists (it never silently passes).

## Tier 6 — Interchange / quantization (mostly complete)

- [x] ONNX import (`tinynn/onnx_import.py`, optional `onnx` dependency):
      MLP-style models — Gemm/MatMul/Relu/Tanh/Sigmoid/Softmax/Add/Sub/Mul/
      Identity, 1D activations with batch-1 squeeze; clear errors otherwise
- [x] int8 quantization (`QuantizedLinear` op + `quantize_linear` pass):
      real int8 inference — symmetric weight quantization, dynamic per-call
      activation quantization, int64 accumulation, float rescale; identical
      rounding semantics in interpreter and C++ so the quantized paths
      compare exactly. Opt-in (not in the default pipeline)
- [x] `QuantizedFusedLinearReLU` op + `quantize_fused_linear_relu` pass:
      same int8 scheme as `QuantizedLinear`, ReLU clamp applied LAST (after
      rescale + bias); `fuse_linear_relu` -> `quantize_fused_linear_relu` ->
      `quantize_linear` lowers a Linear/ReLU MLP fully to int8. Opt-in
- [ ] Batched/2D ONNX activations (future work)

## Tier 7 — Rigor ✅ (complete for current ops; extend alongside new ops)

- [x] Randomized (seeded) graph fuzzing: interpreter vs compiled C++,
      and optimized vs unoptimized graphs — `tests/test_fuzz.py`
- [x] General graph-shape fuzzing (branching, diamonds, multi-consumer,
      dead branches, 2D tensors, full Tier 2 op set, 4-way comparison) —
      `tests/test_fuzz_graphs.py`
- [x] Stronger correctness suite across all ops and passes

## Tier 8 — Demo / story ✅ (complete)

- [x] End-to-end example (`examples/`): build → interpret → optimize →
      compile → verify — `examples/mlp_demo.py`
- [x] Benchmark harness with results written to `results/`
      (`python -m tinynn.benchmark`)
- [x] MNIST-style MLP demo — `examples/mnist_mlp_demo.py`: pure-NumPy
      training on sklearn digits (8x8; honestly labeled, NOT real MNIST),
      ONNX export via `onnx.helper` (no torch), TinyNN import → interpret →
      optimize → compile (naive/optimized/int8) → verify against the
      interpreter oracle → benchmark → `results/mnist/report.md`
- [x] Real CLI (`python -m tinynn` / `python -m tinynn.cli`, argparse only):
      `compile` (pipelines, opt-in `--quantize`, `--fast`, `--benchmark`,
      `--emit-dot`, `--backend cpp|embedded_c`), `run`, `benchmark`,
      `visualize`; `.onnx` and `.json` models; clean errors when `onnx`
      is missing — `tinynn/cli.py`, `tests/test_cli.py`
- [x] Experimental static-memory embedded C backend (`tinynn/embedded.py`,
      `compile_embedded` / `generate_embedded_c`): plain C99, file-scope
      `static` arrays only (no malloc/new/std::vector — enforced by a
      source-inspection test), int8 weights as `static const int8_t`;
      Linear/ReLU/FusedLinearReLU/QuantizedLinear/QuantizedFusedLinearReLU
      only; verified against the interpreter and the C++ backend

## Standing rules

- The interpreter is the correctness oracle: every backend and every
  optimization pass is verified against it.
- One feature at a time; the repo stays green at every step.
- Clean architecture over feature count; no destructive rewrites.
