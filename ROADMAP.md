# TinyNN Compiler — Tier Roadmap

> NOTE: This file was reconstructed from the project owner's prompt on
> 2026-07-06 because no roadmap file existed in the repo. Treat it as the
> source of truth going forward and edit it if the intended roadmap differs.

## Tier 0 — End-to-end foundation ✅ (complete)

- Shared Graph IR (`tinynn/graph.py`, `tinynn/ops.py`): Input / Linear / ReLU / Output
- NumPy reference interpreter as the correctness oracle (`tinynn/interpreter.py`)
- Basic C++ codegen + g++ compile/run wrapper (`tinynn/codegen.py`)
- Correctness tests comparing compiled C++ against the interpreter

## Tier 1 — Graph infrastructure

- [ ] Topological sort (accept unordered node lists, detect cycles)
- [ ] Shape inference (standalone shape-rule reference / consistency check)
- [ ] Graph serialization to/from JSON files
- [ ] Graph visualization (Graphviz DOT export)

## Tier 2 — More operators (only when needed)

- [ ] Elementwise Add / Sub / Mul (requires multi-input node semantics)
- [ ] MatMul
- [ ] Softmax and/or LayerNorm if feasible

## Tier 3 — Optimizer structure

- [ ] Pass manager (passes are pure `Graph -> Graph` functions)
- [ ] Dead code elimination
- [ ] Linear + ReLU -> FusedLinearReLU fusion (op supported in IR,
      interpreter, and codegen)
- [ ] Constant folding / algebraic simplification (needs a Const op first)

## Tier 4 — CPU codegen performance

- [ ] Loop tiling / blocking
- [ ] SIMD hints / vectorization-friendly codegen
- [ ] OpenMP multithreading
- [ ] Memory planning / buffer reuse

## Tier 5 — GPU (deferred until CPU story is solid)

- [ ] CUDA backend

## Tier 6 — Interchange / quantization (deferred)

- [ ] ONNX import
- [ ] int8 quantization

## Tier 7 — Rigor

- [ ] Randomized (seeded) graph fuzzing: interpreter vs compiled C++,
      and optimized vs unoptimized graphs
- [ ] Stronger correctness suite across all ops and passes

## Tier 8 — Demo / story

- [ ] End-to-end example (`examples/`): build → interpret → optimize →
      compile → verify
- [ ] Benchmark harness with results written to `results/`
- [ ] MNIST MLP demo if feasible

## Standing rules

- The interpreter is the correctness oracle: every backend and every
  optimization pass is verified against it.
- One feature at a time; the repo stays green at every step.
- Clean architecture over feature count; no destructive rewrites.
