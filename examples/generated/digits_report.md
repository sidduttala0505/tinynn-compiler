# TinyNN Compile Report

- model: `examples/models/digits_mlp.onnx`
- backend: `cpp`
- quantization: disabled
- embedded/static memory: no

## Original Graph

- nodes: 4
- ops: Input: 1, Linear: 2, ReLU: 1
- inputs: x(64,)
- output: logits(10,)
- parameters: 2410

## Optimized Graph

- nodes: 3
- ops: FusedLinearReLU: 1, Input: 1, Linear: 1
- inputs: x(64,)
- output: logits(10,)
- parameters: 2410

## Benchmark

- iters: 200
- warmup: 20
- OpenMP used: False
- correctness verified: True

| variant | seconds/iter |
|---|---:|
| interpreter | 2.820840e-06 |
| cpu naive | 7.179150e-07 |
| cpu optimized | 4.237500e-07 |

| speedup | value |
|---|---:|
| naive vs interpreter | 3.93x |
| optimized vs naive | 1.69x |
| optimized vs interpreter | 6.66x |

## Output Artifacts

- `examples/generated/digits_float.cpp`
- `examples/generated/digits_graph.dot`
- `examples/generated/digits_int8_embedded.c`
- `examples/generated/digits_report.md`
