# Benchmark: matmul_128

- when: 2026-07-07T22:57:25.055888+00:00
- graph: 3 nodes, ops {'Input': 2, 'MatMul': 1}
- iters: 200 (warmup 20), OpenMP: False
- correctness: all compiled variants verified against the interpreter (rtol/atol 1e-09) before timing

| variant | seconds/iter | vs interpreter |
|---|---|---|
| interpreter (NumPy) | 1.289e-05 | 1.0x |
| C++ naive (-O2) | 1.266e-03 | 0.01x |
| C++ optimized (passes + fast codegen) | 4.837e-04 | 0.03x |

Optimized vs naive: 2.62x. Numbers are machine- and load-dependent; treat as indicative only. Note that the NumPy interpreter delegates matmul-heavy work to an optimized BLAS (e.g. Apple Accelerate), so it can outrun TinyNN's scalar C++ loops; the meaningful codegen comparison is optimized vs naive.
