# CUDA Benchmark Methodology

Run:

```bash
python -m tinynn.cuda_benchmark --results-dir results/cuda --iters 10000 --warmup 1000
```

Each workload is first checked against TinyNN's NumPy interpreter at `rtol=1e-5`
and `atol=1e-6`. The generated binary allocates device buffers and uploads model
parameters and one input once, runs the requested warmup iterations, then records
CUDA events around repeated kernel launches.

Reported `cuda_event_seconds_per_inference` measures device kernel execution only.
It excludes process startup, file I/O, device allocation, host-to-device input
transfer, parameter upload, and device-to-host output transfer. Results are
workload-, GPU-, driver-, compiler-, and configuration-specific. Do not present
them as a cross-framework speedup without a separately controlled comparison.
