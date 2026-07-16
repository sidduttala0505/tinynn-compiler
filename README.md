# TinyNN Compiler

TinyNN is a small ahead-of-time neural-network compiler for edge inference.
It has a shared Graph IR, a NumPy interpreter used as the correctness oracle,
optimization passes, C++/embedded-C codegen, ONNX import, int8 quantization,
and a minimal CUDA backend vertical slice.

The project is organized by tiers in [ROADMAP.md](ROADMAP.md). The standing
rule is that generated backends and optimization passes are validated against
the interpreter.

## Backends

- C++ backend (`tinynn/codegen.py`): the main generated-code path.
- Embedded C backend (`tinynn/embedded.py`): experimental static-memory C99
  path for the supported Linear/ReLU/int8 subset.
- Tier 5 CUDA backend (`tinynn/cuda.py`): minimal vertical slice only.

## Tier 5 CUDA backend

The CUDA backend is not complete GPU support. It is a narrow vertical slice
that generates CUDA C++ code, compiles it with `nvcc`, executes a real CUDA
binary, and compares GPU output against the interpreter oracle.

Current supported CUDA path:

- 1D `Input`
- `Linear`
- `ReLU`
- graph output through the selected output node or an explicit `Output` node

Broader op coverage, batched tensors, and general GPU scheduling are future
work. CUDA tests skip cleanly when `nvcc`, the CUDA runtime, or a usable GPU is
unavailable.

TACC rtx-small GPU validation setup:

```bash
module reset
module load gcc/13.2.0
module load cuda/12.8
source .venv/bin/activate
```

CUDA validation commands:

```bash
python -m pytest tests/test_cuda.py -v
python -m pytest tests
```

Latest reported TACC rtx-small result:

```text
python -m pytest tests
237 passed

tests/test_cuda.py::test_generate_cuda_contains_kernels PASSED
tests/test_cuda.py::test_cuda_input_linear_relu_matches_interpreter PASSED
tests/test_cuda.py::test_cuda_explicit_output_node_matches_interpreter PASSED
```

## Development

Install the package in a virtual environment, then run the CPU/non-CUDA tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -k "not cuda"
```

Run the full suite when the local toolchain supports it:

```bash
python -m pytest tests
```

Some backend tests are conditional: C++ tests require `g++`, embedded-C tests
require a C/C++ compiler, ONNX tests require the optional `onnx` dependency,
and CUDA tests require `nvcc` plus a working CUDA runtime/GPU.
