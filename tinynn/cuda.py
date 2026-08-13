"""CUDA backend for TinyNN's 1D MLP graph slice.

Normal generated binaries read and write float32 files. Their ``--benchmark``
mode performs one-time setup, warms up kernels, then measures device execution
with CUDA events.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .graph import FUSED_LINEAR_RELU, Graph, INPUT, LINEAR, OUTPUT, RELU

__all__ = [
    "CudaCompiledModel",
    "compile_graph_cuda",
    "generate_cuda",
]

_INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")
_BENCHMARK_LINE_RE = re.compile(r"^CUDA_EVENT_SECONDS_PER_ITER=([0-9.eE+-]+)$")
_SUPPORTED = {INPUT, LINEAR, FUSED_LINEAR_RELU, RELU, OUTPUT}
_WRAP_WIDTH = 88


def _sanitize_name(name: str) -> str:
    return "v_" + _INVALID_CHARS_RE.sub("_", name)


def _build_identifier_map(graph: Graph) -> Dict[str, str]:
    used: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for node in graph.nodes:
        base = _sanitize_name(node.name)
        if base not in used:
            used[base] = 1
            mapping[node.name] = base
            continue

        used[base] += 1
        candidate = f"{base}_{used[base]}"
        while candidate in used:
            used[base] += 1
            candidate = f"{base}_{used[base]}"
        used[candidate] = 1
        mapping[node.name] = candidate
    return mapping


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _validate_for_cuda(graph: Graph) -> None:
    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    if not input_nodes:
        raise ValueError("CUDA backend requires at least one Input node")

    defined: set[str] = set()
    for node in graph.nodes:
        if node.op not in _SUPPORTED:
            raise ValueError(
                f"CUDA backend does not support op {node.op!r} on node "
                f"{node.name!r}; supported ops are {sorted(_SUPPORTED)}"
            )
        if len(node.shape) != 1:
            raise ValueError(
                f"CUDA backend currently supports only 1D node shapes, but "
                f"node {node.name!r} has shape {node.shape!r}"
            )
        for inp in node.inputs:
            if inp not in defined:
                raise ValueError(
                    f"Node {node.name!r} references input {inp!r} before it "
                    "has been computed"
                )
        defined.add(node.name)


def _format_float_array(values: np.ndarray) -> str:
    tokens = [f"{float(v):.9g}" for v in values.astype(np.float32).ravel()]
    tokens = [
        (tok if any(c in tok for c in ".eE") else tok + ".0") + "f"
        for tok in tokens
    ]
    if not tokens:
        return ""
    lines: List[str] = []
    current = "    "
    for i, tok in enumerate(tokens):
        piece = tok + ("," if i != len(tokens) - 1 else "")
        if current.strip() and len(current) + 1 + len(piece) > _WRAP_WIDTH:
            lines.append(current)
            current = "    " + piece
        else:
            current = (current + " " + piece) if current.strip() else (current + piece)
    lines.append(current)
    return "\n".join(lines)


def generate_cuda(graph: Graph) -> str:
    """Return standalone CUDA/C++ for the supported graph slice.

    A ``FusedLinearReLU`` node receives its own output allocation and emits one
    kernel. This deliberately avoids materializing the corresponding linear
    activation buffer. Benchmark mode times only repeated kernel launches after
    input upload, parameter upload, and all allocations have completed.
    """
    _validate_for_cuda(graph)

    ids = _build_identifier_map(graph)
    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    output_node = graph.get_node(graph.output_node)
    globals_lines: List[str] = []
    setup_lines: List[str] = []
    execute_lines: List[str] = []
    cleanup_lines: List[str] = []
    allocated: List[str] = []

    for node in graph.nodes:
        ident = ids[node.name]
        n = _numel(node.shape)

        if node.op == INPUT:
            input_index = input_nodes.index(node)
            setup_lines.extend(
                [
                    f"    float* {ident} = nullptr;",
                    f"    CUDA_CHECK(cudaMalloc(&{ident}, {n} * sizeof(float)));",
                    f"    if (read_input_file(argv[input_arg_offset + {input_index}], hbuf, {n}) != 0) return 1;",
                    f"    CUDA_CHECK(cudaMemcpy({ident}, hbuf.data(), {n} * sizeof(float), cudaMemcpyHostToDevice));",
                    "",
                ]
            )
            allocated.append(ident)
        elif node.op in (LINEAR, FUSED_LINEAR_RELU):
            src = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            w_ident = f"{ident}_w"
            b_ident = f"{ident}_b"
            kernel = (
                "fused_linear_relu_kernel"
                if node.op == FUSED_LINEAR_RELU
                else "linear_kernel"
            )
            globals_lines.extend(
                [
                    f"static const float {w_ident}[] = {{",
                    _format_float_array(node.weight),
                    "};",
                    f"static const float {b_ident}[] = {{",
                    _format_float_array(node.bias),
                    "};",
                    "",
                ]
            )
            setup_lines.extend(
                [
                    f"    float* {ident} = nullptr;",
                    f"    float* d_{w_ident} = nullptr;",
                    f"    float* d_{b_ident} = nullptr;",
                    f"    CUDA_CHECK(cudaMalloc(&{ident}, {out_features} * sizeof(float)));",
                    f"    CUDA_CHECK(cudaMalloc(&d_{w_ident}, {in_features * out_features} * sizeof(float)));",
                    f"    CUDA_CHECK(cudaMalloc(&d_{b_ident}, {out_features} * sizeof(float)));",
                    f"    CUDA_CHECK(cudaMemcpy(d_{w_ident}, {w_ident}, {in_features * out_features} * sizeof(float), cudaMemcpyHostToDevice));",
                    f"    CUDA_CHECK(cudaMemcpy(d_{b_ident}, {b_ident}, {out_features} * sizeof(float), cudaMemcpyHostToDevice));",
                    "",
                ]
            )
            execute_lines.extend(
                [
                    f"    {kernel}<<<blocks({out_features}), 256>>>({src}, d_{w_ident}, d_{b_ident}, {ident}, {in_features}, {out_features});",
                    "    CUDA_CHECK(cudaGetLastError());",
                ]
            )
            allocated.extend([ident, f"d_{w_ident}", f"d_{b_ident}"])
        elif node.op == RELU:
            src = ids[node.inputs[0]]
            setup_lines.extend(
                [
                    f"    float* {ident} = nullptr;",
                    f"    CUDA_CHECK(cudaMalloc(&{ident}, {n} * sizeof(float)));",
                    "",
                ]
            )
            execute_lines.extend(
                [
                    f"    relu_kernel<<<blocks({n}), 256>>>({src}, {ident}, {n});",
                    "    CUDA_CHECK(cudaGetLastError());",
                ]
            )
            allocated.append(ident)
        elif node.op == OUTPUT:
            setup_lines.extend([f"    float* {ident} = {ids[node.inputs[0]]};", ""])

    for ident in reversed(allocated):
        cleanup_lines.append(f"    cudaFree({ident});")

    output_size = _numel(output_node.shape)
    input_count = len(input_nodes)
    benchmark_execute = ["    " + line for line in execute_lines]
    return "\n".join(
        [
            "#include <cuda_runtime.h>",
            "",
            "#include <cerrno>",
            "#include <cstddef>",
            "#include <cstdio>",
            "#include <cstdlib>",
            "#include <fstream>",
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "",
            "#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { std::cerr << \"CUDA error: \" << cudaGetErrorString(err) << \"\\n\"; return 1; } } while (0)",
            "",
            *globals_lines,
            "__global__ void linear_kernel(const float* x, const float* w, const float* b,",
            "                              float* y, int in_features, int out_features) {",
            "    int j = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (j >= out_features) return;",
            "    float acc = b[j];",
            "    for (int i = 0; i < in_features; ++i) acc += x[i] * w[i * out_features + j];",
            "    y[j] = acc;",
            "}",
            "",
            "__global__ void fused_linear_relu_kernel(const float* x, const float* w, const float* b,",
            "                                         float* y, int in_features, int out_features) {",
            "    int j = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (j >= out_features) return;",
            "    float acc = b[j];",
            "    for (int i = 0; i < in_features; ++i) acc += x[i] * w[i * out_features + j];",
            "    y[j] = acc > 0.0f ? acc : 0.0f;",
            "}",
            "",
            "__global__ void relu_kernel(const float* x, float* y, int n) {",
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (i >= n) return;",
            "    float v = x[i];",
            "    y[i] = v > 0.0f ? v : 0.0f;",
            "}",
            "",
            "int blocks(int n) { return (n + 255) / 256; }",
            "",
            "int parse_nonnegative_int(const char* value, int* parsed) {",
            "    char* end = nullptr;",
            "    errno = 0;",
            "    long result = std::strtol(value, &end, 10);",
            "    if (errno != 0 || end == value || *end != '\\0' || result < 0 || result > 2147483647L) return 1;",
            "    *parsed = static_cast<int>(result);",
            "    return 0;",
            "}",
            "",
            "int read_input_file(const char* path, std::vector<float>& dst, std::size_t n) {",
            "    dst.assign(n, 0.0f);",
            "    std::ifstream in(path, std::ios::binary);",
            "    if (!in) { std::cerr << \"could not open input \" << path << \"\\n\"; return 1; }",
            "    in.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n * sizeof(float)));",
            "    if (in.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {",
            "        std::cerr << \"short read from input \" << path << \"\\n\";",
            "        return 1;",
            "    }",
            "    return 0;",
            "}",
            "",
            "int write_output_file(const char* path, const std::vector<float>& src) {",
            "    std::ofstream out(path, std::ios::binary);",
            "    if (!out) { std::cerr << \"could not open output \" << path << \"\\n\"; return 1; }",
            "    out.write(reinterpret_cast<const char*>(src.data()), static_cast<std::streamsize>(src.size() * sizeof(float)));",
            "    return out ? 0 : 1;",
            "}",
            "",
            "int main(int argc, char** argv) {",
            '    const bool benchmark_mode = argc > 1 && std::string(argv[1]) == "--benchmark";',
            f"    if ((!benchmark_mode && argc != {input_count + 2}) ||",
            f"        (benchmark_mode && argc != {input_count + 4})) {{",
            '        std::cerr << "usage: " << argv[0] << " <input...> <output>\\n";',
            '        std::cerr << "   or: " << argv[0] << " --benchmark <input...> <iters> <warmup>\\n";',
            "        return 1;",
            "    }",
            "    const int input_arg_offset = benchmark_mode ? 2 : 1;",
            "    std::vector<float> hbuf;",
            *setup_lines,
            "    if (benchmark_mode) {",
            "        int iters = 0;",
            "        int warmup = 0;",
            f"        if (parse_nonnegative_int(argv[input_arg_offset + {input_count}], &iters) != 0 || iters == 0 ||",
            f"            parse_nonnegative_int(argv[input_arg_offset + {input_count + 1}], &warmup) != 0) {{",
            '            std::cerr << "iters must be positive and warmup must be nonnegative\\n";',
            "            return 1;",
            "        }",
            "        for (int i = 0; i < warmup; ++i) {",
            *benchmark_execute,
            "        }",
            "        CUDA_CHECK(cudaDeviceSynchronize());",
            "        cudaEvent_t start = nullptr;",
            "        cudaEvent_t stop = nullptr;",
            "        CUDA_CHECK(cudaEventCreate(&start));",
            "        CUDA_CHECK(cudaEventCreate(&stop));",
            "        CUDA_CHECK(cudaEventRecord(start));",
            "        for (int i = 0; i < iters; ++i) {",
            *benchmark_execute,
            "        }",
            "        CUDA_CHECK(cudaEventRecord(stop));",
            "        CUDA_CHECK(cudaEventSynchronize(stop));",
            "        float elapsed_ms = 0.0f;",
            "        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));",
            '        std::printf("CUDA_EVENT_SECONDS_PER_ITER=%.9g\\n", static_cast<double>(elapsed_ms) / (1000.0 * iters));',
            "        CUDA_CHECK(cudaEventDestroy(start));",
            "        CUDA_CHECK(cudaEventDestroy(stop));",
            *cleanup_lines,
            "        return 0;",
            "    }",
            *execute_lines,
            "    CUDA_CHECK(cudaDeviceSynchronize());",
            f"    std::vector<float> out({output_size});",
            f"    CUDA_CHECK(cudaMemcpy(out.data(), {ids[graph.output_node]}, {output_size} * sizeof(float), cudaMemcpyDeviceToHost));",
            f"    if (write_output_file(argv[{input_count + 1}], out) != 0) return 1;",
            *cleanup_lines,
            "    return 0;",
            "}",
            "",
        ]
    )


@dataclass(frozen=True)
class CudaCompiledModel:
    cu_path: Path
    binary_path: Path
    output_shape: Tuple[int, ...]
    input_specs: Tuple[Tuple[str, Tuple[int, ...]], ...]
    work_dir: Path

    def run(self, inputs) -> np.ndarray:
        """Run the generated binary once and return its output."""
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp:
            tmp_path = Path(tmp)
            input_paths = self._write_inputs(inputs, tmp_path)
            output_path = tmp_path / "output.bin"
            proc = subprocess.run(
                [str(self.binary_path), *(str(p) for p in input_paths), str(output_path)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CUDA model binary {self.binary_path} exited with code "
                    f"{proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )

            out = np.fromfile(output_path, dtype=np.float32)
            expected_out = _numel(self.output_shape)
            if out.size != expected_out:
                raise ValueError(
                    f"CUDA model produced {out.size} values, expected "
                    f"{expected_out} for output shape {self.output_shape}"
                )
            return out.astype(np.float64).reshape(self.output_shape)

    def benchmark(self, inputs, iters: int = 1000, warmup: int = 100) -> float:
        """Return CUDA-event seconds/inference after one-time model setup.

        The timed interval excludes process startup, file I/O, host-to-device
        input upload, parameter upload, and allocation. It includes all graph
        kernel launches for each inference.
        """
        if not isinstance(iters, int) or iters <= 0:
            raise ValueError(f"iters must be a positive integer, got {iters!r}")
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError(f"warmup must be a nonnegative integer, got {warmup!r}")

        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp:
            input_paths = self._write_inputs(inputs, Path(tmp))
            proc = subprocess.run(
                [
                    str(self.binary_path),
                    "--benchmark",
                    *(str(p) for p in input_paths),
                    str(iters),
                    str(warmup),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CUDA benchmark binary {self.binary_path} exited with code "
                    f"{proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            for line in proc.stdout.splitlines():
                match = _BENCHMARK_LINE_RE.match(line.strip())
                if match:
                    seconds = float(match.group(1))
                    if seconds > 0.0:
                        return seconds
            raise RuntimeError(
                "CUDA benchmark did not emit a positive "
                f"CUDA_EVENT_SECONDS_PER_ITER line. stdout:\n{proc.stdout}"
            )

    def _write_inputs(self, inputs, directory: Path) -> List[Path]:
        provided = self._resolve_inputs(inputs)
        input_paths: List[Path] = []
        for input_index, (name, shape) in enumerate(self.input_specs):
            arr = np.asarray(provided[name], dtype=np.float32)
            expected = _numel(shape)
            if arr.size != expected:
                raise ValueError(
                    f"Input {name!r} expected {expected} values (shape "
                    f"{shape}), got {arr.size} (shape {arr.shape})"
                )
            path = directory / f"input_{input_index}.bin"
            arr.ravel(order="C").tofile(path)
            input_paths.append(path)
        return input_paths

    def _resolve_inputs(self, inputs) -> Dict[str, np.ndarray]:
        if isinstance(inputs, dict):
            missing = [name for name, _ in self.input_specs if name not in inputs]
            if missing:
                raise KeyError(
                    f"Missing value(s) for Input node(s): {missing}; this model "
                    f"expects inputs for {[name for name, _ in self.input_specs]}"
                )
            return inputs

        if len(self.input_specs) != 1:
            raise ValueError(
                "A bare array may only be passed as `inputs` when the CUDA "
                f"model has exactly one Input node; this model has "
                f"{len(self.input_specs)} ({[name for name, _ in self.input_specs]})"
            )
        return {self.input_specs[0][0]: inputs}


def compile_graph_cuda(
    graph: Graph,
    output_dir,
    binary_name: str = "tinynn_cuda_model",
    cu_name: str = "tinynn_cuda_model.cu",
    nvcc: str = "nvcc",
    extra_flags: List[str] | None = None,
) -> CudaCompiledModel:
    """Generate CUDA for ``graph``, compile it with nvcc, and return a runner."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cu_path = output_dir / cu_name
    binary_path = output_dir / binary_name
    cu_path.write_text(generate_cuda(graph))

    cmd = [nvcc, str(cu_path), "-O2", "-o", str(binary_path)]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"CUDA compiler {nvcc!r} not found") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Compilation of {cu_path} with {nvcc!r} failed "
            f"(exit code {proc.returncode}):\n{proc.stderr}"
        )

    return CudaCompiledModel(
        cu_path=cu_path,
        binary_path=binary_path,
        output_shape=graph.get_node(graph.output_node).shape,
        input_specs=tuple((n.name, n.shape) for n in graph.nodes if n.op == INPUT),
        work_dir=output_dir,
    )
