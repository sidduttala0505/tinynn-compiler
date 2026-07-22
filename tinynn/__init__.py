"""TinyNN - a little compiler for small neural nets.

Pulls the useful stuff up to the top level: the graph IR, the interpreter, the
C++ codegen, the passes, and the graph tools (toposort, shapes, JSON, dot).
"""

from __future__ import annotations

from .analysis import infer_shapes, topological_sort
from .benchmark import benchmark_graph
from .codegen import (
    CodegenOptions,
    CompiledModel,
    compile_graph,
    generate_cpp,
    openmp_available,
)
from .embedded import compile_embedded, generate_embedded_c
from .graph import (
    ADD,
    CONST,
    FUSED_LINEAR_RELU,
    MATMUL,
    MUL,
    QUANTIZED_FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
    SIGMOID,
    SOFTMAX,
    SUB,
    TANH,
    INPUT,
    LINEAR,
    OUTPUT,
    RELU,
    SUPPORTED_OPS,
    Graph,
    GraphBuilder,
    Node,
)
from .interpreter import run
from .onnx_import import import_onnx
from .passes import (
    PassManager,
    constant_folding,
    dead_code_elimination,
    default_pipeline,
    fuse_linear_relu,
    quantize_fused_linear_relu,
    quantize_linear,
    simplify_algebraic,
)
from .serialize import load_json, save_json
from .viz import save_dot, to_dot

__all__ = [
    "INPUT",
    "LINEAR",
    "RELU",
    "OUTPUT",
    "FUSED_LINEAR_RELU",
    "SUPPORTED_OPS",
    "Node",
    "Graph",
    "GraphBuilder",
    "ADD",
    "SUB",
    "MUL",
    "MATMUL",
    "SOFTMAX",
    "TANH",
    "SIGMOID",
    "run",
    "CompiledModel",
    "CodegenOptions",
    "compile_graph",
    "generate_cpp",
    "openmp_available",
    "compile_embedded",
    "generate_embedded_c",
    "benchmark_graph",
    "CONST",
    "QUANTIZED_LINEAR",
    "QUANTIZED_FUSED_LINEAR_RELU",
    "PassManager",
    "constant_folding",
    "simplify_algebraic",
    "dead_code_elimination",
    "fuse_linear_relu",
    "quantize_linear",
    "quantize_fused_linear_relu",
    "default_pipeline",
    "import_onnx",
    "topological_sort",
    "infer_shapes",
    "save_json",
    "load_json",
    "to_dot",
    "save_dot",
]
