"""TinyNN: a tiny compiler for simple neural networks.

Exposes the shared Graph IR, the NumPy reference interpreter (the project's
correctness oracle), the C++ code generator / compiler driver, optimization
passes, and graph tooling (topological sort, shape inference, JSON
serialization, Graphviz DOT export).
"""

from __future__ import annotations

from .analysis import infer_shapes, topological_sort
from .codegen import CompiledModel, compile_graph, generate_cpp
from .graph import (
    FUSED_LINEAR_RELU,
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
from .passes import (
    PassManager,
    dead_code_elimination,
    default_pipeline,
    fuse_linear_relu,
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
    "run",
    "CompiledModel",
    "compile_graph",
    "generate_cpp",
    "PassManager",
    "dead_code_elimination",
    "fuse_linear_relu",
    "default_pipeline",
    "topological_sort",
    "infer_shapes",
    "save_json",
    "load_json",
    "to_dot",
    "save_dot",
]
