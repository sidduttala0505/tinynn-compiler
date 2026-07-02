"""TinyNN: a tiny compiler for simple neural networks.

Tier 0 exposes the shared Graph IR, the NumPy reference interpreter, and the
C++ code generator / compiler driver.
"""

from __future__ import annotations

from .codegen import CompiledModel, compile_graph, generate_cpp
from .graph import (
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

__all__ = [
    "INPUT",
    "LINEAR",
    "RELU",
    "OUTPUT",
    "SUPPORTED_OPS",
    "Node",
    "Graph",
    "GraphBuilder",
    "run",
    "CompiledModel",
    "compile_graph",
    "generate_cpp",
]
