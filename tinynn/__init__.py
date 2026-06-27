"""TinyNN: a tiny compiler for simple neural networks.

Phase 0, Step 1 exposes only the shared Graph IR.
"""

from __future__ import annotations

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

__all__ = [
    "INPUT",
    "LINEAR",
    "RELU",
    "OUTPUT",
    "SUPPORTED_OPS",
    "Node",
    "Graph",
    "GraphBuilder",
]
