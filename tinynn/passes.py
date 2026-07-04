"""Optimization passes over the TinyNN Graph IR.

Every pass is a pure function ``Graph -> Graph``: it never mutates the graph
it is given (``Graph``/``Node`` are frozen dataclasses anyway) and always
returns a freshly constructed ``Graph``. This makes passes trivially
composable and easy to reason about: given the same input graph, a pass
always produces the same output graph.

The NumPy reference interpreter (:func:`tinynn.interpreter.run`) is the
correctness oracle for every pass in this module: for any input graph and any
concrete input values, running the *original* graph through the interpreter
must produce the same output as running the *optimized* graph through the
interpreter (see ``tests/test_passes.py``).

This module provides:

    * :func:`dead_code_elimination` -- drop nodes not reachable (backward,
      from ``graph.output_node``) from the final output.
    * :func:`fuse_linear_relu` -- fuse a ``Linear`` node immediately followed
      by a ``ReLU`` node (and consumed by nothing else) into a single
      ``FusedLinearReLU`` node.
    * :class:`PassManager` -- runs a sequence of passes in order.
    * :func:`default_pipeline` -- a ``PassManager`` with a sensible default
      pass order (dead code elimination, then fusion).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .graph import Graph, Node
from .ops import FUSED_LINEAR_RELU, LINEAR, RELU

__all__ = [
    "dead_code_elimination",
    "fuse_linear_relu",
    "PassManager",
    "default_pipeline",
]

# A pass is any callable taking a Graph and returning a new Graph.
PassFn = Callable[[Graph], Graph]


# --------------------------------------------------------------------------- #
# Dead code elimination
# --------------------------------------------------------------------------- #
def dead_code_elimination(graph: Graph) -> Graph:
    """Return a new graph containing only nodes reachable from the output.

    A node is *live* if it is (transitively) an input of ``graph.output_node``,
    including ``graph.output_node`` itself. Dead nodes (e.g. an ``Input`` node
    whose value is never consumed, or a whole unused branch) are dropped.
    Relative order of the surviving nodes is preserved.
    """
    node_by_name: Dict[str, Node] = graph.node_map()

    live: set = set()
    stack = [graph.output_node]
    while stack:
        name = stack.pop()
        if name in live:
            continue
        live.add(name)
        for inp in node_by_name[name].inputs:
            stack.append(inp)

    kept = tuple(n for n in graph.nodes if n.name in live)
    return Graph(nodes=kept, output_node=graph.output_node)


# --------------------------------------------------------------------------- #
# Linear + ReLU fusion
# --------------------------------------------------------------------------- #
def fuse_linear_relu(graph: Graph) -> Graph:
    """Fuse ``Linear -> ReLU`` pairs into a single ``FusedLinearReLU`` node.

    A ``Linear`` node ``l`` and the ``ReLU`` node ``r`` that consumes it are
    fused when:

        * ``r`` has exactly one input, ``l``;
        * ``l`` is consumed by nothing *other* than ``r`` (fusing would
          otherwise discard a value some other node still needs); and
        * ``l`` is not ``graph.output_node`` (fusing would otherwise discard
          the pre-activation value the graph is supposed to expose).

    The fused node is placed at ``l``'s position in the node order and takes
    ``r``'s name, so any node referencing ``r`` (including ``output_node``)
    keeps working unchanged. A single left-to-right scan is sufficient to
    fuse chains such as ``Linear -> ReLU -> Linear -> ReLU``, since each fused
    node is emitted before later nodes are examined.
    """
    nodes = graph.nodes
    node_by_name: Dict[str, Node] = {n.name: n for n in nodes}

    # Count how many nodes consume each node's output (output_node is treated
    # as an implicit extra consumer of whatever it points to).
    consumer_count: Dict[str, int] = {n.name: 0 for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            consumer_count[inp] += 1
    consumer_count[graph.output_node] += 1

    # Find every Linear node `l` with a fusable ReLU consumer `r`: r's only
    # input is l, l has no other consumer, and l is not the graph output.
    fuse_partner: Dict[str, Node] = {}  # Linear name -> its fusable ReLU node.
    for n in nodes:
        if (
            n.op == RELU
            and len(n.inputs) == 1
            and n.inputs[0] in node_by_name
        ):
            src = node_by_name[n.inputs[0]]
            if (
                src.op == LINEAR
                and consumer_count.get(src.name, 0) == 1
                and src.name != graph.output_node
            ):
                fuse_partner[src.name] = n

    # Emit each fused pair at the Linear node's position (taking the ReLU's
    # name), and drop the ReLU node when we reach its original position.
    fused_relu_names = {r.name for r in fuse_partner.values()}
    new_nodes: List[Node] = []
    for node in nodes:
        if node.name in fuse_partner:
            relu = fuse_partner[node.name]
            new_nodes.append(
                Node(
                    name=relu.name,
                    op=FUSED_LINEAR_RELU,
                    inputs=node.inputs,
                    shape=relu.shape,
                    weight=node.weight,
                    bias=node.bias,
                )
            )
        elif node.name in fused_relu_names:
            continue  # already emitted (fused) at its Linear's position.
        else:
            new_nodes.append(node)

    return Graph(nodes=tuple(new_nodes), output_node=graph.output_node)


# --------------------------------------------------------------------------- #
# Pass manager
# --------------------------------------------------------------------------- #
class PassManager:
    """Runs a sequence of ``Graph -> Graph`` passes in order.

    ``passes`` may be plain functions (each used as its own name) or
    ``(name, fn)`` pairs.
    """

    def __init__(self, passes: Sequence[object]) -> None:
        self._passes: Tuple[Tuple[str, PassFn], ...] = tuple(
            self._normalize(p) for p in passes
        )

    @staticmethod
    def _normalize(p: object) -> Tuple[str, PassFn]:
        if isinstance(p, tuple):
            name, fn = p
            return name, fn
        return getattr(p, "__name__", repr(p)), p  # type: ignore[return-value]

    @property
    def passes(self) -> Tuple[Tuple[str, PassFn], ...]:
        return self._passes

    def run(self, graph: Graph) -> Graph:
        for _name, fn in self._passes:
            graph = fn(graph)
        return graph


def default_pipeline() -> PassManager:
    """Return the default optimization pipeline: DCE, then Linear+ReLU fusion."""
    return PassManager([dead_code_elimination, fuse_linear_relu])
