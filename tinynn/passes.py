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

    * :func:`constant_folding` -- evaluate the constant-only parts of the
      graph ahead of time, replacing each fully-constant node with a
      ``Const`` node holding its computed value.
    * :func:`simplify_algebraic` -- eliminate algebraic-identity nodes
      (add/sub zero, mul one, redundant ReLU) by aliasing them to an
      existing node.
    * :func:`dead_code_elimination` -- drop nodes not reachable (backward,
      from ``graph.output_node``) from the final output.
    * :func:`fuse_linear_relu` -- fuse a ``Linear`` node immediately followed
      by a ``ReLU`` node (and consumed by nothing else) into a single
      ``FusedLinearReLU`` node.
    * :func:`quantize_linear` -- replace every ``Linear`` node with an
      equivalent ``QuantizedLinear`` node (int8 quantized inference; see
      :mod:`tinynn.ops`). Opt-in: numerically different from float Linear,
      so it is NOT part of :func:`default_pipeline`.
    * :class:`PassManager` -- runs a sequence of passes in order.
    * :func:`default_pipeline` -- a ``PassManager`` with a sensible default
      pass order (constant folding, algebraic simplification, dead code
      elimination, then fusion).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph import Graph, Node
from .ops import (
    ADD,
    CONST,
    FUSED_LINEAR_RELU,
    LINEAR,
    MATMUL,
    MUL,
    QUANTIZED_LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    SUB,
    TANH,
)

__all__ = [
    "constant_folding",
    "simplify_algebraic",
    "dead_code_elimination",
    "fuse_linear_relu",
    "quantize_linear",
    "PassManager",
    "default_pipeline",
]

# A pass is any callable taking a Graph and returning a new Graph.
PassFn = Callable[[Graph], Graph]


# --------------------------------------------------------------------------- #
# Constant folding
# --------------------------------------------------------------------------- #
# Ops that constant_folding is willing to evaluate ahead of time. Input and
# Output are deliberately excluded: an Input's value is only known at graph
# run time, and folding Output would discard the "this is the declared
# output" marker for no benefit (its value is just a pass-through anyway).
_FOLDABLE_OPS = {ADD, SUB, MUL, MATMUL, SOFTMAX, TANH, SIGMOID, RELU, LINEAR, FUSED_LINEAR_RELU}


def _fold_value(node: Node, known: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute ``node``'s value given that all of its inputs are known constants.

    Mirrors the exact NumPy expression used by :mod:`tinynn.interpreter` for
    each op, so folding never changes results (beyond ordinary floating point
    associativity, which the interpreter itself is also subject to).
    """
    if node.op == ADD:
        a, b = known[node.inputs[0]], known[node.inputs[1]]
        return a + b
    if node.op == SUB:
        a, b = known[node.inputs[0]], known[node.inputs[1]]
        return a - b
    if node.op == MUL:
        a, b = known[node.inputs[0]], known[node.inputs[1]]
        return a * b
    if node.op == MATMUL:
        a, b = known[node.inputs[0]], known[node.inputs[1]]
        return a @ b
    if node.op == SOFTMAX:
        x = known[node.inputs[0]]
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
    if node.op == TANH:
        return np.tanh(known[node.inputs[0]])
    if node.op == SIGMOID:
        x = known[node.inputs[0]]
        return 1.0 / (1.0 + np.exp(-x))
    if node.op == RELU:
        return np.maximum(known[node.inputs[0]], 0.0)
    if node.op == LINEAR:
        x = known[node.inputs[0]]
        return x @ node.weight + node.bias
    if node.op == FUSED_LINEAR_RELU:
        x = known[node.inputs[0]]
        return np.maximum(x @ node.weight + node.bias, 0.0)
    raise AssertionError(f"_fold_value called on non-foldable op {node.op!r}")  # pragma: no cover


def constant_folding(graph: Graph) -> Graph:
    """Evaluate fully-constant nodes ahead of time, replacing them with ``Const``.

    Performs a single forward scan, keeping a ``name -> np.ndarray`` dict of
    known-constant values seeded by the graph's existing ``Const`` nodes. A
    node folds when every one of its inputs is a known constant and its op is
    one of :data:`_FOLDABLE_OPS` (notably *not* ``Input`` or ``Output``); its
    value is computed with :func:`_fold_value` and the node is replaced with
    ``Node(name=<same name>, op=Const, shape=<same shape>, weight=<value>)``.
    Because the folded node keeps its original name, every downstream
    reference (including ``graph.output_node``) keeps working unchanged, and
    a freshly-folded ``Const`` immediately feeds folding of later nodes in the
    same scan -- so chains of constant computation collapse in one pass.
    """
    known: Dict[str, np.ndarray] = {}
    new_nodes: List[Node] = []

    for node in graph.nodes:
        if node.op == CONST:
            known[node.name] = node.weight
            new_nodes.append(node)
            continue

        if node.op in _FOLDABLE_OPS and all(inp in known for inp in node.inputs):
            value = _fold_value(node, known)
            folded = Node(name=node.name, op=CONST, shape=node.shape, weight=value)
            known[node.name] = value
            new_nodes.append(folded)
            continue

        new_nodes.append(node)

    return Graph(nodes=tuple(new_nodes), output_node=graph.output_node)


# --------------------------------------------------------------------------- #
# Algebraic simplification
# --------------------------------------------------------------------------- #
def _is_zero_const(node: Optional[Node]) -> bool:
    return node is not None and node.op == CONST and bool(np.all(node.weight == 0.0))


def _is_one_const(node: Optional[Node]) -> bool:
    return node is not None and node.op == CONST and bool(np.all(node.weight == 1.0))


def _simplify_algebraic_once(graph: Graph) -> Tuple[Graph, bool]:
    """Run a single scan+rebuild of the algebraic simplification rules.

    Returns ``(new_graph, changed)``. Scans in order, building a
    ``name -> name`` replacement map (resolving chains through the map as it
    goes, so an operand that was itself just replaced resolves to its final
    target within the same scan); then rebuilds the graph once: dropping
    replaced nodes, rewriting every surviving node's inputs through the
    resolved map, and mapping ``output_node`` through it too.
    """
    node_by_name: Dict[str, Node] = graph.node_map()
    replacement: Dict[str, str] = {}

    def resolve(name: str) -> str:
        while name in replacement:
            name = replacement[name]
        return name

    for node in graph.nodes:
        alias: Optional[str] = None

        if node.op == ADD:
            a, b = resolve(node.inputs[0]), resolve(node.inputs[1])
            if _is_zero_const(node_by_name.get(a)):
                alias = b
            elif _is_zero_const(node_by_name.get(b)):
                alias = a
        elif node.op == SUB:
            # Only Sub(x, c) with c == 0 simplifies to x; Sub(c, x) == c - x
            # is not an identity (it's negation), so operand order matters.
            a, b = resolve(node.inputs[0]), resolve(node.inputs[1])
            if _is_zero_const(node_by_name.get(b)):
                alias = a
        elif node.op == MUL:
            a, b = resolve(node.inputs[0]), resolve(node.inputs[1])
            if _is_one_const(node_by_name.get(a)):
                alias = b
            elif _is_one_const(node_by_name.get(b)):
                alias = a
        elif node.op == RELU:
            src_name = resolve(node.inputs[0])
            src = node_by_name.get(src_name)
            if src is not None and src.op in (RELU, FUSED_LINEAR_RELU):
                alias = src_name

        if alias is not None:
            replacement[node.name] = alias

    if not replacement:
        return graph, False

    new_nodes: List[Node] = []
    for node in graph.nodes:
        if node.name in replacement:
            continue
        new_inputs = tuple(resolve(inp) for inp in node.inputs)
        if new_inputs != node.inputs:
            node = Node(
                name=node.name,
                op=node.op,
                inputs=new_inputs,
                shape=node.shape,
                weight=node.weight,
                bias=node.bias,
            )
        new_nodes.append(node)

    new_output = resolve(graph.output_node)
    return Graph(nodes=tuple(new_nodes), output_node=new_output), True


def simplify_algebraic(graph: Graph) -> Graph:
    """Eliminate algebraic-identity nodes by aliasing them to an existing node.

    Exactly four rules, applied by aliasing the identity node's name to an
    existing node's name (never mutating values):

        * ``Add(x, c)`` or ``Add(c, x)`` where ``c`` is a ``Const`` of all
          zeros (same shape as ``x``) -> alias to ``x``.
        * ``Sub(x, c)`` where ``c`` is all zeros -> alias to ``x`` (only this
          operand order; ``Sub(c, x)`` is negation, not an identity).
        * ``Mul(x, c)`` or ``Mul(c, x)`` where ``c`` is a ``Const`` of all
          ones -> alias to ``x``.
        * ``ReLU`` whose single input is itself a ``ReLU`` or
          ``FusedLinearReLU`` node -> alias to that input (``relu`` is
          idempotent).

    Runs the scan+rebuild in :func:`_simplify_algebraic_once` to a fixpoint
    (stopping as soon as a scan makes no changes, with a hard cap of 10
    iterations) so stacked identities -- e.g. ``Add(Add(x, zeros), zeros)``
    -- fully collapse.
    """
    for _ in range(10):
        graph, changed = _simplify_algebraic_once(graph)
        if not changed:
            break
    return graph


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
# Int8 quantization (Tier 6)
# --------------------------------------------------------------------------- #
def quantize_linear(graph: Graph) -> Graph:
    """Replace every ``Linear`` node with an equivalent ``QuantizedLinear`` node.

    Each ``Linear`` node is swapped 1:1 for a ``QuantizedLinear`` node with
    the identical name, inputs, shape, weight and bias -- so every reference
    to it (including ``graph.output_node``) keeps working unchanged, and the
    replacement is purely a change of *op* (float matmul -> int8 quantized
    matmul); see :mod:`tinynn.ops` for the exact quantized semantics.

    ``FusedLinearReLU`` nodes are deliberately left untouched: fusing and
    quantizing are only composable, today, in the sense that running this
    pass after :func:`fuse_linear_relu` is a no-op for already-fused nodes
    (it quantizes only the remaining, unfused ``Linear`` nodes). A quantized
    fused ``Linear+ReLU`` op is future work.

    Not part of :func:`default_pipeline`: quantization changes numerics
    (int8 rounding + rescale), so callers must opt in explicitly.
    """
    new_nodes: List[Node] = []
    for node in graph.nodes:
        if node.op == LINEAR:
            new_nodes.append(
                Node(
                    name=node.name,
                    op=QUANTIZED_LINEAR,
                    inputs=node.inputs,
                    shape=node.shape,
                    weight=node.weight,
                    bias=node.bias,
                )
            )
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
    """Return the default optimization pipeline.

    Order: constant folding (creates ``Const`` nodes, some now dead), then
    algebraic simplification (creates more dead nodes, e.g. eliminated
    zero/one Consts and redundant ReLUs), then dead code elimination (sweeps
    every orphaned node left behind by the previous two passes), then
    Linear+ReLU fusion last.
    """
    return PassManager(
        [constant_folding, simplify_algebraic, dead_code_elimination, fuse_linear_relu]
    )
