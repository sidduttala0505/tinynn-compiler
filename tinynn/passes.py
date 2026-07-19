"""Graph optimization passes.

Each pass is just a function that takes a Graph and returns a new one - it never
edits the input (Graph is frozen anyway). The interpreter is the oracle: an
optimized graph has to give the same answers as the original.

Passes here: constant_folding, simplify_algebraic, dead_code_elimination,
fuse_linear_relu, plus quantize_linear / quantize_fused_linear_relu (opt-in,
they change the numbers). PassManager chains them; default_pipeline() is the
usual order.
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
    QUANTIZED_FUSED_LINEAR_RELU,
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
    "quantize_fused_linear_relu",
    "PassManager",
    "default_pipeline",
]

PassFn = Callable[[Graph], Graph]


# ops we're willing to evaluate at compile time. Input isn't here (value only
# known at runtime) and neither is Output (no point, it's just a passthrough).
_FOLDABLE_OPS = {ADD, SUB, MUL, MATMUL, SOFTMAX, TANH, SIGMOID, RELU, LINEAR, FUSED_LINEAR_RELU}


def _fold_value(node: Node, known: Dict[str, np.ndarray]) -> np.ndarray:
    # same math as the interpreter, just run now instead of at runtime
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
    """Precompute nodes whose inputs are all constant and turn them into Consts.

    One left-to-right pass. We track the known constant values as we go (starting
    from the existing Const nodes); when a node's inputs are all known we compute
    it and swap it for a Const with the same name. Keeping the name means nothing
    downstream breaks, and the new Const can feed the next fold - so a whole chain
    of constant math collapses in a single pass.
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


def _is_zero_const(node: Optional[Node]) -> bool:
    return node is not None and node.op == CONST and bool(np.all(node.weight == 0.0))


def _is_one_const(node: Optional[Node]) -> bool:
    return node is not None and node.op == CONST and bool(np.all(node.weight == 1.0))


def _simplify_algebraic_once(graph: Graph) -> Tuple[Graph, bool]:
    # one scan + rebuild. builds a name->name "replace this with that" map, then
    # rewrites every node's inputs (and the output) through it. returns
    # (new_graph, did_anything).
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
            # only x - 0 simplifies. 0 - x is negation, not an identity, so order matters
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
            if src is not None and src.op in (
                RELU,
                FUSED_LINEAR_RELU,
                QUANTIZED_FUSED_LINEAR_RELU,
            ):
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
    """Drop no-op nodes by pointing whatever used them at their real input.

    The rules:
      - x + 0  or  0 + x   ->  x
      - x - 0              ->  x   (but not 0 - x)
      - x * 1  or  1 * x   ->  x
      - relu(relu(x))      ->  relu(x)   (relu, and the fused relu ops, are
                                          idempotent - output's already >= 0)

    Runs until nothing changes (capped at 10 rounds) so stacked no-ops like
    (x + 0) + 0 collapse all the way.
    """
    for _ in range(10):
        graph, changed = _simplify_algebraic_once(graph)
        if not changed:
            break
    return graph


def dead_code_elimination(graph: Graph) -> Graph:
    """Keep only the nodes the output actually depends on; drop the rest.

    Walk backwards from output_node collecting everything reachable, then throw
    away anything we didn't hit (unused branches, dead inputs, ...).
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


def fuse_linear_relu(graph: Graph) -> Graph:
    """Merge a Linear directly followed by a ReLU into one FusedLinearReLU node.

    Only fuse when the ReLU's single input is the Linear, nothing else uses that
    Linear, and the Linear isn't the graph output (we'd lose the pre-activation
    value otherwise). The fused node takes the ReLU's name and the Linear's spot,
    so references still resolve. One left-to-right pass handles chains fine.
    """
    nodes = graph.nodes
    node_by_name: Dict[str, Node] = {n.name: n for n in nodes}

    # how many things use each node (the output counts as one extra user)
    consumer_count: Dict[str, int] = {n.name: 0 for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            consumer_count[inp] += 1
    consumer_count[graph.output_node] += 1

    # map each fusable Linear to the ReLU that eats it
    fuse_partner: Dict[str, Node] = {}
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

    # emit the fused node where the Linear was, and skip the ReLU later on
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
            continue  # already handled at the Linear's spot
        else:
            new_nodes.append(node)

    return Graph(nodes=tuple(new_nodes), output_node=graph.output_node)


def quantize_linear(graph: Graph) -> Graph:
    """Swap every Linear node for a QuantizedLinear (same name/weights/etc).

    Only the op changes (float -> int8). FusedLinearReLU is left alone - use
    quantize_fused_linear_relu for those. Not in default_pipeline since it
    changes the actual numbers - you opt in.
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


def quantize_fused_linear_relu(graph: Graph) -> Graph:
    """Like quantize_linear but for FusedLinearReLU -> QuantizedFusedLinearReLU.

    Only the op changes. Bare Linears are left for quantize_linear. To quantize a
    Linear -> ReLU chain, run fuse_linear_relu first, then this. Opt-in (changes
    the numbers), not in default_pipeline.
    """
    new_nodes: List[Node] = []
    for node in graph.nodes:
        if node.op == FUSED_LINEAR_RELU:
            new_nodes.append(
                Node(
                    name=node.name,
                    op=QUANTIZED_FUSED_LINEAR_RELU,
                    inputs=node.inputs,
                    shape=node.shape,
                    weight=node.weight,
                    bias=node.bias,
                )
            )
        else:
            new_nodes.append(node)

    return Graph(nodes=tuple(new_nodes), output_node=graph.output_node)


class PassManager:
    """Runs a list of passes one after another. Each pass can be a plain function
    or a (name, function) pair."""

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
    # fold first (makes dead consts), then simplify (makes more dead nodes),
    # then DCE to sweep them all up, then fuse. order matters here.
    return PassManager(
        [constant_folding, simplify_algebraic, dead_code_elimination, fuse_linear_relu]
    )
