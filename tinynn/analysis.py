"""Two helpers: sort nodes into a valid order, and recompute shapes.

infer_shapes mostly duplicates the checks Graph already does at construction,
but it's handy as a standalone sanity check and it documents the shape rules
in one place.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .graph import Graph, Node
from .ops import ADD, CONST, INPUT, MATMUL, MUL, SOFTMAX, SUB


def topological_sort(nodes) -> Tuple[Node, ...]:
    """Reorder nodes so every node comes after its inputs (Kahn's algorithm).

    Stable - if two nodes are both ready, the one that came first in the input
    stays first, so an already-sorted list comes back unchanged. Raises if there
    are duplicate names, a missing input, or a cycle.
    """
    node_list: List[Node] = list(nodes)

    name_to_node: Dict[str, Node] = {}
    for n in node_list:
        if n.name in name_to_node:
            raise ValueError(f"Duplicate node name: {n.name!r}")
        name_to_node[n.name] = n

    all_names = set(name_to_node)
    for n in node_list:
        for inp in n.inputs:
            if inp not in all_names:
                raise ValueError(
                    f"Node {n.name!r} references unknown input {inp!r}"
                )

    order_index: Dict[str, int] = {n.name: i for i, n in enumerate(node_list)}

    indegree: Dict[str, int] = {n.name: len(n.inputs) for n in node_list}
    dependents: Dict[str, List[str]] = {n.name: [] for n in node_list}
    for n in node_list:
        for inp in n.inputs:
            dependents[inp].append(n.name)

    # nodes with no remaining deps, kept in original order so ties are stable
    ready: List[str] = sorted(
        (n.name for n in node_list if indegree[n.name] == 0),
        key=lambda nm: order_index[nm],
    )

    result: List[Node] = []
    processed: set = set()
    while ready:
        name = ready.pop(0)
        processed.add(name)
        result.append(name_to_node[name])
        newly_ready = []
        for dep in dependents[name]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                newly_ready.append(dep)
        if newly_ready:
            ready.extend(newly_ready)
            ready.sort(key=lambda nm: order_index[nm])

    if len(result) != len(node_list):
        remaining = [n.name for n in node_list if n.name not in processed]
        raise ValueError(
            "Cycle detected in graph nodes; could not order the following "
            f"nodes (they participate in or depend on a cycle): {remaining}"
        )

    return tuple(result)


def infer_shapes(graph: Graph) -> Dict[str, Tuple[int, ...]]:
    """Work out each node's output shape and return a name -> shape dict.

    Raises if a computed shape doesn't match what the node claims, or if two
    inputs don't line up (e.g. Linear weight vs its input width).
    """
    inferred: Dict[str, Tuple[int, ...]] = {}

    for node in graph.nodes:
        if node.op == INPUT:
            shape = node.shape
        elif node.op == CONST:
            # has to be checked before the 2D-weight case below - a Const also
            # keeps its value in `weight` but has no inputs, so it'd get treated
            # as a Linear and blow up looking for an input that isn't there
            shape = node.weight.shape
        elif node.weight is not None and node.weight.ndim == 2:
            # Linear-like (Linear or FusedLinearReLU): anything with a 2D weight
            if len(node.inputs) != 1:
                raise ValueError(
                    f"Linear-like node {node.name!r} must have exactly one "
                    f"input, got {list(node.inputs)}"
                )
            src_name = node.inputs[0]
            src_shape = inferred[src_name]
            in_features = int(node.weight.shape[0])
            if src_shape != (in_features,):
                raise ValueError(
                    f"Node {node.name!r} expects input shape ({in_features},) "
                    f"from weight, but input {src_name!r} has inferred shape "
                    f"{src_shape}"
                )
            shape = (int(node.weight.shape[1]),)
        elif node.op in (ADD, SUB, MUL):
            if len(node.inputs) != 2:
                raise ValueError(
                    f"{node.op} node {node.name!r} must have exactly two "
                    f"inputs, got {list(node.inputs)}"
                )
            a_name, b_name = node.inputs[0], node.inputs[1]
            a_shape = inferred[a_name]
            b_shape = inferred[b_name]
            if a_shape != b_shape:
                raise ValueError(
                    f"{node.op} node {node.name!r} inputs must have equal "
                    f"shapes, got {a_name!r} shape {a_shape} and {b_name!r} "
                    f"shape {b_shape}"
                )
            shape = a_shape
        elif node.op == MATMUL:
            if len(node.inputs) != 2:
                raise ValueError(
                    f"MatMul node {node.name!r} must have exactly two "
                    f"inputs, got {list(node.inputs)}"
                )
            a_name, b_name = node.inputs[0], node.inputs[1]
            a_shape = inferred[a_name]
            b_shape = inferred[b_name]
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise ValueError(
                    f"MatMul node {node.name!r} inputs must both be 2D, got "
                    f"{a_name!r} shape {a_shape} and {b_name!r} shape {b_shape}"
                )
            m, k = a_shape
            k2, n = b_shape
            if k != k2:
                raise ValueError(
                    f"MatMul node {node.name!r} inner dimensions must match, "
                    f"got {a_name!r} shape {a_shape} and {b_name!r} shape {b_shape}"
                )
            shape = (m, n)
        elif node.op == SOFTMAX:
            if len(node.inputs) != 1:
                raise ValueError(
                    f"Softmax node {node.name!r} must have exactly one "
                    f"input, got {list(node.inputs)}"
                )
            src_name = node.inputs[0]
            src_shape = inferred[src_name]
            if len(src_shape) != 1:
                raise ValueError(
                    f"Softmax node {node.name!r} input {src_name!r} must be "
                    f"1D, got shape {src_shape}"
                )
            shape = src_shape
        else:
            # ReLU/Output/Tanh/Sigmoid etc - shape just passes through
            if len(node.inputs) != 1:
                raise ValueError(
                    f"Node {node.name!r} ({node.op}) must have exactly one "
                    f"input for shape inference, got {list(node.inputs)}"
                )
            src_name = node.inputs[0]
            shape = inferred[src_name]

        if node.shape != shape:
            raise ValueError(
                f"Node {node.name!r} declared shape {node.shape} does not "
                f"match inferred shape {shape}"
            )
        inferred[node.name] = shape

    return inferred
