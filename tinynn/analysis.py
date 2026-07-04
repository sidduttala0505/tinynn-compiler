"""Graph analysis utilities for the TinyNN compiler.

This module provides standalone helpers that operate on :class:`~tinynn.graph.Node`
and :class:`~tinynn.graph.Graph` objects but do not modify existing modules:

    - :func:`topological_sort`: order an arbitrary (possibly shuffled) collection
      of nodes into a valid topological order suitable for ``Graph(...)``.
    - :func:`infer_shapes`: recompute every node's output shape from first
      principles, as a documented, standalone consistency check independent of
      ``Graph.__post_init__``'s validation.

Shape inference rules
----------------------
* ``Input`` nodes: shape is whatever was declared on the node.
* Any node with a 2D ``weight`` (i.e. "Linear-like" -- covers ``Linear`` and any
  future fused op such as ``FusedLinearReLU``): output shape is
  ``(weight.shape[1],)``, and the single input's inferred shape must equal
  ``(weight.shape[0],)``.
* Any other single-input, weight-less op (``ReLU``, ``Output``, or any future
  op of the same shape): output shape equals the input's inferred shape.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .graph import Graph, Node
from .ops import INPUT


def topological_sort(nodes) -> Tuple[Node, ...]:
    """Return ``nodes`` reordered into a valid topological order.

    ``nodes`` may be any iterable of :class:`Node` in arbitrary order. Uses
    Kahn's algorithm. The result is *stable*: among nodes that are ready to be
    emitted at the same time, nodes that appeared earlier in the input are
    emitted first. In particular, if ``nodes`` is already topologically
    sorted, the returned order is identical to the input order.

    Raises
    ------
    ValueError
        If two nodes share a name, if a node references an input name that
        does not exist among ``nodes``, or if the nodes contain a cycle.
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

    # Ready queue, kept ordered by original index so ties resolve stably.
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
    """Recompute every node's output shape from first principles.

    Returns a ``name -> shape`` dict. Raises ``ValueError`` if an inferred
    shape contradicts the node's declared ``.shape`` or if an input-dimension
    mismatch is found (e.g. a Linear-like node's weight input dimension does
    not match its input node's shape).
    """
    inferred: Dict[str, Tuple[int, ...]] = {}

    for node in graph.nodes:
        if node.op == INPUT:
            shape = node.shape
        elif node.weight is not None and node.weight.ndim == 2:
            # Linear-like: Linear, or any future op with a 2D weight
            # (e.g. FusedLinearReLU).
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
        else:
            # Weight-less single-input op (ReLU, Output, or any future op of
            # the same shape-preserving kind).
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
