"""Seeded fuzz suite over *general* graph shapes for the TinyNN compiler.

Unlike ``tests/test_fuzz.py`` (which fuzzes simple linear-chain MLPs),
:func:`random_graph` here builds graphs with branching, diamonds,
multi-consumer nodes, dead branches, 2D tensors, and the full Tier 2 op set
(Add/Sub/Mul/MatMul/Softmax/Tanh/Sigmoid) alongside Linear/ReLU. Operands for
each new node are sampled from *all* previously-created nodes (not just the
most recent one), which naturally produces:

    * multi-consumer nodes (the same node feeding two or more later nodes),
    * diamonds (``x -> f``, ``x -> g``, ``f`` and ``g`` both feeding a later
      Add/Sub/Mul),
    * dead branches (nodes never selected as anyone's operand and not the
      graph output -- left for dead code elimination to prune).

Four things are compared against each other across the suite:

    1. the original graph through the reference interpreter,
    2. the ``default_pipeline()``-optimized graph through the interpreter,
    3. the original graph compiled to C++ and run,
    4. the optimized graph compiled to C++ and run (checked against the
       *original* graph's interpreter output -- the strongest end-to-end
       property).

All randomness is drawn from an explicitly seeded ``np.random.Generator``
(``np.random.default_rng(seed)``); there is zero unseeded randomness anywhere
in this module, so any failure is reproducible from the seed alone. Tests are
parametrized over explicit seed lists so a failing seed shows up directly in
the pytest test id (e.g. ``test_fuzz_graph_passes_vs_interpreter[7]``).
"""

from __future__ import annotations

import shutil
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tinynn.codegen import compile_graph
from tinynn.graph import Graph, GraphBuilder
from tinynn.interpreter import run
from tinynn.ops import ADD, FUSED_LINEAR_RELU, MATMUL, MUL, SOFTMAX, SUB
from tinynn.passes import default_pipeline

_NO_GXX = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not found on PATH")

# --------------------------------------------------------------------------- #
# Tunable generation probabilities (kept as module constants so they can be
# adjusted in one place; all draws still come from the caller-supplied rng).
# --------------------------------------------------------------------------- #
_TWO_D_INPUT_PROB = 0.35
_MIN_EXTRA_NODES = 3
_MAX_EXTRA_NODES = 10
_OPS_POOL = (
    "linear",
    "relu",
    "relu",  # weighted up slightly: raises the odds a ReLU lands directly on
    "tanh",  # the immediately-preceding Linear node, so Linear+ReLU fusion
    "sigmoid",  # actually fires somewhere in the population (see
    "softmax",  # test_fuzz_population_properties).
    "add",
    "sub",
    "mul",
    "matmul",
)


# --------------------------------------------------------------------------- #
# Random graph generator
# --------------------------------------------------------------------------- #
def random_graph(rng: np.random.Generator) -> Tuple[Graph, Dict[str, np.ndarray]]:
    """Build a random branching graph exercising the full Tier 2 op set.

    Returns ``(graph, inputs)`` where ``inputs`` maps every Input node's name
    to a seeded standard-normal array of the right shape.
    """
    builder = GraphBuilder()
    shape_of: Dict[str, Tuple[int, ...]] = {}
    all_names: List[str] = []
    inputs: Dict[str, np.ndarray] = {}

    # -- 1 or 2 Input nodes, each 1D or (with some probability) 2D. -------- #
    n_inputs = int(rng.integers(1, 3))  # 1 or 2
    for i in range(n_inputs):
        if rng.random() < _TWO_D_INPUT_PROB:
            shape = tuple(int(d) for d in rng.integers(2, 5, size=2))  # (2..4, 2..4)
        else:
            shape = (int(rng.integers(2, 7)),)  # width 2..6
        name = f"input_{i}"
        builder.input(name, shape)
        shape_of[name] = shape
        all_names.append(name)
        inputs[name] = rng.standard_normal(shape)

    def pick_any() -> str:
        return all_names[int(rng.integers(len(all_names)))]

    # -- Iteratively add 3..10 more nodes. ---------------------------------- #
    n_extra = int(rng.integers(_MIN_EXTRA_NODES, _MAX_EXTRA_NODES + 1))
    last_non_input = None

    for step in range(n_extra):
        op = _OPS_POOL[int(rng.integers(len(_OPS_POOL)))]
        name = f"n{step}_{op}"
        shape = None

        if op == "linear":
            ones_1d = [n for n in all_names if len(shape_of[n]) == 1]
            if ones_1d:
                src = ones_1d[int(rng.integers(len(ones_1d)))]
                in_w = shape_of[src][0]
                out_w = int(rng.integers(1, 7))
                weight = rng.standard_normal((in_w, out_w))
                bias = rng.standard_normal(out_w)
                builder.linear(name, src, weight, bias)
                shape = (out_w,)
        elif op == "relu":
            src = pick_any()
            builder.relu(name, src)
            shape = shape_of[src]
        elif op == "tanh":
            src = pick_any()
            builder.tanh(name, src)
            shape = shape_of[src]
        elif op == "sigmoid":
            src = pick_any()
            builder.sigmoid(name, src)
            shape = shape_of[src]
        elif op == "softmax":
            ones_1d = [n for n in all_names if len(shape_of[n]) == 1]
            if ones_1d:
                src = ones_1d[int(rng.integers(len(ones_1d)))]
                builder.softmax(name, src)
                shape = shape_of[src]
        elif op in ("add", "sub", "mul"):
            # Any shape that already occurs works: pick a shape present among
            # existing nodes, then sample two nodes with that shape (with
            # replacement -- sampling the same node twice is legal).
            seen_shapes: List[Tuple[int, ...]] = []
            for n in all_names:
                if shape_of[n] not in seen_shapes:
                    seen_shapes.append(shape_of[n])
            shape_key = seen_shapes[int(rng.integers(len(seen_shapes)))]
            candidates = [n for n in all_names if shape_of[n] == shape_key]
            a = candidates[int(rng.integers(len(candidates)))]
            b = candidates[int(rng.integers(len(candidates)))]
            if op == "add":
                builder.add(name, a, b)
            elif op == "sub":
                builder.sub(name, a, b)
            else:
                builder.mul(name, a, b)
            shape = shape_key
        elif op == "matmul":
            twos = [n for n in all_names if len(shape_of[n]) == 2]
            pairs = [
                (n1, n2)
                for n1 in twos
                for n2 in twos
                if shape_of[n1][1] == shape_of[n2][0]
            ]
            if pairs:
                n1, n2 = pairs[int(rng.integers(len(pairs)))]
                builder.matmul(name, n1, n2)
                shape = (shape_of[n1][0], shape_of[n2][1])

        if shape is None:
            # Requirements couldn't be satisfied (e.g. no 1D node yet, or no
            # compatible MatMul pair yet) -- fall back to a unary op, which
            # is always satisfiable since at least one node already exists.
            unary_op = ("relu", "tanh", "sigmoid")[int(rng.integers(3))]
            src = pick_any()
            name = f"n{step}_{unary_op}_fallback"
            if unary_op == "relu":
                builder.relu(name, src)
            elif unary_op == "tanh":
                builder.tanh(name, src)
            else:
                builder.sigmoid(name, src)
            shape = shape_of[src]

        shape_of[name] = shape
        all_names.append(name)
        last_non_input = name

    graph = builder.build(output_node=last_non_input)
    return graph, inputs


# --------------------------------------------------------------------------- #
# Graph-population inspection helpers (used by the property test).
# --------------------------------------------------------------------------- #
def _has_multi_consumer(graph: Graph) -> bool:
    consumer_count: Dict[str, int] = {n.name: 0 for n in graph.nodes}
    for n in graph.nodes:
        for inp in n.inputs:
            consumer_count[inp] += 1
    consumer_count[graph.output_node] += 1
    return any(c >= 2 for c in consumer_count.values())


def _has_dead_node(graph: Graph) -> bool:
    node_by_name = graph.node_map()
    live: set = set()
    stack = [graph.output_node]
    while stack:
        name = stack.pop()
        if name in live:
            continue
        live.add(name)
        stack.extend(node_by_name[name].inputs)
    return any(n.name not in live for n in graph.nodes)


def _has_2d_node(graph: Graph) -> bool:
    return any(len(n.shape) == 2 for n in graph.nodes)


def _fusion_fires(graph: Graph) -> bool:
    optimized = default_pipeline().run(graph)
    return any(n.op == FUSED_LINEAR_RELU for n in optimized.nodes)


def _ops_present(graph: Graph) -> set:
    return {n.op for n in graph.nodes}


# --------------------------------------------------------------------------- #
# Seed lists (explicit so failing seeds show up in the test id).
# --------------------------------------------------------------------------- #
PASSES_SEEDS = list(range(1000, 1025))  # 25 seeds
POPULATION_SEEDS = list(range(2000, 2040))  # 40 seeds
CODEGEN_SEEDS = list(range(3000, 3005))  # 5 seeds
OPT_CODEGEN_SEEDS = list(range(4000, 4004))  # 4 seeds


# --------------------------------------------------------------------------- #
# 1. Optimization passes vs. interpreter (on the original graph's inputs).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", PASSES_SEEDS)
def test_fuzz_graph_passes_vs_interpreter(seed):
    rng = np.random.default_rng(seed)
    graph, inputs = random_graph(rng)

    y0 = run(graph, inputs)

    optimized = default_pipeline().run(graph)
    assert isinstance(optimized, Graph)

    y1 = run(optimized, inputs)

    np.testing.assert_allclose(
        y0, y1, rtol=1e-12, atol=1e-12,
        err_msg=f"seed={seed}\n{graph.summary()}",
    )


# --------------------------------------------------------------------------- #
# 2. Population properties: make sure the generator actually exercises what
#    we claim it does, across a fixed scan of seeds.
# --------------------------------------------------------------------------- #
def test_fuzz_population_properties():
    seen_multi_consumer = []
    seen_dead_node = []
    seen_2d = []
    seen_fusion = []
    ops_seen: set = set()
    ops_seed_hit: Dict[str, int] = {}

    for seed in POPULATION_SEEDS:
        rng = np.random.default_rng(seed)
        graph, _ = random_graph(rng)

        if _has_multi_consumer(graph):
            seen_multi_consumer.append(seed)
        if _has_dead_node(graph):
            seen_dead_node.append(seed)
        if _has_2d_node(graph):
            seen_2d.append(seed)
        if _fusion_fires(graph):
            seen_fusion.append(seed)

        graph_ops = _ops_present(graph)
        for op in (ADD, SUB, MUL, MATMUL, SOFTMAX):
            if op in graph_ops and op not in ops_seed_hit:
                ops_seed_hit[op] = seed
        ops_seen |= graph_ops

    assert seen_multi_consumer, "expected at least one graph with a multi-consumer node"
    assert seen_dead_node, "expected at least one graph with a dead (unreachable) node"
    assert seen_2d, "expected at least one graph with a 2D node"
    assert seen_fusion, "expected default_pipeline() to fuse Linear+ReLU in at least one graph"

    for op in (ADD, SUB, MUL, MATMUL, SOFTMAX):
        assert op in ops_seen, f"expected {op} to appear somewhere across the population"

    # Informational only (not asserted): where each property first fired.
    # multi-consumer: seen_multi_consumer[:5]
    # dead node:      seen_dead_node[:5]
    # 2D node:        seen_2d[:5]
    # fusion fires:   seen_fusion[:5]
    # per-op first hit: ops_seed_hit


# --------------------------------------------------------------------------- #
# 3. C++ codegen vs. interpreter (original graph).
# --------------------------------------------------------------------------- #
@_NO_GXX
@pytest.mark.parametrize("seed", CODEGEN_SEEDS)
def test_fuzz_compiled_vs_interpreter(seed, tmp_path):
    rng = np.random.default_rng(seed)
    graph, inputs = random_graph(rng)

    expected = run(graph, inputs)

    model = compile_graph(graph, tmp_path)
    actual = model.run(inputs)

    np.testing.assert_allclose(
        actual, expected, rtol=1e-9, atol=1e-9,
        err_msg=f"seed={seed}\n{graph.summary()}",
    )


# --------------------------------------------------------------------------- #
# 4. Optimized-graph codegen vs. original-graph interpreter (strongest,
#    end-to-end property: optimize + compile must still match the
#    un-optimized reference semantics).
# --------------------------------------------------------------------------- #
@_NO_GXX
@pytest.mark.parametrize("seed", OPT_CODEGEN_SEEDS)
def test_fuzz_optimized_compiled_vs_original_interpreter(seed, tmp_path):
    rng = np.random.default_rng(seed)
    graph, inputs = random_graph(rng)

    expected = run(graph, inputs)

    optimized = default_pipeline().run(graph)
    model = compile_graph(optimized, tmp_path)
    actual = model.run(inputs)

    np.testing.assert_allclose(
        actual, expected, rtol=1e-9, atol=1e-9,
        err_msg=f"seed={seed}\n{graph.summary()}",
    )
