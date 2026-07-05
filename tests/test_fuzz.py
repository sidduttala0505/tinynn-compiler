"""Seeded randomized fuzzing suite for the TinyNN compiler.

The NumPy reference interpreter (:func:`tinynn.interpreter.run`) is the
correctness oracle throughout: for any randomly generated graph, running the
graph through the interpreter must agree with (a) running an *optimized*
version of the same graph through the interpreter, and (b) running the C++
codegen backend's compiled binary.

All randomness is drawn from an explicitly seeded ``np.random.Generator``
(``np.random.default_rng(seed)``); there is zero unseeded randomness anywhere
in this module, so any failure is reproducible from the seed alone. Tests are
parametrized over explicit seed lists so a failing seed shows up directly in
the pytest test id (e.g. ``test_fuzz_codegen_vs_interpreter[seed=23]``).
"""

from __future__ import annotations

import shutil
from typing import Tuple

import numpy as np
import pytest

from tinynn.codegen import compile_graph
from tinynn.graph import Graph, GraphBuilder
from tinynn.interpreter import run
from tinynn.ops import FUSED_LINEAR_RELU
from tinynn.passes import dead_code_elimination, default_pipeline, fuse_linear_relu

_NO_GXX = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not found on PATH")


# --------------------------------------------------------------------------- #
# Random graph generator
# --------------------------------------------------------------------------- #
def random_mlp(
    rng: np.random.Generator,
    *,
    allow_dead_branch: bool = False,
    min_layers: int = 1,
    max_layers: int = 5,
    min_width: int = 1,
    max_width: int = 8,
    relu_prob: float = 0.5,
    dead_branch_prob: float = 0.4,
    explicit_output_prob: float = 0.5,
) -> Tuple[Graph, np.ndarray]:
    """Build a random-depth, random-width MLP: ``Input -> [Linear -> maybe ReLU]*``.

    All randomness (layer count, widths, weights, biases, whether each Linear
    is followed by a ReLU, whether a dead branch is inserted, whether an
    explicit ``Output`` node is appended, and the input vector) is drawn from
    ``rng``. Always exactly one ``Input`` node (codegen requires this).

    Returns ``(graph, x)`` where ``x`` is a random input vector matching the
    Input node's shape.
    """
    n_layers = int(rng.integers(min_layers, max_layers + 1))
    in_features = int(rng.integers(min_width, max_width + 1))

    b = GraphBuilder()
    x_name = b.input("x", (in_features,))

    cur = x_name
    cur_width = in_features
    for i in range(n_layers):
        out_width = int(rng.integers(min_width, max_width + 1))
        weight = rng.standard_normal((cur_width, out_width))
        bias = rng.standard_normal(out_width)
        lin_name = b.linear(f"linear_{i}", cur, weight, bias)
        cur = lin_name
        cur_width = out_width
        if rng.random() < relu_prob:
            cur = b.relu(f"relu_{i}", cur)

    last_comp = cur

    if allow_dead_branch and rng.random() < dead_branch_prob:
        dead_out_width = int(rng.integers(min_width, max_width + 1))
        dead_weight = rng.standard_normal((in_features, dead_out_width))
        dead_bias = rng.standard_normal(dead_out_width)
        # A Linear fed by the Input node whose output nothing consumes -- pure
        # dead code for DCE to remove.
        b.linear("dead_linear", x_name, dead_weight, dead_bias)

    if rng.random() < explicit_output_prob:
        out_name = b.output("output", last_comp)
        graph = b.build(output_node=out_name)
    else:
        graph = b.build(output_node=last_comp)

    x = rng.standard_normal(in_features)
    return graph, x


# --------------------------------------------------------------------------- #
# Seed lists (explicit so failing seeds show up in the test id)
# --------------------------------------------------------------------------- #
PASSES_SEEDS = list(range(20))
CODEGEN_SEEDS = list(range(20, 26))
OPT_CODEGEN_SEEDS = list(range(100, 104))


# --------------------------------------------------------------------------- #
# 1. Optimization passes vs. interpreter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", PASSES_SEEDS)
def test_fuzz_passes_vs_interpreter(seed):
    rng = np.random.default_rng(seed)
    graph, x = random_mlp(rng, allow_dead_branch=True)

    y0 = run(graph, x)

    optimized = default_pipeline().run(graph)
    # Construction of `optimized` already proves it is a valid Graph (Graph's
    # __post_init__ validates eagerly); re-running the interpreter on it
    # further proves it is semantically equivalent to the original.
    assert isinstance(optimized, Graph)

    y1 = run(optimized, x)

    np.testing.assert_allclose(y0, y1, rtol=1e-12, atol=1e-12)


def test_fuzz_fusion_fires_in_population():
    """Sanity check: Linear+ReLU fusion should actually fire for some fuzzed
    graphs, not just be a no-op. Scans a fixed, self-contained set of seeds
    (independent of any other test) until it finds one where fusion fires.
    """
    fired_seeds = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        graph, _ = random_mlp(rng, allow_dead_branch=True)
        optimized = fuse_linear_relu(dead_code_elimination(graph))
        if any(n.op == FUSED_LINEAR_RELU for n in optimized.nodes):
            fired_seeds.append(seed)

    assert fired_seeds, "expected Linear+ReLU fusion to fire for at least one fuzzed graph in 50 seeds"


# --------------------------------------------------------------------------- #
# 2. C++ codegen vs. interpreter (original graph)
# --------------------------------------------------------------------------- #
@_NO_GXX
@pytest.mark.parametrize("seed", CODEGEN_SEEDS)
def test_fuzz_codegen_vs_interpreter(seed, tmp_path):
    rng = np.random.default_rng(seed)
    # No dead branches here: keep the graphs sent to codegen simple. (Codegen
    # itself tolerates dead nodes fine, but the passes fuzz test above already
    # exercises that path.)
    graph, x = random_mlp(rng, allow_dead_branch=False)

    model = compile_graph(graph, tmp_path)
    actual = model.run(x)
    expected = run(graph, x)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Optimized-graph codegen vs. original-graph interpreter (strongest,
#    end-to-end property: optimize + compile must still match the un-optimized
#    reference semantics).
# --------------------------------------------------------------------------- #
@_NO_GXX
@pytest.mark.parametrize("seed", OPT_CODEGEN_SEEDS)
def test_fuzz_optimized_codegen_vs_original_interpreter(seed, tmp_path):
    rng = np.random.default_rng(seed)
    graph, x = random_mlp(rng, allow_dead_branch=False)

    expected = run(graph, x)

    optimized = default_pipeline().run(graph)
    model = compile_graph(optimized, tmp_path)
    actual = model.run(x)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
