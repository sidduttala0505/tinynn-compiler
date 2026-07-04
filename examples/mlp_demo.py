"""End-to-end TinyNN demo: build -> interpret -> optimize -> compile -> verify.

Builds a small 2-layer MLP with a deliberately dead branch, runs it through
the NumPy interpreter (the correctness oracle), optimizes it with the default
pass pipeline (dead code elimination + Linear/ReLU fusion), compiles the
optimized graph to a native binary via the C++ backend, and checks that every
path agrees. Also writes a JSON snapshot and Graphviz DOT files.

Run from the repo root:

    ./venv/bin/python examples/mlp_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinynn import (  # noqa: E402
    GraphBuilder,
    compile_graph,
    default_pipeline,
    run,
    save_dot,
    save_json,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def build_graph():
    """A 4 -> 8 -> 3 MLP with an extra dead Linear branch off the input."""
    rng = np.random.default_rng(0)
    b = GraphBuilder()
    x = b.input("x", shape=(4,))
    h = b.linear("linear_0", x, weight=rng.standard_normal((4, 8)),
                 bias=rng.standard_normal(8))
    h = b.relu("relu_0", h)
    y = b.linear("linear_1", h, weight=rng.standard_normal((8, 3)),
                 bias=rng.standard_normal(3))
    # Dead branch: computed by nothing downstream; DCE should remove it.
    b.linear("dead_linear", x, weight=rng.standard_normal((4, 5)),
             bias=rng.standard_normal(5))
    b.output("y", y)
    return b.build()


def main() -> None:
    graph = build_graph()
    x = np.array([0.5, -1.0, 2.0, 0.25])

    print("Original graph:")
    print(graph.summary())

    interp_out = run(graph, x)
    print("\nInterpreter output:", interp_out)

    optimized = default_pipeline().run(graph)
    print("\nOptimized graph (after DCE + Linear/ReLU fusion):")
    print(optimized.summary())

    opt_interp_out = run(optimized, x)
    assert np.allclose(interp_out, opt_interp_out), "optimization changed output!"
    print("\nOptimized interpreter output matches original: OK")

    model = compile_graph(optimized, OUT_DIR)
    cpp_out = model.run(x)
    assert np.allclose(interp_out, cpp_out), "compiled output mismatch!"
    print(f"Compiled binary ({model.binary_path.name}) output matches: OK")

    save_json(graph, OUT_DIR / "mlp.json")
    save_dot(graph, OUT_DIR / "mlp.dot")
    save_dot(optimized, OUT_DIR / "mlp_optimized.dot")
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  mlp.json           - graph snapshot (load with tinynn.load_json)")
    print("  mlp.dot            - original graph (render: dot -Tpng mlp.dot)")
    print("  mlp_optimized.dot  - optimized graph")
    print(f"  {model.cpp_path.name}    - generated C++")


if __name__ == "__main__":
    main()
