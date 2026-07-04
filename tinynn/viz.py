"""Graphviz DOT text generation for :class:`tinynn.graph.Graph`.

Pure text generation: this module does not depend on the ``graphviz`` Python
package and does not invoke the ``dot`` binary. Callers who want a rendered
image can pipe the returned string through their own ``dot`` invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .graph import Graph

PathLike = Union[str, "Path"]


def _escape(s: str) -> str:
    """Escape backslashes and double quotes for safe use inside a DOT string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(graph: Graph) -> str:
    """Render ``graph`` as deterministic Graphviz DOT text.

    Each node becomes one node statement with a label of the form
    ``"name\\nOp\\n(shape,)"``. Nodes with a weight (Linear-like) are drawn as
    boxes, other nodes as ellipses. The graph's output node is visually
    distinguished with a heavier, bold outline. Edges are emitted for each
    node's ``inputs`` (input -> node).
    """
    lines = ["digraph tinynn {"]

    for node in graph.nodes:
        label = "\\n".join(
            [_escape(node.name), _escape(node.op), _escape(str(node.shape))]
        )
        node_shape = "box" if node.weight is not None else "ellipse"
        attrs = [f'label="{label}"', f"shape={node_shape}"]
        if node.name == graph.output_node:
            attrs.append("penwidth=2")
            attrs.append("style=bold")
        lines.append(f'  "{_escape(node.name)}" [{", ".join(attrs)}];')

    for node in graph.nodes:
        for inp in node.inputs:
            lines.append(f'  "{_escape(inp)}" -> "{_escape(node.name)}";')

    lines.append("}")
    return "\n".join(lines) + "\n"


def save_dot(graph: Graph, path: PathLike) -> Path:
    """Write :func:`to_dot` output to ``path``, creating parent dirs as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_dot(graph))
    return out_path
