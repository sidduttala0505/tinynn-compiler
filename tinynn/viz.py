"""Turn a Graph into Graphviz DOT text.

Just builds the text - doesn't need the graphviz package or the dot binary.
If you want a picture, pipe the output into `dot` yourself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .graph import Graph

PathLike = Union[str, "Path"]


def _escape(s: str) -> str:
    # backslashes and quotes need escaping inside DOT strings
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(graph: Graph) -> str:
    """Build DOT text for the graph. Weighted nodes are boxes, the rest ellipses,
    and the output node gets a bold outline."""
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
    """Write to_dot output to a file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_dot(graph))
    return out_path
