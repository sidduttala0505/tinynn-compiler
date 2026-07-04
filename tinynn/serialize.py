"""JSON (de)serialization helpers for :class:`tinynn.graph.Graph`.

Thin wrappers around ``Graph.to_dict()`` / ``Graph.from_dict()`` that handle
file I/O, so callers don't need to juggle ``json`` and ``pathlib`` themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .graph import Graph

PathLike = Union[str, "Path"]


def save_json(graph: Graph, path: PathLike) -> Path:
    """Write ``graph.to_dict()`` as JSON to ``path``.

    Creates parent directories if needed. Returns the ``Path`` written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(graph.to_dict(), f, indent=2, sort_keys=False)
    return out_path


def load_json(path: PathLike) -> Graph:
    """Read a JSON file previously written by :func:`save_json` and return a
    :class:`Graph`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (propagated from the underlying open()).
    ValueError
        If the file contents are not valid JSON.
    """
    in_path = Path(path)
    with in_path.open("r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {in_path}: {exc}") from exc
    return Graph.from_dict(data)
