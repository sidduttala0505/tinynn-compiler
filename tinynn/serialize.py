"""Save/load a Graph as JSON so you don't have to deal with json + pathlib yourself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .graph import Graph

PathLike = Union[str, "Path"]


def save_json(graph: Graph, path: PathLike) -> Path:
    """Dump the graph to a JSON file and return the path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(graph.to_dict(), f, indent=2, sort_keys=False)
    return out_path


def load_json(path: PathLike) -> Graph:
    """Load a graph back from a file written by save_json."""
    in_path = Path(path)
    with in_path.open("r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {in_path}: {exc}") from exc
    return Graph.from_dict(data)
