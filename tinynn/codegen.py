"""C++ code generator + compiler driver for the TinyNN Graph IR.

This module turns a validated :class:`tinynn.graph.Graph` into a small,
readable, standalone C++ program and compiles it with a system C++ compiler
(``g++`` by default). The generated program:

    * reads the single graph ``Input`` node's values as whitespace-separated
      doubles from ``stdin``,
    * evaluates the graph (``Linear``, ``ReLU``, ``Output``) using plain
      ``std::vector<double>`` values and simple loops,
    * prints the values of ``graph.output_node`` to ``stdout`` as
      whitespace-separated doubles followed by a newline.

Tier 0 scope: exactly one ``Input`` node, all node shapes are 1D. This keeps
the generated code (and this module) simple; multi-input / multi-dimensional
support can be layered on later without changing this contract.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .graph import Graph, INPUT, LINEAR, OUTPUT, RELU

__all__ = [
    "sanitize_name",
    "generate_cpp",
    "compile_graph",
    "CompiledModel",
]

# Maximum characters per line for wrapped array-literal source, kept small so
# generated files stay readable in a normal editor/terminal.
_WRAP_WIDTH = 88

_INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")


# --------------------------------------------------------------------------- #
# Name sanitization
# --------------------------------------------------------------------------- #
def sanitize_name(name: str) -> str:
    """Turn an arbitrary node name into a valid, ``v_``-prefixed C++ identifier.

    Any character outside ``[A-Za-z0-9_]`` is replaced with ``_``. This does
    *not* de-duplicate collisions across a whole graph; use
    :func:`_build_identifier_map` for that.
    """
    return "v_" + _INVALID_CHARS_RE.sub("_", name)


def _build_identifier_map(graph: Graph) -> Dict[str, str]:
    """Map each node name to a unique, sanitized C++ identifier.

    Two distinct node names can sanitize to the same identifier (e.g.
    ``"a.b"`` and ``"a-b"`` both become ``v_a_b``). When that happens we
    de-duplicate by appending ``_2``, ``_3``, ... to later collisions, in
    graph node order.
    """
    used: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for node in graph.nodes:
        base = sanitize_name(node.name)
        if base not in used:
            used[base] = 1
            mapping[node.name] = base
        else:
            used[base] += 1
            candidate = f"{base}_{used[base]}"
            # Guard against a pathological case where the bumped candidate
            # itself collides with something already assigned.
            while candidate in used:
                used[base] += 1
                candidate = f"{base}_{used[base]}"
            used[candidate] = 1
            mapping[node.name] = candidate
    return mapping


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
_SUPPORTED = {INPUT, LINEAR, RELU, OUTPUT}


def _validate_for_codegen(graph: Graph) -> None:
    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    if len(input_nodes) != 1:
        raise ValueError(
            "codegen currently supports graphs with exactly one Input node, "
            f"got {len(input_nodes)}: {[n.name for n in input_nodes]}"
        )

    defined: set = set()
    for node in graph.nodes:
        if node.op not in _SUPPORTED:
            raise ValueError(
                f"codegen does not support op {node.op!r} on node {node.name!r}; "
                f"supported ops are {sorted(_SUPPORTED)}"
            )
        if len(node.shape) != 1:
            raise ValueError(
                f"codegen only supports 1D node shapes (Tier 0), but node "
                f"{node.name!r} has shape {node.shape!r}"
            )
        for inp in node.inputs:
            if inp not in defined:
                raise ValueError(
                    f"Node {node.name!r} references input {inp!r} which has "
                    "not been computed yet (graph is not topologically ordered)"
                )
        defined.add(node.name)


# --------------------------------------------------------------------------- #
# C++ literal formatting
# --------------------------------------------------------------------------- #
def _format_double_array(values: np.ndarray) -> str:
    """Format a 1D array as a wrapped, comma-separated C++ initializer body."""
    tokens = [format(float(v), ".17g") for v in values.ravel(order="C")]
    if not tokens:
        return ""
    lines: List[str] = []
    current = "    "
    for i, tok in enumerate(tokens):
        piece = tok + ("," if i != len(tokens) - 1 else "")
        if current.strip() and len(current) + 1 + len(piece) > _WRAP_WIDTH:
            lines.append(current)
            current = "    " + piece
        else:
            current = (current + " " + piece) if current.strip() else (current + piece)
    lines.append(current)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# C++ source generation
# --------------------------------------------------------------------------- #
def generate_cpp(graph: Graph) -> str:
    """Generate readable, standalone C++ source implementing ``graph``.

    The generated program reads the graph's single Input node from stdin
    (whitespace-separated doubles) and writes the output node's values to
    stdout (space-separated doubles, newline-terminated).
    """
    _validate_for_codegen(graph)
    ids = _build_identifier_map(graph)

    lines: List[str] = []
    lines.append("// Auto-generated by tinynn.codegen — do not edit by hand.")
    lines.append("//")
    lines.append("// Graph nodes:")
    for node in graph.nodes:
        lines.append(
            f"//   {node.name}: {node.op} inputs={list(node.inputs)} shape={node.shape}"
        )
    lines.append(f"// output_node: {graph.output_node}")
    lines.append("")
    lines.append("#include <iostream>")
    lines.append("#include <vector>")
    lines.append("#include <cmath>")
    lines.append("#include <iomanip>")
    lines.append("#include <algorithm>")
    lines.append("")

    # Global static weight/bias arrays for Linear nodes.
    for node in graph.nodes:
        if node.op != LINEAR:
            continue
        ident = ids[node.name]
        in_features, out_features = node.weight.shape
        w_body = _format_double_array(node.weight)
        b_body = _format_double_array(node.bias)
        lines.append(f"// Linear node {node.name!r}: in_features={in_features}, out_features={out_features}")
        lines.append(f"static const std::vector<double> {ident}_w = {{")
        lines.append(w_body)
        lines.append("};")
        lines.append(f"static const std::vector<double> {ident}_b = {{")
        lines.append(b_body)
        lines.append("};")
        lines.append("")

    lines.append("int main() {")
    lines.append("    std::cout << std::setprecision(17);")
    lines.append("")

    for node in graph.nodes:
        ident = ids[node.name]
        n = node.shape[0]

        if node.op == INPUT:
            lines.append(f"    // Input node {node.name!r}")
            lines.append(f"    std::vector<double> {ident}({n});")
            lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            lines.append(f"        if (!(std::cin >> {ident}[i])) {{")
            lines.append(
                '            std::cerr << "Error: failed to read input value "'
                f' << i << " for node \\"{node.name}\\" (expected {n} doubles)"'
                ' << std::endl;'
            )
            lines.append("            return 1;")
            lines.append("        }")
            lines.append("    }")
            lines.append("")

        elif node.op == LINEAR:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            lines.append(f"    // Linear node {node.name!r}")
            lines.append(f"    std::vector<double> {ident}({out_features});")
            lines.append(f"    for (int j = 0; j < {out_features}; ++j) {{")
            lines.append(f"        double acc = {ident}_b[j];")
            lines.append(f"        for (int i = 0; i < {in_features}; ++i) {{")
            lines.append(f"            acc += {src_ident}[i] * {ident}_w[i * {out_features} + j];")
            lines.append("        }")
            lines.append(f"        {ident}[j] = acc;")
            lines.append("    }")
            lines.append("")

        elif node.op == RELU:
            src_ident = ids[node.inputs[0]]
            lines.append(f"    // ReLU node {node.name!r}")
            lines.append(f"    std::vector<double> {ident}({n});")
            lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            lines.append(f"        {ident}[i] = std::max(0.0, {src_ident}[i]);")
            lines.append("    }")
            lines.append("")

        elif node.op == OUTPUT:
            src_ident = ids[node.inputs[0]]
            lines.append(f"    // Output node {node.name!r}")
            lines.append(f"    std::vector<double> {ident} = {src_ident};")
            lines.append("")

        else:  # pragma: no cover - guarded by _validate_for_codegen
            raise ValueError(f"Unsupported op {node.op!r}")

    out_ident = ids[graph.output_node]
    lines.append(f"    // print output node {graph.output_node!r}")
    lines.append(f"    for (size_t i = 0; i < {out_ident}.size(); ++i) {{")
    lines.append("        if (i) std::cout << \" \";")
    lines.append(f"        std::cout << {out_ident}[i];")
    lines.append("    }")
    lines.append('    std::cout << "\\n";')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Compiler driver
# --------------------------------------------------------------------------- #
@dataclass
class CompiledModel:
    """A compiled TinyNN graph: the generated C++ source and its binary."""

    cpp_path: Path
    binary_path: Path

    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run the compiled binary on ``input_array`` and return the output.

        ``input_array`` is flattened, converted to float64, and passed to the
        binary via stdin as space-separated, full-precision doubles. The
        binary's stdout is parsed as a 1D float64 array.
        """
        x = np.asarray(input_array, dtype=np.float64).ravel()
        stdin_text = " ".join(format(float(v), ".17g") for v in x) + "\n"

        try:
            proc = subprocess.run(
                [str(self.binary_path)],
                input=stdin_text,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Compiled binary not found at {self.binary_path}; "
                "did compile_graph() succeed?"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Compiled model binary {self.binary_path} exited with code "
                f"{proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        out = proc.stdout.strip()
        if not out:
            raise ValueError(
                f"Compiled model binary {self.binary_path} produced no output "
                f"on stdout (stderr: {proc.stderr!r})"
            )

        try:
            values = [float(tok) for tok in out.split()]
        except ValueError as exc:
            raise ValueError(
                f"Could not parse stdout of {self.binary_path} as floats: {out!r}"
            ) from exc

        if not values:
            raise ValueError(
                f"Compiled model binary {self.binary_path} produced no parsable "
                f"floating point values (stdout: {out!r})"
            )

        return np.array(values, dtype=np.float64)


def compile_graph(
    graph: Graph,
    output_dir,
    binary_name: str = "tinynn_model",
    cpp_name: str = "tinynn_model.cpp",
    compiler: str = "g++",
    extra_flags: Optional[List[str]] = None,
) -> CompiledModel:
    """Generate C++ for ``graph``, compile it, and return a :class:`CompiledModel`.

    Parameters
    ----------
    graph:
        The Graph IR to compile. Must have exactly one Input node and only
        1D node shapes (see :func:`generate_cpp`).
    output_dir:
        Directory to write the generated ``.cpp`` file and the compiled
        binary into. Created (with parents) if it does not exist.
    binary_name / cpp_name:
        File names (not paths) for the compiled binary and generated source.
    compiler:
        Compiler executable to invoke (default ``"g++"``).
    extra_flags:
        Additional flags appended to the compiler invocation, after
        ``-O2 -o <binary>``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = output_dir / cpp_name
    binary_path = output_dir / binary_name

    source = generate_cpp(graph)
    cpp_path.write_text(source)

    cmd = [compiler, str(cpp_path), "-O2", "-o", str(binary_path)]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Compiler {compiler!r} not found, is g++ installed?"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Compilation of {cpp_path} with {compiler!r} failed "
            f"(exit code {proc.returncode}):\n{proc.stderr}"
        )

    return CompiledModel(cpp_path=cpp_path, binary_path=binary_path)
