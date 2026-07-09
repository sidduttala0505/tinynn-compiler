"""C++ code generator + compiler driver for the TinyNN Graph IR.

This module turns a validated :class:`tinynn.graph.Graph` into a small,
readable, standalone C++ program and compiles it with a system C++ compiler
(``g++`` by default). The generated program:

    * reads each graph ``Input`` node's values, in graph order, as
      whitespace-separated doubles from ``stdin``,
    * evaluates the graph using plain ``std::vector<double>`` values (flat,
      row-major) and simple loops,
    * prints the values of ``graph.output_node`` to ``stdout`` as
      whitespace-separated doubles followed by a newline.

Tier 2 scope: one or more ``Input`` nodes, node shapes are 1D or 2D (rank 0
and rank >=3 are rejected). Tensors are always represented as flat,
row-major ``std::vector<double>`` regardless of logical rank; the Python
side (:class:`CompiledModel`) is responsible for reshaping results back to
their declared shape.

Tier 4 adds two independent features to the generated program and its driver:

    * repeated-execution / timing support (``argv``-controlled ``iters`` /
      ``warmup``, a timed compute loop, and a ``BENCH_NS ... CHECKSUM ...``
      line on stderr) so :meth:`CompiledModel.benchmark` can measure
      steady-state per-iteration wall time without spawning a new process
      per repetition, and
    * :class:`CodegenOptions`, an opt-in set of CPU optimizations (loop
      interchange / cache tiling, ``-O3``/``-march=native``, and
      option-gated OpenMP) applied by :func:`generate_cpp` /
      :func:`compile_graph` when passed explicitly. Passing no options (the
      default, ``None``) reproduces the exact naive/``-O2`` behavior of
      earlier tiers, byte-for-byte on stdout.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .graph import (
    ADD,
    CONST,
    FUSED_LINEAR_RELU,
    Graph,
    INPUT,
    LINEAR,
    MATMUL,
    MUL,
    OUTPUT,
    QUANTIZED_LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    SUB,
    TANH,
)

__all__ = [
    "sanitize_name",
    "generate_cpp",
    "compile_graph",
    "CompiledModel",
    "CodegenOptions",
    "openmp_available",
]

# Maximum characters per line for wrapped array-literal source, kept small so
# generated files stay readable in a normal editor/terminal.
_WRAP_WIDTH = 88

_INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")

_BENCH_RE = re.compile(r"BENCH_NS\s+(-?\d+)\s+CHECKSUM\s+(\S+)")


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
_SUPPORTED = {
    INPUT,
    LINEAR,
    RELU,
    OUTPUT,
    FUSED_LINEAR_RELU,
    ADD,
    SUB,
    MUL,
    MATMUL,
    SOFTMAX,
    TANH,
    SIGMOID,
    CONST,
    QUANTIZED_LINEAR,
}


def _numel(shape: Tuple[int, ...]) -> int:
    """Return the number of elements in ``shape`` (product of dimensions)."""
    n = 1
    for d in shape:
        n *= d
    return n


def _validate_for_codegen(graph: Graph) -> None:
    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    if not input_nodes:
        raise ValueError("codegen requires at least one Input node, got 0")

    defined: set = set()
    for node in graph.nodes:
        if node.op not in _SUPPORTED:
            raise ValueError(
                f"codegen does not support op {node.op!r} on node {node.name!r}; "
                f"supported ops are {sorted(_SUPPORTED)}"
            )
        if len(node.shape) not in (1, 2):
            raise ValueError(
                f"codegen only supports 1D or 2D node shapes, but node "
                f"{node.name!r} has shape {node.shape!r} (rank {len(node.shape)})"
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


def _format_int_array(values: np.ndarray) -> str:
    """Format an integer-valued array as a wrapped C++ initializer body.

    Used for the ``long long`` quantized-weight globals -- values are plain
    integer literals (no decimal point), unlike :func:`_format_double_array`.
    """
    tokens = [str(int(v)) for v in values.ravel(order="C")]
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
# Quantization (Tier 6, QuantizedLinear)
# --------------------------------------------------------------------------- #
def _quantize_weight_for_codegen(weight: np.ndarray):
    """Quantize a Linear-style float64 weight matrix at C++-generation time.

    Mirrors :func:`tinynn.interpreter._quantize_symmetric` exactly (same
    scale formula, same round-half-away-from-zero rounding via
    ``sign(v) * floor(abs(v) + 0.5)`` rather than NumPy's round-half-to-even
    ``np.round``) so the weight quantization baked into the generated C++
    source is bit-for-bit identical to what the interpreter computes for the
    same node at eval time.

    Returns ``(w_q, s_w)``: ``w_q`` an ``int64`` array of the same shape as
    ``weight`` holding integer levels in ``[-127, 127]``, and ``s_w`` the
    float scale.
    """
    maxabs = float(np.max(np.abs(weight)))
    s_w = maxabs / 127.0 if maxabs > 0.0 else 1.0
    scaled = weight / s_w
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    clipped = np.clip(rounded, -127, 127)
    w_q = clipped.astype(np.int64)
    return w_q, s_w


# --------------------------------------------------------------------------- #
# Codegen options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CodegenOptions:
    """Opt-in CPU codegen/compile optimizations.

    ``generate_cpp``/``compile_graph`` treat ``options=None`` as "exactly
    today's naive emission, compiled with ``-O2``" -- passing an explicit
    ``CodegenOptions`` (even ``CodegenOptions()`` with all defaults) switches
    on the option-driven code paths below, though with every field at its
    default it still emits the naive form (only ``opt_level`` changes vs.
    ``None``).

    Attributes
    ----------
    opt_level:
        Compiler optimization flag, e.g. ``"-O2"``/``"-O3"``.
    native:
        When true, add ``-march=native`` to the compile command.
    tile_size:
        When set, enables loop interchange for Linear/FusedLinearReLU and
        cache-blocked tiling (tile size ``tile_size``) for MatMul.
    openmp:
        When true, emit ``#pragma omp parallel for`` above race-free outer
        loops and add ``-fopenmp`` to the compile command. Callers are
        responsible for checking :func:`openmp_available` first -- this
        module does not silently fall back if the compiler rejects
        ``-fopenmp``.
    reuse_buffers:
        When true, run liveness-based, exact-size greedy slot assignment
        over every non-Input, non-Const node's buffer so the generated
        program allocates fewer, reused ``std::vector<double>`` physical
        slots instead of one per node (see :func:`_plan_buffer_reuse`).
        Purely a memory-footprint optimization -- computed values are
        unchanged. Left out of :meth:`fast` so benchmark comparisons across
        that preset stay apples-to-apples.
    """

    opt_level: str = "-O2"
    native: bool = False
    tile_size: Optional[int] = None
    openmp: bool = False
    reuse_buffers: bool = False

    @classmethod
    def fast(cls, openmp: bool = False) -> "CodegenOptions":
        """A reasonable "fast" preset: ``-O3 -march=native``, tiled, optionally OpenMP."""
        return cls(opt_level="-O3", native=True, tile_size=32, openmp=openmp)


_openmp_availability_cache: Dict[str, bool] = {}


def openmp_available(compiler: str = "g++") -> bool:
    """Return whether ``compiler`` accepts ``-fopenmp`` on this machine.

    Trial-compiles a tiny OpenMP program in a temporary directory. The
    result is cached per ``compiler`` for the lifetime of the process (this
    is a real subprocess compile, so callers should not call it in a hot
    loop). On a machine whose ``g++`` is actually Apple clang (no bundled
    OpenMP runtime), this returns ``False``.
    """
    if compiler in _openmp_availability_cache:
        return _openmp_availability_cache[compiler]

    probe_src = "int main() {\n#pragma omp parallel\n{ }\nreturn 0;\n}\n"
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "omp_probe.cpp"
            bin_path = Path(tmpdir) / "omp_probe"
            src_path.write_text(probe_src)
            proc = subprocess.run(
                [compiler, str(src_path), "-fopenmp", "-o", str(bin_path)],
                capture_output=True,
                text=True,
            )
            ok = proc.returncode == 0
    except FileNotFoundError:
        ok = False

    _openmp_availability_cache[compiler] = ok
    return ok


# --------------------------------------------------------------------------- #
# Buffer reuse planning (options.reuse_buffers)
# --------------------------------------------------------------------------- #
def _plan_buffer_reuse(graph: Graph) -> Tuple[Dict[str, int], List[int]]:
    """Liveness-based, exact-size greedy slot assignment for node buffers.

    Every non-Input, non-Const node owns one logical buffer (Input buffers
    are filled once in the prelude but re-read on every repetition of the
    compute loop, so they are never reusable; Const values are emitted as
    global static arrays and are not buffers at all). This function maps
    each logical buffer to a *physical slot* index such that slots are
    reused across non-overlapping lifetimes.

    Liveness: a buffer's last use is the maximum node index among (a) every
    node that consumes it as an input, (b) the producing node itself, and
    (c) +infinity when it belongs to ``graph.output_node`` -- the output
    buffer is read *after* the compute loop (checksum + printing), so it is
    never reusable.

    Assignment: scanning nodes in graph order, a node needing ``numel`` s
    reuses a free slot of numel EXACTLY s if one exists, else opens a new
    slot. A slot becomes free only for nodes whose index is STRICTLY
    GREATER than the last-use index of its previous occupant.

    Invariant (why in-place aliasing can never happen): any buffer consumed
    as an input by node ``i`` has last use >= ``i``, so under the
    strictly-greater-than rule it cannot be free when node ``i``'s output
    slot is chosen -- a node is never handed one of its own input buffers.
    This matters because MatMul, Linear, and Softmax read their inputs
    repeatedly while writing their output.

    Returns ``(assignment, slot_sizes)`` where ``assignment`` maps node
    name -> slot index and ``slot_sizes[k]`` is slot ``k``'s element count.
    """
    node_index = {node.name: i for i, node in enumerate(graph.nodes)}
    planned = [n for n in graph.nodes if n.op not in (INPUT, CONST)]

    last_use: Dict[str, float] = {n.name: float(node_index[n.name]) for n in planned}
    for node in graph.nodes:
        i = float(node_index[node.name])
        for inp in node.inputs:
            if inp in last_use:
                last_use[inp] = max(last_use[inp], i)
    if graph.output_node in last_use:
        last_use[graph.output_node] = float("inf")

    assignment: Dict[str, int] = {}
    slot_sizes: List[int] = []
    slot_last_use: List[float] = []  # last-use index of each slot's current occupant
    for node in planned:
        i = node_index[node.name]
        size = _numel(node.shape)
        chosen: Optional[int] = None
        for k in range(len(slot_sizes)):
            # Exact-size match, and free only STRICTLY AFTER the previous
            # occupant's last use (see invariant in the docstring).
            if slot_sizes[k] == size and i > slot_last_use[k]:
                chosen = k
                break
        if chosen is None:
            chosen = len(slot_sizes)
            slot_sizes.append(size)
            slot_last_use.append(last_use[node.name])
        else:
            slot_last_use[chosen] = last_use[node.name]
        assignment[node.name] = chosen

    return assignment, slot_sizes


def _emit_reuse_decl_lines(
    graph: Graph,
    ids: Dict[str, str],
    assignment: Dict[str, int],
    slot_sizes: List[int],
) -> List[str]:
    """Emit the decl section for a buffer-reuse plan: physical slots first,
    then one reference per logical buffer so all compute code (which refers
    to the per-node identifiers) is unchanged."""
    lines: List[str] = []
    lines.append("    // Physical buffer slots (reuse_buffers=on).")
    lines.append("    // Invariant: a slot is handed to the node at index i only when i is")
    lines.append("    // STRICTLY GREATER than the last-use index of the slot's previous")
    lines.append("    // occupant. Any buffer consumed as an input by node i has last use")
    lines.append("    // >= i, so a node can never receive one of its own input buffers as")
    lines.append("    // its output slot -- in-place aliasing would be incorrect for ops")
    lines.append("    // like MatMul, Linear, and Softmax, which read their inputs while")
    lines.append("    // writing their output.")
    for k, size in enumerate(slot_sizes):
        lines.append(f"    std::vector<double> tinynn_buf_{k}({size});")
    lines.append("")
    for node in graph.nodes:
        if node.name not in assignment:
            continue
        k = assignment[node.name]
        lines.append(
            f"    std::vector<double>& {ids[node.name]} = tinynn_buf_{k};  // {node.name}"
        )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# C++ source generation
# --------------------------------------------------------------------------- #
def generate_cpp(graph: Graph, options: Optional[CodegenOptions] = None) -> str:
    """Generate readable, standalone C++ source implementing ``graph``.

    The generated program reads every graph Input node from stdin, in graph
    order (whitespace-separated doubles, ``_numel(shape)`` values per Input
    node), once. It then computes the graph once (default invocation,
    ``argc == 1``) or ``iters`` times (after ``warmup`` untimed repetitions)
    when invoked as ``<binary> <iters> <warmup>``, writing the *final*
    repetition's output node values to stdout exactly as before. Passing
    explicit ``iters`` > 1 additionally prints one line to stderr:
    ``BENCH_NS <total_nanoseconds> CHECKSUM <checksum>``.

    ``options`` is ``None`` by default, which reproduces the exact naive
    per-op C++ emitted by earlier tiers (only the repeat/timing scaffolding
    around it is new -- with ``iters=1, warmup=0`` -- the default when no
    argv is given -- stdout is byte-identical to before). Passing a
    :class:`CodegenOptions` with ``tile_size`` set switches Linear /
    FusedLinearReLU to loop-interchanged accumulation and MatMul to
    cache-blocked tiling; ``openmp=True`` additionally emits
    ``#pragma omp parallel for`` above race-free outer loops (Linear/Fused
    always use the naive j-outer form under OpenMP, since the interchanged
    accumulation races across the parallelized dimension -- see inline
    comments below).

    ``reuse_buffers=True`` replaces the one-``std::vector``-per-node decl
    section with a set of physical buffer slots plus one reference per node
    (see :func:`_plan_buffer_reuse`); compute statements are unchanged and
    the computed values are identical. With ``reuse_buffers=False`` the
    emission is byte-identical to not having the feature at all.
    """
    _validate_for_codegen(graph)
    ids = _build_identifier_map(graph)
    node_by_name = graph.node_map()

    tile_size = options.tile_size if options is not None else None
    use_openmp = bool(options is not None and options.openmp)
    reuse_buffers = bool(options is not None and options.reuse_buffers)

    buffer_assignment: Dict[str, int] = {}
    buffer_slot_sizes: List[int] = []
    if reuse_buffers:
        buffer_assignment, buffer_slot_sizes = _plan_buffer_reuse(graph)

    lines: List[str] = []
    lines.append("// Auto-generated by tinynn.codegen — do not edit by hand.")
    lines.append("//")
    lines.append("// Graph nodes:")
    for node in graph.nodes:
        lines.append(
            f"//   {node.name}: {node.op} inputs={list(node.inputs)} shape={node.shape}"
        )
    lines.append(f"// output_node: {graph.output_node}")
    if reuse_buffers:
        lines.append(
            f"// buffer plan: {len(buffer_assignment)} logical buffers -> "
            f"{len(buffer_slot_sizes)} physical slots (reuse_buffers=on)"
        )
    lines.append("")
    lines.append("#include <iostream>")
    lines.append("#include <vector>")
    lines.append("#include <cmath>")
    lines.append("#include <iomanip>")
    lines.append("#include <algorithm>")
    lines.append("#include <cstdlib>")
    lines.append("#include <chrono>")
    lines.append("")

    # Global static weight/bias arrays for Linear (and FusedLinearReLU) nodes,
    # plus a global static array for each Const node's value. Const globals
    # are referenced directly by name (no separate decl-section entry, no
    # compute statement) by whichever downstream op consumes them.
    for node in graph.nodes:
        if node.op == CONST:
            ident = ids[node.name]
            lines.append(f"// Const node {node.name!r}: shape={node.shape}")
            lines.append(f"static const std::vector<double> {ident} = {{")
            lines.append(_format_double_array(node.weight))
            lines.append("};")
            lines.append("")
            continue
        if node.op == QUANTIZED_LINEAR:
            ident = ids[node.name]
            in_features, out_features = node.weight.shape
            w_q, s_w = _quantize_weight_for_codegen(node.weight)
            wq_body = _format_int_array(w_q)
            b_body = _format_double_array(node.bias)
            lines.append(
                f"// QuantizedLinear node {node.name!r}: in_features={in_features}, "
                f"out_features={out_features}, s_w={format(s_w, '.17g')}"
            )
            lines.append(f"// Weight quantized at codegen time (deterministic, round-half-away-from-zero).")
            lines.append(f"static const std::vector<long long> {ident}_wq = {{")
            lines.append(wq_body)
            lines.append("};")
            lines.append(f"static const std::vector<double> {ident}_b = {{")
            lines.append(b_body)
            lines.append("};")
            lines.append(f"static const double {ident}_sw = {format(s_w, '.17g')};")
            lines.append("")
            continue
        if node.op not in (LINEAR, FUSED_LINEAR_RELU):
            continue
        ident = ids[node.name]
        in_features, out_features = node.weight.shape
        w_body = _format_double_array(node.weight)
        b_body = _format_double_array(node.bias)
        lines.append(f"// {node.op} node {node.name!r}: in_features={in_features}, out_features={out_features}")
        lines.append(f"static const std::vector<double> {ident}_w = {{")
        lines.append(w_body)
        lines.append("};")
        lines.append(f"static const std::vector<double> {ident}_b = {{")
        lines.append(b_body)
        lines.append("};")
        lines.append("")

    # Three streams, assembled at the end:
    #   prelude_lines  -- Input reads (run exactly once, never repeated).
    #   decl_lines     -- sizing declarations for every other node's buffer,
    #                     emitted once before the compute loop.
    #   compute_lines  -- the actual per-node statements (assignments only,
    #                     no declarations of the sized output buffers), run
    #                     once per repetition inside the `compute` lambda.
    prelude_lines: List[str] = []
    decl_lines: List[str] = []
    compute_lines: List[str] = []
    # QuantizedLinear int8 scratch buffer declarations, tracked separately so
    # they survive wholesale replacement of decl_lines under reuse_buffers=on
    # -- they never participate in the double-precision buffer-reuse pool.
    quantized_scratch_decl_lines: List[str] = []

    for node in graph.nodes:
        ident = ids[node.name]
        n = _numel(node.shape)

        if node.op == INPUT:
            prelude_lines.append(f"    // Input node {node.name!r}")
            prelude_lines.append(f"    std::vector<double> {ident}({n});")
            prelude_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            prelude_lines.append(f"        if (!(std::cin >> {ident}[i])) {{")
            prelude_lines.append(
                '            std::cerr << "Error: failed to read input value "'
                f' << i << " for node \\"{node.name}\\" (expected {n} doubles)"'
                ' << std::endl;'
            )
            prelude_lines.append("            return 1;")
            prelude_lines.append("        }")
            prelude_lines.append("    }")
            prelude_lines.append("")
            continue

        if node.op == CONST:
            # Emitted as a global static array above; no decl-section entry
            # and no compute statement -- downstream ops reference the
            # global identifier directly.
            continue

        if node.op == LINEAR:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            decl_lines.append(f"    // Linear node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({out_features});")
            decl_lines.append("")

            compute_lines.append(f"    // Linear node {node.name!r}")
            compute_lines.extend(
                _emit_linear_like(
                    ident, src_ident, in_features, out_features,
                    tile_size=tile_size, openmp=use_openmp, relu=False,
                )
            )
            compute_lines.append("")

        elif node.op == QUANTIZED_LINEAR:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            decl_lines.append(f"    // QuantizedLinear node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({out_features});")
            scratch_decl = (
                f"    std::vector<long long> {ident}_xq({in_features});"
                "  // int8 scratch, not in the double-slot reuse pool"
            )
            decl_lines.append(scratch_decl)
            decl_lines.append("")
            quantized_scratch_decl_lines.append(scratch_decl)

            compute_lines.append(f"    // QuantizedLinear node {node.name!r}")
            compute_lines.extend(
                _emit_quantized_linear(ident, src_ident, in_features, out_features)
            )
            compute_lines.append("")

        elif node.op == RELU:
            src_ident = ids[node.inputs[0]]
            decl_lines.append(f"    // ReLU node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // ReLU node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"        {ident}[i] = std::max(0.0, {src_ident}[i]);")
            compute_lines.append("    }")
            compute_lines.append("")

        elif node.op == FUSED_LINEAR_RELU:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            decl_lines.append(f"    // FusedLinearReLU node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({out_features});")
            decl_lines.append("")

            compute_lines.append(f"    // FusedLinearReLU node {node.name!r}")
            compute_lines.extend(
                _emit_linear_like(
                    ident, src_ident, in_features, out_features,
                    tile_size=tile_size, openmp=use_openmp, relu=True,
                )
            )
            compute_lines.append("")

        elif node.op == OUTPUT:
            src_ident = ids[node.inputs[0]]
            decl_lines.append(f"    // Output node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // Output node {node.name!r}")
            compute_lines.append(f"    {ident} = {src_ident};")
            compute_lines.append("")

        elif node.op in (ADD, SUB, MUL):
            op_symbol = {ADD: "+", SUB: "-", MUL: "*"}[node.op]
            a_ident = ids[node.inputs[0]]
            b_ident = ids[node.inputs[1]]
            decl_lines.append(f"    // {node.op} node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // {node.op} node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"        {ident}[i] = {a_ident}[i] {op_symbol} {b_ident}[i];")
            compute_lines.append("    }")
            compute_lines.append("")

        elif node.op == MATMUL:
            a_node = node_by_name[node.inputs[0]]
            b_node = node_by_name[node.inputs[1]]
            m, k = a_node.shape
            k2, n_out = b_node.shape
            a_ident = ids[node.inputs[0]]
            b_ident = ids[node.inputs[1]]

            decl_lines.append(
                f"    // MatMul node {node.name!r}: ({m}x{k}) @ ({k}x{n_out})"
            )
            decl_lines.append(f"    std::vector<double> {ident}({m * n_out});")
            decl_lines.append("")

            compute_lines.append(
                f"    // MatMul node {node.name!r}: ({m}x{k}) @ ({k}x{n_out})"
            )
            compute_lines.extend(
                _emit_matmul(
                    ident, a_ident, b_ident, m, k, n_out,
                    tile_size=tile_size, openmp=use_openmp,
                )
            )
            compute_lines.append("")

        elif node.op == SOFTMAX:
            src_ident = ids[node.inputs[0]]
            decl_lines.append(f"    // Softmax node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // Softmax node {node.name!r}")
            compute_lines.append("    {")
            compute_lines.append(f"        double max_val = {src_ident}[0];")
            compute_lines.append(f"        for (int i = 1; i < {n}; ++i) {{")
            compute_lines.append(f"            max_val = std::max(max_val, {src_ident}[i]);")
            compute_lines.append("        }")
            compute_lines.append("        double sum = 0.0;")
            compute_lines.append(f"        for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"            {ident}[i] = std::exp({src_ident}[i] - max_val);")
            compute_lines.append(f"            sum += {ident}[i];")
            compute_lines.append("        }")
            compute_lines.append(f"        for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"            {ident}[i] /= sum;")
            compute_lines.append("        }")
            compute_lines.append("    }")
            compute_lines.append("")

        elif node.op == TANH:
            src_ident = ids[node.inputs[0]]
            decl_lines.append(f"    // Tanh node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // Tanh node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"        {ident}[i] = std::tanh({src_ident}[i]);")
            compute_lines.append("    }")
            compute_lines.append("")

        elif node.op == SIGMOID:
            src_ident = ids[node.inputs[0]]
            decl_lines.append(f"    // Sigmoid node {node.name!r}")
            decl_lines.append(f"    std::vector<double> {ident}({n});")
            decl_lines.append("")

            compute_lines.append(f"    // Sigmoid node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"        {ident}[i] = 1.0 / (1.0 + std::exp(-{src_ident}[i]));")
            compute_lines.append("    }")
            compute_lines.append("")

        else:  # pragma: no cover - guarded by _validate_for_codegen
            raise ValueError(f"Unsupported op {node.op!r}")

    if reuse_buffers:
        # Replace the per-node buffer declarations wholesale with physical
        # slot declarations plus one reference per node; every compute
        # statement above refers only to the per-node identifiers, so the
        # compute stream needs no changes.
        decl_lines = _emit_reuse_decl_lines(
            graph, ids, buffer_assignment, buffer_slot_sizes
        )
        if quantized_scratch_decl_lines:
            decl_lines.append(
                "    // QuantizedLinear int8 scratch buffers (never part of the"
            )
            decl_lines.append("    // double-precision buffer-reuse pool above).")
            decl_lines.extend(quantized_scratch_decl_lines)
            decl_lines.append("")

    out_ident = ids[graph.output_node]

    lines.append("int main(int argc, char** argv) {")
    lines.append("    std::cout << std::setprecision(17);")
    lines.append("    long iters = argc > 1 ? std::atol(argv[1]) : 1;")
    lines.append("    long warmup = argc > 2 ? std::atol(argv[2]) : 0;")
    lines.append("")

    lines.extend(prelude_lines)
    lines.extend(decl_lines)

    lines.append("    double checksum = 0.0;")
    lines.append("")
    lines.append("    auto compute = [&]() {")
    lines.extend("    " + l if l else l for l in compute_lines)
    lines.append("    };")
    lines.append("")
    lines.append("    for (long rep = 0; rep < warmup; ++rep) {")
    lines.append("        compute();")
    lines.append("    }")
    lines.append("")
    lines.append("    auto tinynn_bench_start = std::chrono::steady_clock::now();")
    lines.append("    for (long rep = 0; rep < iters; ++rep) {")
    lines.append("        compute();")
    lines.append(
        f"        checksum += {out_ident}.empty() ? 0.0 : {out_ident}.back();"
    )
    lines.append("    }")
    lines.append("    auto tinynn_bench_end = std::chrono::steady_clock::now();")
    lines.append("")

    lines.append(f"    // print output node {graph.output_node!r}")
    lines.append(f"    for (size_t i = 0; i < {out_ident}.size(); ++i) {{")
    lines.append("        if (i) std::cout << \" \";")
    lines.append(f"        std::cout << {out_ident}[i];")
    lines.append("    }")
    lines.append('    std::cout << "\\n";')
    lines.append("")

    lines.append("    if (iters > 1) {")
    lines.append(
        "        long long tinynn_bench_ns = "
        "std::chrono::duration_cast<std::chrono::nanoseconds>"
        "(tinynn_bench_end - tinynn_bench_start).count();"
    )
    lines.append(
        '        std::cerr << "BENCH_NS " << tinynn_bench_ns << " CHECKSUM "'
        ' << std::setprecision(17) << checksum << std::endl;'
    )
    lines.append("    }")
    lines.append("")
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _emit_linear_like(
    ident: str,
    src_ident: str,
    in_features: int,
    out_features: int,
    tile_size: Optional[int],
    openmp: bool,
    relu: bool,
) -> List[str]:
    """Return compute-only statements for a Linear or FusedLinearReLU node.

    ``openmp`` always forces the naive j-outer form (even when ``tile_size``
    is set): the loop-interchanged accumulation form below writes
    ``ident[j] += ...`` from the *outer* ``i`` loop, so parallelizing over
    ``i`` would race every thread on the same ``ident[j]`` slots. The naive
    form's outer loop is over ``j``, and each ``j`` iteration only ever
    touches its own ``ident[j]`` slot, so it is race-free to parallelize.
    """
    lines: List[str] = []
    assign = "std::max(0.0, acc)" if relu else "acc"

    if tile_size is not None and not openmp:
        # Loop interchange: stream the weight matrix row-major (over i, then
        # j) instead of the naive strided w[i * out_features + j] access
        # pattern from the j-outer form.
        lines.append(f"for (int j = 0; j < {out_features}; ++j) {{")
        lines.append(f"    {ident}[j] = {ident}_b[j];")
        lines.append("}")
        lines.append(f"for (int i = 0; i < {in_features}; ++i) {{")
        lines.append(f"    const double xi = {src_ident}[i];")
        lines.append(f"    for (int j = 0; j < {out_features}; ++j) {{")
        lines.append(f"        {ident}[j] += xi * {ident}_w[i * {out_features} + j];")
        lines.append("    }")
        lines.append("}")
        if relu:
            lines.append(f"for (int j = 0; j < {out_features}; ++j) {{")
            lines.append(f"    {ident}[j] = std::max(0.0, {ident}[j]);")
            lines.append("}")
        return lines

    # Naive j-outer form (default, and forced whenever openmp is requested).
    if openmp:
        lines.append("#pragma omp parallel for")
    lines.append(f"for (int j = 0; j < {out_features}; ++j) {{")
    lines.append(f"    double acc = {ident}_b[j];")
    lines.append(f"    for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"        acc += {src_ident}[i] * {ident}_w[i * {out_features} + j];")
    lines.append("    }")
    lines.append(f"    {ident}[j] = {assign};")
    lines.append("}")
    return lines


def _emit_quantized_linear(
    ident: str,
    src_ident: str,
    in_features: int,
    out_features: int,
) -> List[str]:
    """Return compute-only statements for a ``QuantizedLinear`` node.

    Always plain loops -- ``CodegenOptions.tile_size``/``openmp`` apply only
    to the float Linear/FusedLinearReLU/MatMul paths, never to quantized
    inference (the int8 matmul here is intentionally not tiled/parallelized).

    Mirrors :func:`tinynn.interpreter._eval_quantized_linear` exactly:
    per-call activation quantization (dynamic ``s_x``), a plain int8 x int8
    matmul accumulated into a 64-bit ``acc``, then one float rescale
    (``sx * sw``) plus the float bias add. Rounding is round-half-away-from-
    zero via ``std::lround`` (matches the interpreter's
    ``sign(v) * floor(abs(v) + 0.5)``, NOT NumPy's round-half-to-even).
    """
    lines: List[str] = []
    lines.append("{")
    lines.append(
        "    // Dynamic activation quantization: s_x = max(|x|) / 127, or 1.0 if x is all zero."
    )
    lines.append("    double maxabs = 0.0;")
    lines.append(f"    for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"        maxabs = std::max(maxabs, std::abs({src_ident}[i]));")
    lines.append("    }")
    lines.append("    const double sx = maxabs > 0.0 ? maxabs / 127.0 : 1.0;")
    lines.append(
        "    // Quantize: round-half-away-from-zero (std::lround), clip to int8 range."
    )
    lines.append(f"    for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"        long long q = std::lround({src_ident}[i] / sx);")
    lines.append("        q = std::max<long long>(-127, std::min<long long>(127, q));")
    lines.append(f"        {ident}_xq[i] = q;")
    lines.append("    }")
    lines.append(
        "    // Integer matmul (64-bit accumulation), then a single float rescale + bias add."
    )
    lines.append(f"    for (int j = 0; j < {out_features}; ++j) {{")
    lines.append("        long long acc = 0;")
    lines.append(f"        for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"            acc += {ident}_xq[i] * {ident}_wq[i * {out_features} + j];")
    lines.append("        }")
    lines.append(
        f"        {ident}[j] = static_cast<double>(acc) * (sx * {ident}_sw) + {ident}_b[j];"
    )
    lines.append("    }")
    lines.append("}")
    return lines


def _emit_matmul(
    ident: str,
    a_ident: str,
    b_ident: str,
    m: int,
    k: int,
    n_out: int,
    tile_size: Optional[int],
    openmp: bool,
) -> List[str]:
    """Return compute-only statements for a MatMul node."""
    lines: List[str] = []

    if tile_size is not None:
        t = int(tile_size)
        lines.append(f"for (int i = 0; i < {m}; ++i) {{")
        lines.append(f"    for (int j = 0; j < {n_out}; ++j) {{")
        lines.append(f"        {ident}[i * {n_out} + j] = 0.0;")
        lines.append("    }")
        lines.append("}")
        if openmp:
            lines.append("#pragma omp parallel for")
        lines.append(f"for (int ii = 0; ii < {m}; ii += {t}) {{")
        lines.append(f"    for (int pp = 0; pp < {k}; pp += {t}) {{")
        lines.append(f"        for (int jj = 0; jj < {n_out}; jj += {t}) {{")
        lines.append(f"            for (int i = ii; i < std::min(ii + {t}, {m}); ++i) {{")
        lines.append(f"                for (int p = pp; p < std::min(pp + {t}, {k}); ++p) {{")
        lines.append(f"                    const double a_ip = {a_ident}[i * {k} + p];")
        lines.append(
            f"                    for (int j = jj; j < std::min(jj + {t}, {n_out}); ++j) {{"
        )
        lines.append(
            f"                        {ident}[i * {n_out} + j] += a_ip * {b_ident}[p * {n_out} + j];"
        )
        lines.append("                    }")
        lines.append("                }")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        lines.append("}")
        return lines

    # Naive triple loop (unchanged from earlier tiers).
    if openmp:
        lines.append("#pragma omp parallel for")
    lines.append(f"for (int i = 0; i < {m}; ++i) {{")
    lines.append(f"    for (int j = 0; j < {n_out}; ++j) {{")
    lines.append("        double acc = 0.0;")
    lines.append(f"        for (int p = 0; p < {k}; ++p) {{")
    lines.append(
        f"            acc += {a_ident}[i * {k} + p] * {b_ident}[p * {n_out} + j];"
    )
    lines.append("        }")
    lines.append(f"        {ident}[i * {n_out} + j] = acc;")
    lines.append("    }")
    lines.append("}")
    return lines


# --------------------------------------------------------------------------- #
# Compiler driver
# --------------------------------------------------------------------------- #
@dataclass
class CompiledModel:
    """A compiled TinyNN graph: the generated C++ source and its binary.

    ``input_specs`` records each Input node's name and declared shape, in
    graph order (the same order the generated binary reads them from
    stdin). ``output_shape`` is the declared shape of ``graph.output_node``,
    used to reshape the binary's flat stdout output before returning it.
    """

    cpp_path: Path
    binary_path: Path
    output_shape: Tuple[int, ...]
    input_specs: Tuple[Tuple[str, Tuple[int, ...]], ...]

    def run(self, inputs) -> np.ndarray:
        """Run the compiled binary on ``inputs`` and return the output.

        ``inputs`` may be a ``dict`` mapping every Input node's name to an
        array-like value, or a bare array/array-like -- the latter is only
        accepted when the model has exactly one Input node (this keeps
        single-input, 1D-output models call-compatible with earlier tiers).

        Each input value is converted to float64 and must have exactly as
        many elements as its Input node's declared shape; values are
        flattened row-major and concatenated, in graph order, to build the
        binary's stdin. The binary's stdout is parsed as a flat float64
        array and reshaped to ``output_shape`` before being returned.

        The binary is invoked with no arguments, i.e. ``iters=1, warmup=0``
        -- exactly the same single computation as earlier tiers.
        """
        stdin_text = self._stdin_text(inputs)

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

        return self._parse_stdout(proc.stdout, proc.stderr)

    def benchmark(self, inputs, iters: int = 100, warmup: int = 10) -> float:
        """Run the compiled binary ``iters`` timed repetitions and return seconds/iter.

        Invokes the binary as ``<binary> <iters> <warmup>`` with the same
        stdin as :meth:`run`. The binary runs ``warmup`` untimed
        repetitions of the compute, then ``iters`` timed repetitions,
        printing ``BENCH_NS <total_ns> CHECKSUM <checksum>`` to stderr (in
        addition to its normal stdout output, from the final repetition).
        Returns ``total_ns / iters / 1e9`` -- wall-clock seconds per
        iteration.
        """
        stdin_text = self._stdin_text(inputs)

        try:
            proc = subprocess.run(
                [str(self.binary_path), str(int(iters)), str(int(warmup))],
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
                f"{proc.returncode} while benchmarking (iters={iters}, "
                f"warmup={warmup}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        match = _BENCH_RE.search(proc.stderr)
        if not match:
            raise RuntimeError(
                f"Compiled model binary {self.binary_path} did not print a "
                f"BENCH_NS line to stderr (iters={iters}, warmup={warmup}); "
                f"stderr:\n{proc.stderr!r}"
            )

        total_ns = int(match.group(1))
        return total_ns / int(iters) / 1e9

    def _stdin_text(self, inputs) -> str:
        """Build the flat, whitespace-separated stdin text for ``inputs``."""
        provided = self._resolve_inputs(inputs)

        parts: List[np.ndarray] = []
        for name, shape in self.input_specs:
            arr = np.asarray(provided[name], dtype=np.float64)
            expected = _numel(shape)
            if arr.size != expected:
                raise ValueError(
                    f"Input {name!r} expected {expected} values (shape "
                    f"{shape}), got {arr.size} (shape {arr.shape})"
                )
            parts.append(arr.ravel(order="C"))

        x = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        return " ".join(format(float(v), ".17g") for v in x) + "\n"

    def _parse_stdout(self, stdout: str, stderr: str) -> np.ndarray:
        out = stdout.strip()
        if not out:
            raise ValueError(
                f"Compiled model binary {self.binary_path} produced no output "
                f"on stdout (stderr: {stderr!r})"
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

        expected_out = _numel(self.output_shape)
        if len(values) != expected_out:
            raise ValueError(
                f"Compiled model binary {self.binary_path} produced {len(values)} "
                f"values, expected {expected_out} for output shape "
                f"{self.output_shape} (stdout: {out!r})"
            )

        return np.array(values, dtype=np.float64).reshape(self.output_shape)

    def _resolve_inputs(self, inputs) -> Dict[str, np.ndarray]:
        """Normalize ``inputs`` (dict or bare array) to a ``name -> value`` dict."""
        if isinstance(inputs, dict):
            missing = [name for name, _ in self.input_specs if name not in inputs]
            if missing:
                raise KeyError(
                    f"Missing value(s) for Input node(s): {missing}; this model "
                    f"expects inputs for {[name for name, _ in self.input_specs]}"
                )
            return inputs

        if len(self.input_specs) != 1:
            raise ValueError(
                "A bare array may only be passed as `inputs` when the compiled "
                f"model has exactly one Input node; this model has "
                f"{len(self.input_specs)} ({[name for name, _ in self.input_specs]})"
            )
        return {self.input_specs[0][0]: inputs}


def compile_graph(
    graph: Graph,
    output_dir,
    binary_name: str = "tinynn_model",
    cpp_name: str = "tinynn_model.cpp",
    compiler: str = "g++",
    extra_flags: Optional[List[str]] = None,
    options: Optional[CodegenOptions] = None,
) -> CompiledModel:
    """Generate C++ for ``graph``, compile it, and return a :class:`CompiledModel`.

    Parameters
    ----------
    graph:
        The Graph IR to compile. Must have at least one Input node and only
        1D or 2D node shapes (see :func:`generate_cpp`).
    output_dir:
        Directory to write the generated ``.cpp`` file and the compiled
        binary into. Created (with parents) if it does not exist.
    binary_name / cpp_name:
        File names (not paths) for the compiled binary and generated source.
    compiler:
        Compiler executable to invoke (default ``"g++"``).
    extra_flags:
        Additional flags appended to the compiler invocation, after the
        optimization flags derived from ``options`` (or ``-O2`` when
        ``options`` is ``None``).
    options:
        Optional :class:`CodegenOptions`. ``None`` (the default) reproduces
        the exact naive codegen and ``-O2`` compile flags of earlier tiers.
        When given, ``options.opt_level`` replaces ``-O2``, ``-march=native``
        is added when ``options.native``, and ``-fopenmp`` is added when
        ``options.openmp`` (callers should check :func:`openmp_available`
        first -- this function does not check for them).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = output_dir / cpp_name
    binary_path = output_dir / binary_name

    source = generate_cpp(graph, options)
    cpp_path.write_text(source)

    if options is None:
        cmd = [compiler, str(cpp_path), "-O2", "-o", str(binary_path)]
    else:
        cmd = [compiler, str(cpp_path), options.opt_level]
        if options.native:
            cmd.append("-march=native")
        if options.openmp:
            cmd.append("-fopenmp")
        cmd.extend(["-o", str(binary_path)])

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

    input_specs = tuple((n.name, n.shape) for n in graph.nodes if n.op == INPUT)
    output_shape = graph.get_node(graph.output_node).shape

    return CompiledModel(
        cpp_path=cpp_path,
        binary_path=binary_path,
        output_shape=output_shape,
        input_specs=input_specs,
    )
