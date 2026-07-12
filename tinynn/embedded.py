"""Experimental static-memory C99 backend for the TinyNN Graph IR.

This module is an alternate code generator to :mod:`tinynn.codegen`, aimed
at embedded targets where ``malloc``/``free``/``std::vector`` and C++
runtime support are unavailable or undesirable. It turns a validated
:class:`tinynn.graph.Graph` into a small, standalone **C99** program that:

    * reads each graph ``Input`` node's values, in graph order, as
      whitespace-separated doubles from ``stdin``,
    * evaluates the graph using plain, file-scope ``static`` arrays --
      every buffer (weights, biases, quantization scratch, and per-node
      activations) is a fixed-size C array declared once at file scope, and
      ``compute()`` is a single function with no dynamic allocation
      anywhere,
    * prints the values of ``graph.output_node`` to ``stdout`` as
      whitespace-separated doubles followed by a newline.

Scope: this backend is intentionally much narrower than
:mod:`tinynn.codegen`. It supports exactly ``Input``, ``Output``, ``Linear``,
``ReLU``, ``FusedLinearReLU``, ``QuantizedLinear``, and
``QuantizedFusedLinearReLU``; node shapes must be rank 1 or 2. Anything else
raises :class:`ValueError`. It reuses :mod:`tinynn.codegen`'s name
sanitization, literal formatting, and weight-quantization helpers so its
numerics (in particular quantized-linear rounding/clamping) are exactly the
interpreter's, and its ``BENCH_NS``/``CHECKSUM`` protocol on stderr is
identical to :mod:`tinynn.codegen`'s so the same :class:`CompiledModel`
drives both backends.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .codegen import (
    CompiledModel,
    _build_identifier_map,
    _format_double_array,
    _format_int_array,
    _numel,
    _quantize_weight_for_codegen,
    sanitize_name,
)
from .graph import (
    FUSED_LINEAR_RELU,
    Graph,
    INPUT,
    LINEAR,
    OUTPUT,
    QUANTIZED_FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
    RELU,
)

__all__ = [
    "generate_embedded_c",
    "compile_embedded",
]

# Ops this experimental backend knows how to emit. Anything outside this set
# is rejected by _validate_for_embedded with a ValueError naming the op.
_SUPPORTED = {
    INPUT,
    OUTPUT,
    LINEAR,
    RELU,
    FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
    QUANTIZED_FUSED_LINEAR_RELU,
}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_for_embedded(graph: Graph) -> None:
    input_nodes = [n for n in graph.nodes if n.op == INPUT]
    if not input_nodes:
        raise ValueError("embedded codegen requires at least one Input node, got 0")

    defined: set = set()
    for node in graph.nodes:
        if node.op not in _SUPPORTED:
            raise ValueError(
                f"embedded codegen does not support op {node.op!r} on node "
                f"{node.name!r}; the embedded backend is experimental and "
                f"supports only {sorted(_SUPPORTED)}"
            )
        if len(node.shape) not in (1, 2):
            raise ValueError(
                f"embedded codegen only supports 1D or 2D node shapes, but "
                f"node {node.name!r} has shape {node.shape!r} "
                f"(rank {len(node.shape)})"
            )
        for inp in node.inputs:
            if inp not in defined:
                raise ValueError(
                    f"Node {node.name!r} references input {inp!r} which has "
                    "not been computed yet (graph is not topologically ordered)"
                )
        defined.add(node.name)


# --------------------------------------------------------------------------- #
# C source generation
# --------------------------------------------------------------------------- #
def generate_embedded_c(graph: Graph) -> str:
    """Generate a standalone, static-memory C99 program implementing ``graph``.

    Every buffer the program needs -- Linear/FusedLinearReLU weights and
    biases, QuantizedLinear/QuantizedFusedLinearReLU quantized weights,
    biases, scales, and int8 activation scratch, and every node's activation
    array -- is a file-scope ``static`` array with a compile-time-constant
    size. There is no ``malloc``/``free``/``new`` and no C++ standard
    library anywhere in the output: this is plain C99, compiled with
    ``-std=c99``.

    The generated program reads every graph Input node from stdin, in graph
    order (whitespace-separated doubles, ``_numel(shape)`` values per Input
    node), once. It then computes the graph once (default invocation,
    ``argc == 1``) or ``iters`` times (after ``warmup`` untimed repetitions)
    when invoked as ``<binary> <iters> <warmup>``, writing the *final*
    repetition's output node values to stdout. Passing explicit ``iters`` >
    1 additionally prints one line to stderr:
    ``BENCH_NS <total_nanoseconds> CHECKSUM <checksum>`` -- the same
    protocol :mod:`tinynn.codegen` uses, so :class:`CompiledModel` drives
    both backends unmodified.
    """
    _validate_for_embedded(graph)
    ids = _build_identifier_map(graph)

    lines: List[str] = []
    lines.append(
        "// Auto-generated by tinynn.embedded — experimental static-memory backend."
    )
    lines.append("// Plain C99: no dynamic allocation, no C++ containers, no C++ at all.")
    lines.append("//")
    lines.append("// Graph nodes:")
    for node in graph.nodes:
        lines.append(
            f"//   {node.name}: {node.op} inputs={list(node.inputs)} shape={node.shape}"
        )
    lines.append(f"// output_node: {graph.output_node}")
    lines.append("")
    lines.append("#define _POSIX_C_SOURCE 199309L")
    lines.append("#include <stdio.h>")
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <time.h>")
    lines.append("")

    # ------------------------------------------------------------------- #
    # Static memory plan: file-scope arrays only, every size a compile-time
    # constant. Weight/bias globals for Linear-like nodes come first, then
    # one static activation array per node (including Input and Output).
    # ------------------------------------------------------------------- #
    lines.append("// --- Static memory plan -------------------------------------------------")
    lines.append("// All weights, biases, quantization scratch, and per-node activation")
    lines.append("// buffers below are file-scope `static` arrays with fixed, compile-time")
    lines.append("// sizes -- there is no dynamic allocation anywhere in this program.")
    lines.append("")

    for node in graph.nodes:
        ident = ids[node.name]

        if node.op in (LINEAR, FUSED_LINEAR_RELU):
            in_features, out_features = node.weight.shape
            w_body = _format_double_array(node.weight)
            b_body = _format_double_array(node.bias)
            lines.append(
                f"// {node.op} node {node.name!r}: in_features={in_features}, "
                f"out_features={out_features}"
            )
            lines.append(f"static const double {ident}_w[{in_features * out_features}] = {{")
            lines.append(w_body)
            lines.append("};")
            lines.append(f"static const double {ident}_b[{out_features}] = {{")
            lines.append(b_body)
            lines.append("};")
            lines.append("")

        elif node.op in (QUANTIZED_LINEAR, QUANTIZED_FUSED_LINEAR_RELU):
            in_features, out_features = node.weight.shape
            w_q, s_w = _quantize_weight_for_codegen(node.weight)
            wq_body = _format_int_array(w_q)
            b_body = _format_double_array(node.bias)
            lines.append(
                f"// {node.op} node {node.name!r}: in_features={in_features}, "
                f"out_features={out_features}, s_w={format(s_w, '.17g')}"
            )
            lines.append(
                "// Weight quantized at codegen time (deterministic, "
                "round-half-away-from-zero)."
            )
            lines.append(
                f"static const int8_t {ident}_wq[{in_features * out_features}] = {{"
            )
            lines.append(wq_body)
            lines.append("};")
            lines.append(f"static const double {ident}_b[{out_features}] = {{")
            lines.append(b_body)
            lines.append("};")
            lines.append(f"static const double {ident}_sw = {format(s_w, '.17g')};")
            lines.append(
                f"static int8_t {ident}_xq[{in_features}];  // int8 activation scratch"
            )
            lines.append("")

    for node in graph.nodes:
        ident = ids[node.name]
        n = _numel(node.shape)
        lines.append(f"static double {ident}[{n}];  // {node.name}: {node.op} shape={node.shape}")
    lines.append("")

    # ------------------------------------------------------------------- #
    # compute(): the per-node statements, run once per repetition.
    # ------------------------------------------------------------------- #
    compute_lines: List[str] = []
    for node in graph.nodes:
        ident = ids[node.name]
        n = _numel(node.shape)

        if node.op == INPUT:
            # Filled once in main()'s prelude, not touched by compute().
            continue

        if node.op == LINEAR:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            compute_lines.append(f"    // Linear node {node.name!r}")
            compute_lines.extend(
                _emit_linear_like(ident, src_ident, in_features, out_features, relu=False)
            )
            compute_lines.append("")

        elif node.op == FUSED_LINEAR_RELU:
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            compute_lines.append(f"    // FusedLinearReLU node {node.name!r}")
            compute_lines.extend(
                _emit_linear_like(ident, src_ident, in_features, out_features, relu=True)
            )
            compute_lines.append("")

        elif node.op in (QUANTIZED_LINEAR, QUANTIZED_FUSED_LINEAR_RELU):
            src_ident = ids[node.inputs[0]]
            in_features, out_features = node.weight.shape
            compute_lines.append(f"    // {node.op} node {node.name!r}")
            compute_lines.extend(
                _emit_quantized_linear(
                    ident, src_ident, in_features, out_features,
                    relu=node.op == QUANTIZED_FUSED_LINEAR_RELU,
                )
            )
            compute_lines.append("")

        elif node.op == RELU:
            src_ident = ids[node.inputs[0]]
            compute_lines.append(f"    // ReLU node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(
                f"        {ident}[i] = {src_ident}[i] > 0.0 ? {src_ident}[i] : 0.0;"
            )
            compute_lines.append("    }")
            compute_lines.append("")

        elif node.op == OUTPUT:
            src_ident = ids[node.inputs[0]]
            compute_lines.append(f"    // Output node {node.name!r}")
            compute_lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
            compute_lines.append(f"        {ident}[i] = {src_ident}[i];")
            compute_lines.append("    }")
            compute_lines.append("")

        else:  # pragma: no cover - guarded by _validate_for_embedded
            raise ValueError(f"Unsupported op {node.op!r}")

    lines.append("static void compute(void) {")
    lines.extend(compute_lines)
    lines.append("}")
    lines.append("")

    out_ident = ids[graph.output_node]
    out_n = _numel(graph.get_node(graph.output_node).shape)

    lines.append("int main(int argc, char** argv) {")
    lines.append("    long iters = argc > 1 ? atol(argv[1]) : 1;")
    lines.append("    long warmup = argc > 2 ? atol(argv[2]) : 0;")
    lines.append("")

    for node in graph.nodes:
        if node.op != INPUT:
            continue
        ident = ids[node.name]
        n = _numel(node.shape)
        lines.append(f"    // Input node {node.name!r}")
        lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
        lines.append(f"        if (scanf(\"%lf\", &{ident}[i]) != 1) {{")
        lines.append(
            f'            fprintf(stderr, "Error: failed to read input value %d for '
            f'node \\"{node.name}\\" (expected {n} doubles)\\n", i);'
        )
        lines.append("            return 1;")
        lines.append("        }")
        lines.append("    }")
        lines.append("")

    lines.append("    double checksum = 0.0;")
    lines.append("")
    lines.append("    for (long rep = 0; rep < warmup; ++rep) {")
    lines.append("        compute();")
    lines.append("    }")
    lines.append("")
    lines.append("    struct timespec tinynn_bench_start, tinynn_bench_end;")
    lines.append("    clock_gettime(CLOCK_MONOTONIC, &tinynn_bench_start);")
    lines.append("    for (long rep = 0; rep < iters; ++rep) {")
    lines.append("        compute();")
    lines.append(f"        checksum += {out_ident}[{out_n} - 1];")
    lines.append("    }")
    lines.append("    clock_gettime(CLOCK_MONOTONIC, &tinynn_bench_end);")
    lines.append("")

    lines.append(f"    // print output node {graph.output_node!r}")
    lines.append(f"    for (int i = 0; i < {out_n}; ++i) {{")
    lines.append(f'        printf(i ? " %.17g" : "%.17g", {out_ident}[i]);')
    lines.append("    }")
    lines.append('    printf("\\n");')
    lines.append("")

    lines.append("    if (iters > 1) {")
    lines.append(
        "        long long tinynn_bench_ns = "
        "(long long)(tinynn_bench_end.tv_sec - tinynn_bench_start.tv_sec) * 1000000000LL"
        " + (long long)(tinynn_bench_end.tv_nsec - tinynn_bench_start.tv_nsec);"
    )
    lines.append(
        '        fprintf(stderr, "BENCH_NS %lld CHECKSUM %.17g\\n", '
        "tinynn_bench_ns, checksum);"
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
    relu: bool,
) -> List[str]:
    """Return compute-only statements for a Linear or FusedLinearReLU node."""
    lines: List[str] = []
    lines.append(f"    for (int j = 0; j < {out_features}; ++j) {{")
    lines.append(f"        double acc = {ident}_b[j];")
    lines.append(f"        for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"            acc += {src_ident}[i] * {ident}_w[i * {out_features} + j];")
    lines.append("        }")
    if relu:
        lines.append(f"        {ident}[j] = acc > 0.0 ? acc : 0.0;")
    else:
        lines.append(f"        {ident}[j] = acc;")
    lines.append("    }")
    return lines


def _emit_quantized_linear(
    ident: str,
    src_ident: str,
    in_features: int,
    out_features: int,
    relu: bool,
) -> List[str]:
    """Return compute-only statements for a QuantizedLinear node (or, with
    ``relu=True``, a QuantizedFusedLinearReLU node).

    Mirrors :func:`tinynn.codegen._emit_quantized_linear` exactly (which in
    turn mirrors :func:`tinynn.interpreter._eval_quantized_linear`):
    per-call activation quantization (dynamic ``s_x``), a plain int8 x int8
    matmul accumulated into a 64-bit ``acc``, then one float rescale
    (``sx * sw``) plus the float bias add. Rounding is round-half-away-from-
    zero via ``lround`` (matches the interpreter's
    ``sign(v) * floor(abs(v) + 0.5)``, NOT round-half-to-even). With
    ``relu=True`` the ReLU clamp is applied LAST, after the rescale and bias
    add.
    """
    lines: List[str] = []
    lines.append("    {")
    lines.append(
        "        // Dynamic activation quantization: s_x = max(|x|) / 127, "
        "or 1.0 if x is all zero."
    )
    lines.append("        double maxabs = 0.0;")
    lines.append(f"        for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"            double av = fabs({src_ident}[i]);")
    lines.append("            if (av > maxabs) maxabs = av;")
    lines.append("        }")
    lines.append("        double sx = maxabs > 0.0 ? maxabs / 127.0 : 1.0;")
    lines.append(
        "        // Quantize: round-half-away-from-zero (lround), clip to int8 range."
    )
    lines.append(f"        for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(f"            long long q = lround({src_ident}[i] / sx);")
    lines.append("            if (q < -127) q = -127;")
    lines.append("            if (q > 127) q = 127;")
    lines.append(f"            {ident}_xq[i] = (int8_t)q;")
    lines.append("        }")
    lines.append(
        "        // Integer matmul (64-bit accumulation), then a single float "
        "rescale + bias add."
    )
    lines.append(f"        for (int j = 0; j < {out_features}; ++j) {{")
    lines.append("            int64_t acc = 0;")
    lines.append(f"            for (int i = 0; i < {in_features}; ++i) {{")
    lines.append(
        f"                acc += (int64_t){ident}_xq[i] * (int64_t){ident}_wq[i * {out_features} + j];"
    )
    lines.append("            }")
    lines.append(
        f"            double v = (double)acc * (sx * {ident}_sw) + {ident}_b[j];"
    )
    if relu:
        lines.append(f"            {ident}[j] = v > 0.0 ? v : 0.0;")
    else:
        lines.append(f"            {ident}[j] = v;")
    lines.append("        }")
    lines.append("    }")
    return lines


# --------------------------------------------------------------------------- #
# Compiler driver
# --------------------------------------------------------------------------- #
def compile_embedded(
    graph: Graph,
    output_dir,
    binary_name: str = "tinynn_model",
    c_name: str = "tinynn_model.c",
    compiler: str = "cc",
    extra_flags: Optional[List[str]] = None,
) -> CompiledModel:
    """Generate C99 for ``graph``, compile it, and return a :class:`CompiledModel`.

    Parameters
    ----------
    graph:
        The Graph IR to compile. Must have at least one Input node, only 1D
        or 2D node shapes, and use only the op subset documented in
        :func:`generate_embedded_c` (see :mod:`tinynn.embedded`'s module
        docstring).
    output_dir:
        Directory to write the generated ``.c`` file and the compiled
        binary into. Created (with parents) if it does not exist.
    binary_name / c_name:
        File names (not paths) for the compiled binary and generated source.
    compiler:
        Compiler executable to invoke (default ``"cc"``).
    extra_flags:
        Additional flags appended to the compiler invocation, after
        ``-O2 -std=c99 -o <binary> -lm``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c_path = output_dir / c_name
    binary_path = output_dir / binary_name

    source = generate_embedded_c(graph)
    c_path.write_text(source)

    cmd = [compiler, str(c_path), "-O2", "-std=c99", "-o", str(binary_path), "-lm"]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Compiler {compiler!r} not found, is a C compiler installed?"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Compilation of {c_path} with {compiler!r} failed "
            f"(exit code {proc.returncode}):\n{proc.stderr}"
        )

    input_specs = tuple((n.name, n.shape) for n in graph.nodes if n.op == INPUT)
    output_shape = graph.get_node(graph.output_node).shape

    return CompiledModel(
        cpp_path=c_path,
        binary_path=binary_path,
        output_shape=output_shape,
        input_specs=input_specs,
    )
