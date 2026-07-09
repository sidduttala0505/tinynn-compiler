"""Supported operation names for the TinyNN Graph IR.

These constants are the single source of truth for op types. Every later
component (interpreter, C++ codegen, optimization passes) should refer to
these names rather than hard-coding string literals.
"""

from __future__ import annotations

# Canonical op names.
INPUT = "Input"
LINEAR = "Linear"
RELU = "ReLU"
OUTPUT = "Output"
FUSED_LINEAR_RELU = "FusedLinearReLU"

# Tier 2 ops: elementwise binary (Add/Sub/Mul), 2D MatMul, and unary
# activations (Softmax/Tanh/Sigmoid). None of these carry weight/bias.
ADD = "Add"
SUB = "Sub"
MUL = "Mul"
MATMUL = "MatMul"
SOFTMAX = "Softmax"
TANH = "Tanh"
SIGMOID = "Sigmoid"

# Tier 3: a compile-time constant value. A Const node's value lives in the
# existing `weight` field (1D or 2D, deliberately reusing Node's schema
# instead of adding a new one); `bias` must be None and the node has no
# inputs.
CONST = "Const"

# Tier 6: a Linear node executed with int8 quantized weights and dynamically
# (per-call) quantized int8 activations. The node stores the ORIGINAL float64
# weight/bias, exactly like Linear (same shape/validation rules) -- there is
# no separate quantized dtype in the schema. Quantization is instead a pure,
# deterministic function of those floats, applied identically at interpreter
# eval time and at C++ codegen time:
#
#   s_w = max(abs(weight)) / 127.0, or 1.0 if weight is all zeros
#   w_q = clip(round_half_away_from_zero(weight / s_w), -127, 127)   (int)
#   s_x = max(abs(x)) / 127.0, or 1.0 if x is all zeros              (per call)
#   x_q = clip(round_half_away_from_zero(x / s_x), -127, 127)        (int)
#   y   = (sum_i x_q[i] * w_q[i][j]) * (s_x * s_w) + bias[j]
#
# with the matmul accumulated in 64-bit integers and only a single float
# rescale (times bias add) at the end. Rounding must be round-half-away-from-
# zero (matching C++ std::lround), NOT NumPy's default round-half-even.
QUANTIZED_LINEAR = "QuantizedLinear"

# The set of ops the IR currently understands.
# NOTE: "FusedLinearReLU" fuses a Linear immediately followed by a ReLU into a
# single node (same weight/bias as the Linear, output already clamped at 0).
# It is produced by the Linear+ReLU fusion pass in tinynn.passes, but can also
# be constructed directly like any other op.
SUPPORTED_OPS = {
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
