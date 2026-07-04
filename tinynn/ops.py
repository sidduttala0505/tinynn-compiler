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

# The set of ops the IR currently understands.
# NOTE: "FusedLinearReLU" fuses a Linear immediately followed by a ReLU into a
# single node (same weight/bias as the Linear, output already clamped at 0).
# It is produced by the Linear+ReLU fusion pass in tinynn.passes, but can also
# be constructed directly like any other op.
SUPPORTED_OPS = {INPUT, LINEAR, RELU, OUTPUT, FUSED_LINEAR_RELU}
