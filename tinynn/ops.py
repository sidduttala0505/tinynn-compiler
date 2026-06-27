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

# The set of ops the IR currently understands.
# NOTE: "FusedLinearReLU" is intentionally absent; it is introduced later,
# during the operator-fusion pass.
SUPPORTED_OPS = {INPUT, LINEAR, RELU, OUTPUT}
