"""Op name constants. Everything else imports these instead of hardcoding strings."""

from __future__ import annotations

INPUT = "Input"
LINEAR = "Linear"
RELU = "ReLU"
OUTPUT = "Output"
FUSED_LINEAR_RELU = "FusedLinearReLU"

# Tier 2 ops. Add/Sub/Mul are elementwise (same-shape inputs), MatMul is 2D,
# and Softmax/Tanh/Sigmoid are unary. None of them have weights.
ADD = "Add"
SUB = "Sub"
MUL = "Mul"
MATMUL = "MatMul"
SOFTMAX = "Softmax"
TANH = "Tanh"
SIGMOID = "Sigmoid"

# A constant value. I stuff the value into the `weight` field so I didn't have
# to change the Node schema. bias is None and there are no inputs.
CONST = "Const"

# int8 quantized Linear. Stores the normal float weight/bias like Linear does;
# the quantization happens the same way in the interpreter and in the generated
# C++ so the two match exactly. See interpreter._eval_quantized_linear for the math.
QUANTIZED_LINEAR = "QuantizedLinear"

# Same as QuantizedLinear but with the ReLU clamp tacked on at the very end
# (after the rescale + bias). So the quantized part is identical, just max(0, ...)
# on top. The quantize_fused_linear_relu pass makes these out of FusedLinearReLU.
QUANTIZED_FUSED_LINEAR_RELU = "QuantizedFusedLinearReLU"

# FusedLinearReLU is a Linear + ReLU squished into one node. Usually the fusion
# pass makes it but you can build one directly too.
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
    QUANTIZED_FUSED_LINEAR_RELU,
}
