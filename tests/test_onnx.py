"""Tests for the optional ONNX importer (tinynn.onnx_import).

The whole module is skipped when the ``onnx`` package is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tinynn import interpreter  # noqa: E402
from tinynn.onnx_import import import_onnx  # noqa: E402

RNG = np.random.default_rng(1234)


def f32(shape):
    """Deterministic small float32 tensor."""
    return (RNG.standard_normal(shape) * 0.5).astype(np.float32)


def make_model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "tinynn_test",
        inputs,
        outputs,
        initializer=[numpy_helper.from_array(arr, name) for name, arr in initializers],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model


def vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


# --------------------------------------------------------------------------- #
# 1. PyTorch-style MLP: (1, 4) input, Gemm(transB=1) -> Relu -> Gemm(transB=1)
# --------------------------------------------------------------------------- #
def test_pytorch_style_mlp():
    W1, b1 = f32((8, 4)), f32((8,))
    W2, b2 = f32((3, 8)), f32((3,))
    model = make_model(
        nodes=[
            helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1),
            helper.make_node("Relu", ["h"], ["a"]),
            helper.make_node("Gemm", ["a", "W2", "b2"], ["y"], transB=1),
        ],
        inputs=[vi("x", [1, 4])],
        outputs=[vi("y", [1, 3])],
        initializers=[("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)],
    )

    graph = import_onnx(model)

    nodes = graph.node_map()
    assert nodes["x"].op == "Input" and nodes["x"].shape == (4,)  # batch squeezed
    assert nodes["h"].op == "Linear" and nodes["h"].shape == (8,)
    assert nodes["h"].weight.shape == (4, 8)  # transB applied -> (in, out)
    assert nodes["h"].weight.dtype == np.float64
    assert nodes["h"].bias.dtype == np.float64
    assert nodes["a"].op == "ReLU"
    assert nodes["y"].op == "Linear" and nodes["y"].shape == (3,)
    assert graph.output_node == "y"

    x = np.array([0.3, -1.2, 0.7, 2.1])
    got = interpreter.run(graph, {"x": x})
    W1d, b1d = W1.astype(np.float64), b1.astype(np.float64)
    W2d, b2d = W2.astype(np.float64), b2.astype(np.float64)
    expected = np.maximum(x @ W1d.T + b1d, 0.0) @ W2d.T + b2d
    np.testing.assert_allclose(got, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Gemm with transB=0 and no C (bias absent -> zeros)
# --------------------------------------------------------------------------- #
def test_gemm_no_trans_no_bias():
    W = f32((4, 5))
    model = make_model(
        nodes=[helper.make_node("Gemm", ["x", "W"], ["y"])],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [5])],
        initializers=[("W", W)],
    )
    graph = import_onnx(model)
    node = graph.get_node("y")
    assert node.op == "Linear"
    assert node.weight.shape == (4, 5)  # transB=0: used as-is
    np.testing.assert_array_equal(node.bias, np.zeros(5))

    x = np.array([1.0, -2.0, 0.5, 3.0])
    got = interpreter.run(graph, {"x": x})
    np.testing.assert_allclose(got, x @ W.astype(np.float64), atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. Chain with Tanh, Sigmoid, Softmax(axis=-1)
# --------------------------------------------------------------------------- #
def test_tanh_sigmoid_softmax_chain():
    W, b = f32((4, 6)), f32((6,))
    model = make_model(
        nodes=[
            helper.make_node("Gemm", ["x", "W", "b"], ["h"]),
            helper.make_node("Tanh", ["h"], ["t"]),
            helper.make_node("Sigmoid", ["t"], ["s"]),
            helper.make_node("Softmax", ["s"], ["p"], axis=-1),
        ],
        inputs=[vi("x", [1, 4])],
        outputs=[vi("p", [1, 6])],
        initializers=[("W", W), ("b", b)],
    )
    graph = import_onnx(model)
    nodes = graph.node_map()
    assert nodes["t"].op == "Tanh"
    assert nodes["s"].op == "Sigmoid"
    assert nodes["p"].op == "Softmax"

    x = np.array([0.1, -0.4, 1.5, -2.0])
    got = interpreter.run(graph, {"x": x})
    h = x @ W.astype(np.float64) + b.astype(np.float64)
    s = 1.0 / (1.0 + np.exp(-np.tanh(h)))
    e = np.exp(s - np.max(s))  # stable softmax
    expected = e / np.sum(e)
    np.testing.assert_allclose(got, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. MatMul(activation, initializer) -> Linear with zero bias
# --------------------------------------------------------------------------- #
def test_matmul_with_initializer_is_linear():
    W = f32((4, 3))
    model = make_model(
        nodes=[helper.make_node("MatMul", ["x", "W"], ["y"])],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [3])],
        initializers=[("W", W)],
    )
    graph = import_onnx(model)
    node = graph.get_node("y")
    assert node.op == "Linear"
    assert node.weight.shape == (4, 3)
    assert node.weight.dtype == np.float64
    np.testing.assert_array_equal(node.bias, np.zeros(3))

    x = np.array([0.5, 1.5, -0.5, 2.0])
    got = interpreter.run(graph, {"x": x})
    np.testing.assert_allclose(got, x @ W.astype(np.float64), atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. Add of two activation branches (diamond off one input)
# --------------------------------------------------------------------------- #
def test_add_two_activation_branches():
    model = make_model(
        nodes=[
            helper.make_node("Relu", ["x"], ["r"]),
            helper.make_node("Tanh", ["x"], ["t"]),
            helper.make_node("Add", ["r", "t"], ["y"]),
        ],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [4])],
        initializers=[],
    )
    graph = import_onnx(model)
    node = graph.get_node("y")
    assert node.op == "Add"
    assert set(node.inputs) == {"r", "t"}

    x = np.array([-1.0, 0.5, 2.0, -0.25])
    got = interpreter.run(graph, {"x": x})
    np.testing.assert_allclose(got, np.maximum(x, 0.0) + np.tanh(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. Identity node aliasing
# --------------------------------------------------------------------------- #
def test_identity_aliases_without_new_node():
    model = make_model(
        nodes=[
            helper.make_node("Identity", ["x"], ["x_alias"]),
            helper.make_node("Relu", ["x_alias"], ["y"]),
        ],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [4])],
        initializers=[],
    )
    graph = import_onnx(model)
    names = [n.name for n in graph.nodes]
    assert "x_alias" not in names  # no node created for Identity
    assert graph.get_node("y").inputs == ("x",)  # Relu rewired to the source

    x = np.array([-1.0, 2.0, -3.0, 4.0])
    got = interpreter.run(graph, {"x": x})
    np.testing.assert_allclose(got, np.maximum(x, 0.0), atol=1e-6)


def test_identity_as_graph_output():
    model = make_model(
        nodes=[
            helper.make_node("Relu", ["x"], ["r"]),
            helper.make_node("Identity", ["r"], ["y"]),
        ],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [4])],
        initializers=[],
    )
    graph = import_onnx(model)
    assert graph.output_node == "r"  # alias resolved


# --------------------------------------------------------------------------- #
# 7. Error cases
# --------------------------------------------------------------------------- #
def test_unsupported_op_raises_with_op_name():
    model = make_model(
        nodes=[
            helper.make_node(
                "Conv", ["x", "W"], ["y"], kernel_shape=[1], pads=[0, 0], strides=[1]
            )
        ],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [4])],
        initializers=[("W", f32((2, 4, 1)))],
    )
    with pytest.raises(ValueError, match="Conv"):
        import_onnx(model)


def test_gemm_alpha_not_one_raises():
    model = make_model(
        nodes=[helper.make_node("Gemm", ["x", "W", "b"], ["y"], alpha=2.0)],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [5])],
        initializers=[("W", f32((4, 5))), ("b", f32((5,)))],
    )
    with pytest.raises(ValueError, match="alpha"):
        import_onnx(model)


def test_symbolic_input_dim_raises():
    model = make_model(
        nodes=[helper.make_node("Relu", ["x"], ["y"])],
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        outputs=[vi("y", ["batch", 4])],
        initializers=[],
    )
    with pytest.raises(ValueError, match="'x'"):
        import_onnx(model)


def test_batch_dim_not_one_raises():
    model = make_model(
        nodes=[helper.make_node("Relu", ["x"], ["y"])],
        inputs=[vi("x", [3, 4])],
        outputs=[vi("y", [3, 4])],
        initializers=[],
    )
    with pytest.raises(ValueError, match="'x'"):
        import_onnx(model)


def test_add_with_initializer_raises():
    model = make_model(
        nodes=[helper.make_node("Add", ["x", "c"], ["y"])],
        inputs=[vi("x", [4])],
        outputs=[vi("y", [4])],
        initializers=[("c", f32((4,)))],
    )
    with pytest.raises(ValueError, match="unsupported initializer usage"):
        import_onnx(model)


# --------------------------------------------------------------------------- #
# 8. Round-trip through a saved file
# --------------------------------------------------------------------------- #
def test_import_from_saved_file(tmp_path):
    W, b = f32((4, 5)), f32((5,))
    model = make_model(
        nodes=[
            helper.make_node("Gemm", ["x", "W", "b"], ["h"]),
            helper.make_node("Relu", ["h"], ["y"]),
        ],
        inputs=[vi("x", [1, 4])],
        outputs=[vi("y", [1, 5])],
        initializers=[("W", W), ("b", b)],
    )
    path = tmp_path / "mlp.onnx"
    onnx.save(model, str(path))

    for model_arg in (path, str(path)):  # Path and str branches
        graph = import_onnx(model_arg)
        x = np.array([1.0, -1.0, 0.5, 2.0])
        got = interpreter.run(graph, {"x": x})
        expected = np.maximum(
            x @ W.astype(np.float64) + b.astype(np.float64), 0.0
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)
