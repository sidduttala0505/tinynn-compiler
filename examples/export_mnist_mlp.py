"""Dataset loading, pure-NumPy MLP training, and ONNX export for the
"MNIST-style" demo.

HONESTY NOTE: the dataset used here is scikit-learn's ``load_digits`` --
8x8 grayscale digit images (64 features), 10 classes, 1797 samples total.
It is visually and structurally similar to MNIST (small handwritten-digit
images) but it is **not** MNIST (which is 28x28, 70000 samples). Every
report and print statement in this example calls it "sklearn digits (8x8)"
and never claims to be the real MNIST dataset. If scikit-learn is not
installed, a deterministic synthetic Gaussian-blobs dataset (same shapes:
64 features, 10 classes) is used instead as a fallback, and is labeled as
such.

No PyTorch is used anywhere in this example: the MLP is trained with plain
NumPy (forward pass, softmax cross-entropy, manual backprop, mini-batch
SGD), and the trained weights are exported to ONNX by hand with
``onnx.helper`` so the resulting model can be imported by
``tinynn.import_onnx`` and run through the TinyNN interpreter / C++ codegen.

Run directly to train and export a model to ``examples/models/digits_mlp.onnx``:

    ./venv/bin/python examples/export_mnist_mlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Allow running as `python examples/export_mnist_mlp.py` (repo root not on
# sys.path by default) as well as `python -m examples.export_mnist_mlp`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent / "models"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _synthetic_fallback(seed: int, n_samples: int = 1797, n_features: int = 64,
                         n_classes: int = 10):
    """Deterministic synthetic Gaussian-blobs dataset with the same shapes as
    sklearn digits (64 features, 10 classes), used only when scikit-learn is
    not installed."""
    rng = np.random.default_rng(seed)
    class_means = rng.uniform(0.2, 0.8, size=(n_classes, n_features))
    per_class = n_samples // n_classes
    remainder = n_samples - per_class * n_classes

    X_parts = []
    y_parts = []
    for c in range(n_classes):
        n_c = per_class + (1 if c < remainder else 0)
        noise = rng.normal(0.0, 0.08, size=(n_c, n_features))
        X_c = np.clip(class_means[c] + noise, 0.0, 1.0)
        X_parts.append(X_c)
        y_parts.append(np.full(n_c, c, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X.astype(np.float64), y


def load_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the demo dataset and return a deterministic shuffled 75/25 split.

    Returns ``(X_train, y_train, X_test, y_test, dataset_name)``.

    ``dataset_name`` is ``"sklearn digits (8x8)"`` when scikit-learn is
    available (the normal path), or ``"synthetic gaussian blobs (fallback;
    scikit-learn not installed)"`` otherwise. X is scaled to ``[0, 1]``.
    """
    try:
        from sklearn.datasets import load_digits
    except ImportError:
        X, y = _synthetic_fallback(seed)
        dataset_name = "synthetic gaussian blobs (fallback; scikit-learn not installed)"
    else:
        digits = load_digits()
        X = digits.data.astype(np.float64) / 16.0  # pixel values are 0..16
        y = digits.target.astype(np.int64)
        dataset_name = "sklearn digits (8x8)"

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    n_test = int(round(n * 0.25))
    n_train = n - n_test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test, dataset_name


# --------------------------------------------------------------------------- #
# Pure-NumPy MLP: 64 -> hidden (ReLU) -> 10, softmax cross-entropy
# --------------------------------------------------------------------------- #
def _forward(params: Dict[str, np.ndarray], X: np.ndarray):
    """Forward pass. Returns (logits, hidden_pre_relu, hidden_post_relu)."""
    z0 = X @ params["W0"] + params["b0"]
    a0 = np.maximum(z0, 0.0)
    logits = a0 @ params["W1"] + params["b1"]
    return logits, z0, a0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    hidden: int = 32,
    epochs: int = 30,
    lr: float = 0.1,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Train a 64 -> hidden (ReLU) -> 10 MLP with plain NumPy mini-batch SGD.

    Softmax cross-entropy loss, manual backprop, fixed seed for weight
    initialization and per-epoch shuffling (deterministic given ``seed``).
    Returns float64 weights ``{"W0", "b0", "W1", "b1"}``.
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    n_classes = int(np.max(y_train)) + 1
    n_samples = X_train.shape[0]

    # He-style init for ReLU hidden layer, small init for the output layer.
    W0 = rng.standard_normal((n_features, hidden)) * np.sqrt(2.0 / n_features)
    b0 = np.zeros(hidden)
    W1 = rng.standard_normal((hidden, n_classes)) * np.sqrt(2.0 / hidden)
    b1 = np.zeros(n_classes)
    params = {"W0": W0, "b0": b0, "W1": W1, "b1": b1}

    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_train] = 1.0

    for _epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuf = X_train[perm]
        y_shuf = y_onehot[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]
            b = xb.shape[0]

            logits, z0, a0 = _forward(params, xb)
            probs = _softmax(logits)

            # dL/dlogits for mean softmax cross-entropy.
            dlogits = (probs - yb) / b

            dW1 = a0.T @ dlogits
            db1 = np.sum(dlogits, axis=0)

            da0 = dlogits @ params["W1"].T
            dz0 = da0 * (z0 > 0.0)

            dW0 = xb.T @ dz0
            db0 = np.sum(dz0, axis=0)

            params["W0"] -= lr * dW0
            params["b0"] -= lr * db0
            params["W1"] -= lr * dW1
            params["b1"] -= lr * db1

    return {k: v.astype(np.float64) for k, v in params.items()}


def accuracy(params: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    """Forward-pass argmax accuracy of ``params`` on ``(X, y)``."""
    logits, _, _ = _forward(params, X)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(params: Dict[str, np.ndarray], path) -> None:
    """Export ``params`` (a 64 -> 32 ReLU -> 10 MLP) to an ONNX model at ``path``.

    Builds the graph by hand with ``onnx.helper`` (no torch):
    ``x [1, 64] -> Gemm(W0, b0) -> Relu -> Gemm(W1, b1) -> logits [1, 10]``.
    Weights are stored as float32 initializers (standard ONNX practice).
    The model is checked with ``onnx.checker`` and, as a final sanity check,
    round-tripped through ``tinynn.import_onnx`` to confirm the repo's
    importer accepts it.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W0 = params["W0"].astype(np.float32)
    b0 = params["b0"].astype(np.float32)
    W1 = params["W1"].astype(np.float32)
    b1 = params["b1"].astype(np.float32)

    in_features, hidden = W0.shape
    hidden2, n_classes = W1.shape
    assert hidden == hidden2

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_features])
    out_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_classes])

    init_W0 = numpy_helper.from_array(W0, name="W0")
    init_b0 = numpy_helper.from_array(b0, name="b0")
    init_W1 = numpy_helper.from_array(W1, name="W1")
    init_b1 = numpy_helper.from_array(b1, name="b1")

    node_gemm0 = helper.make_node(
        "Gemm", ["x", "W0", "b0"], ["h0"],
        name="gemm0", alpha=1.0, beta=1.0, transA=0, transB=0,
    )
    node_relu = helper.make_node("Relu", ["h0"], ["a0"], name="relu0")
    node_gemm1 = helper.make_node(
        "Gemm", ["a0", "W1", "b1"], ["logits"],
        name="gemm1", alpha=1.0, beta=1.0, transA=0, transB=0,
    )

    graph_def = helper.make_graph(
        [node_gemm0, node_relu, node_gemm1],
        "digits_mlp",
        [x_info],
        [out_info],
        initializer=[init_W0, init_b0, init_W1, init_b1],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="tinynn-examples",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )

    onnx.checker.check_model(model_def)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_def, str(path))

    # Final sanity check: confirm the repo's own importer accepts this file.
    from tinynn import import_onnx

    import_onnx(path)


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    seed = 0
    X_train, y_train, X_test, y_test, dataset_name = load_dataset(seed)
    print(f"Dataset: {dataset_name}")
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    params = train_mlp(X_train, y_train, seed=seed)
    test_acc = accuracy(params, X_test, y_test)
    print(f"NumPy MLP test accuracy: {test_acc:.4f}")

    out_path = MODELS_DIR / "digits_mlp.onnx"
    export_onnx(params, out_path)
    print(f"Exported ONNX model to {out_path}")


if __name__ == "__main__":
    main()
