# MNIST-style MLP demo report (TinyNN)

- generated: 2026-07-07T23:29:10.852070+00:00
- total wall time: 3.9s

## Model architecture

64 -> 32 (ReLU) -> 10, trained in **pure NumPy** (forward pass, manual
backprop, mini-batch SGD, softmax cross-entropy). No PyTorch was used
anywhere in this demo -- the "NumPy baseline" below is that same NumPy
training forward pass, evaluated directly (not re-derived from ONNX).

## Dataset

**sklearn digits (8x8)** -- this is NOT the real MNIST dataset (which is
28x28, 70000 samples); it is scikit-learn's 8x8 handwritten-digit
dataset, used here as a small, fast, MNIST-style stand-in.

- train samples: 1348
- test samples: 449
- features: 64
- classes: 10
- seed: 0, epochs: 30

## Accuracy

| stage | test accuracy |
|---|---|
| NumPy baseline (train) | 0.9770 |
| NumPy baseline (test) | 0.9733 |
| TinyNN interpreter (test) | 0.9733 |
| TinyNN interpreter, optimized graph (test) | 0.9733 |
| TinyNN compiled C++ optimized (test) | 0.9733 |
| int8 interpreter (test) | 0.9755 |
| int8 compiled C++ (test) | 0.9755 |

Max |interpreter logits - NumPy forward logits| over 25 samples: 3.940e-07 (bound: float32 ONNX weight rounding, < 1e-06).

## Latency

Benchmark iterations: 2000 (warmup 200). OpenMP used: False.

| variant | s/iter | speedup vs C++ naive |
|---|---|---|
| interpreter (NumPy) | 2.915e-06 | 0.24x |
| C++ naive | 7.135e-07 | 1.00x |
| C++ optimized (float, FusedLinearReLU) | 4.182e-07 | 1.71x |
| C++ int8 (QuantizedFusedLinearReLU) | 5.728e-07 | 1.25x |

**FusedLinearReLU (float, optimized) vs QuantizedFusedLinearReLU (int8)**: 0.73x (float optimized faster or equal).

## Verification note

All compiled C++ variants (naive, optimized float, int8 quantized) were verified against the NumPy interpreter oracle on 50 test samples with `np.allclose(rtol=1e-09, atol=1e-09)` **before** any latency measurement was taken. The optimized-graph interpreter accuracy was also checked to match the original-graph interpreter accuracy exactly (same float64 numerics; fusion is not lossy).

## Reproduction

```
./venv/bin/python examples/mnist_mlp_demo.py --out results/mnist --seed 0 --epochs 30 --iters 2000
```

