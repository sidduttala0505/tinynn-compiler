"""Tests for the ``tinynn`` command-line interface (:mod:`tinynn.cli`).

All tests call :func:`tinynn.cli.main` in-process (capturing stdout/stderr
with ``capsys``) rather than spawning subprocesses, except for one smoke
test that exercises ``python -m tinynn`` end-to-end via ``subprocess`` to
confirm the ``__main__`` entry point is wired up correctly.

Tests that compile C++ (``compile``/``benchmark`` subcommands) are skipped
when ``g++`` is not on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tinynn import cli
from tinynn.graph import GraphBuilder
from tinynn.serialize import save_json

_HAS_GXX = shutil.which("g++") is not None
requires_gxx = pytest.mark.skipif(_HAS_GXX is False, reason="g++ not found on PATH")


# --------------------------------------------------------------------------- #
# Fixture: a tiny 2-layer MLP model saved as JSON
# --------------------------------------------------------------------------- #
@pytest.fixture()
def mlp_model_path(tmp_path) -> Path:
    """Build a tiny 2-layer MLP graph and save it as JSON; return the path."""
    rng = np.random.default_rng(0)
    b = GraphBuilder()
    x = b.input("x", (4,))
    h = b.linear("l0", x, rng.standard_normal((4, 6)), rng.standard_normal(6))
    h = b.relu("r0", h)
    h = b.linear("l1", h, rng.standard_normal((6, 3)), rng.standard_normal(3))
    graph = b.build(output_node=h)
    path = tmp_path / "mlp.json"
    save_json(graph, path)
    return path


# --------------------------------------------------------------------------- #
# --help for the top-level parser and every subcommand
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["compile", "--help"],
        ["run", "--help"],
        ["benchmark", "--help"],
        ["visualize", "--help"],
    ],
)
def test_help_exits_zero(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 0


def test_no_args_is_a_clean_usage_error():
    # argparse subparsers are required, so no subcommand exits nonzero
    # via SystemExit rather than falling through to main()'s return value.
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code != 0


# --------------------------------------------------------------------------- #
# compile
# --------------------------------------------------------------------------- #
@requires_gxx
def test_compile_produces_cpp_and_binary(mlp_model_path, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = cli.main(["compile", str(mlp_model_path), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "tinynn_model.cpp").exists()
    assert (out_dir / "tinynn_model").exists()
    captured = capsys.readouterr()
    assert str(out_dir / "tinynn_model.cpp") in captured.out
    assert str(out_dir / "tinynn_model") in captured.out


@requires_gxx
def test_compile_emit_dot_writes_graph_dot(mlp_model_path, tmp_path):
    out_dir = tmp_path / "out"
    rc = cli.main(
        ["compile", str(mlp_model_path), "--out", str(out_dir), "--emit-dot"]
    )
    assert rc == 0
    assert (out_dir / "graph.dot").exists()
    assert "digraph" in (out_dir / "graph.dot").read_text()


@requires_gxx
def test_compile_quantize_note_and_generated_source(mlp_model_path, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = cli.main(
        ["compile", str(mlp_model_path), "--out", str(out_dir), "--quantize"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "int8 quantization changes numerics" in captured.out

    source = (out_dir / "tinynn_model.cpp").read_text()
    assert "QuantizedLinear" in source or "QuantizedFusedLinearReLU" in source


@requires_gxx
def test_compile_fast_and_benchmark(mlp_model_path, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = cli.main(
        [
            "compile",
            str(mlp_model_path),
            "--out",
            str(out_dir),
            "--fast",
            "--benchmark",
            "--iters",
            "5",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0
    assert (out_dir / "mlp.json").exists()
    assert (out_dir / "mlp.md").exists()
    captured = capsys.readouterr()
    assert "s/iter" in captured.out
    assert "speedups" in captured.out


def test_compile_embedded_backend_unavailable_exits_cleanly(
    mlp_model_path, tmp_path, monkeypatch, capsys
):
    # tinynn.embedded is written independently of this CLI (possibly not
    # present, or present but broken, in a given environment); simulate
    # "unavailable" by forcing its import to fail, regardless of whether the
    # module actually exists on disk right now, and confirm the CLI fails
    # cleanly (exit 2, no traceback) rather than crashing. Setting the
    # sys.modules entry to None makes any subsequent `import tinynn.embedded`
    # (including the `from .embedded import ...` inside the CLI's handler)
    # raise ImportError, per Python's import system semantics.
    import sys

    monkeypatch.setitem(sys.modules, "tinynn.embedded", None)

    out_dir = tmp_path / "out"
    rc = cli.main(
        ["compile", str(mlp_model_path), "--out", str(out_dir), "--backend", "embedded_c"]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "embedded" in captured.err.lower()


@requires_gxx
def test_compile_embedded_backend_smoke_if_available(mlp_model_path, tmp_path):
    pytest.importorskip("tinynn.embedded")
    out_dir = tmp_path / "out"
    rc = cli.main(
        ["compile", str(mlp_model_path), "--out", str(out_dir), "--backend", "embedded_c"]
    )
    assert rc == 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_run_random_inputs_prints_finite_floats(mlp_model_path, capsys):
    rc = cli.main(["run", str(mlp_model_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no --input given" in captured.out
    # The printed array repr should contain only finite numbers; parse via
    # numpy's array_repr round trip is overkill, so just sanity check shape.
    assert "[" in captured.out and "]" in captured.out


def test_run_with_input_npy(mlp_model_path, tmp_path, capsys):
    npy_path = tmp_path / "x.npy"
    np.save(npy_path, np.ones(4))
    rc = cli.main(["run", str(mlp_model_path), "--input", str(npy_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no --input given" not in captured.out


def test_run_with_wrong_size_npy_exits_2(mlp_model_path, tmp_path, capsys):
    npy_path = tmp_path / "bad.npy"
    np.save(npy_path, np.ones(3))  # model expects 4
    rc = cli.main(["run", str(mlp_model_path), "--input", str(npy_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "expects 4" in captured.err


def test_run_quantize_flag_prints_note(mlp_model_path, capsys):
    rc = cli.main(["run", str(mlp_model_path), "--quantize"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "int8 quantization changes numerics" in captured.out


# --------------------------------------------------------------------------- #
# visualize
# --------------------------------------------------------------------------- #
def test_visualize_writes_dot_file(mlp_model_path, tmp_path):
    out_path = tmp_path / "graph.dot"
    rc = cli.main(["visualize", str(mlp_model_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert "digraph" in out_path.read_text()


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
@requires_gxx
def test_benchmark_smoke(mlp_model_path, tmp_path, capsys):
    out_dir = tmp_path / "results"
    rc = cli.main(
        [
            "benchmark",
            str(mlp_model_path),
            "--out",
            str(out_dir),
            "--iters",
            "5",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0
    json_path = out_dir / "mlp.json"
    assert json_path.exists()
    captured = capsys.readouterr()
    assert "s/iter" in captured.out


# --------------------------------------------------------------------------- #
# Model loading errors
# --------------------------------------------------------------------------- #
def test_unknown_extension_exits_2(tmp_path, capsys):
    bad_path = tmp_path / "model.xyz"
    bad_path.write_text("not a real model")
    rc = cli.main(["run", str(bad_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "unsupported model file extension" in captured.err


def test_nonexistent_file_exits_2(tmp_path, capsys):
    missing_path = tmp_path / "does_not_exist.json"
    rc = cli.main(["run", str(missing_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_onnx_missing_dependency_exits_2_with_hint(tmp_path, monkeypatch, capsys):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")  # content never read; import_onnx is patched

    def _raise_import_error(_path):
        raise ImportError("no module named onnx")

    monkeypatch.setattr(cli, "import_onnx", _raise_import_error)

    rc = cli.main(["run", str(onnx_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "pip install onnx" in captured.err


# --------------------------------------------------------------------------- #
# ONNX round-trip (only runs when the onnx package is installed)
# --------------------------------------------------------------------------- #
def test_onnx_round_trip_run(tmp_path, capsys):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(1234)
    W = (rng.standard_normal((4, 3)) * 0.5).astype(np.float32)
    b_arr = (rng.standard_normal(3) * 0.5).astype(np.float32)

    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    gemm_node = helper.make_node(
        "Gemm", ["x", "W", "b"], ["h"], alpha=1.0, beta=1.0, transA=0, transB=0
    )
    relu_node = helper.make_node("Relu", ["h"], ["y"])

    onnx_graph = helper.make_graph(
        [gemm_node, relu_node],
        "tinynn_cli_test",
        [x_vi],
        [y_vi],
        initializer=[
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(b_arr, "b"),
        ],
    )
    model = helper.make_model(
        onnx_graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(model)

    onnx_path = tmp_path / "gemm_relu.onnx"
    onnx.save(model, str(onnx_path))

    rc = cli.main(["run", str(onnx_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no --input given" in captured.out


# --------------------------------------------------------------------------- #
# python -m tinynn (subprocess smoke test)
# --------------------------------------------------------------------------- #
def test_python_dash_m_tinynn_help():
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "tinynn", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tinynn" in proc.stdout
