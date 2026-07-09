"""Tier 5 (CUDA backend) test structure.

There is deliberately NO CUDA backend in the repo yet: the development
machine (arm64 macOS) has no nvcc/nvidia-smi, so a GPU backend could not be
compiled, executed, or verified against the interpreter oracle. Rather than
ship unverifiable kernels, Tier 5 is marked blocked-on-environment.

This module keeps that status honest and machine-checked:

* On machines WITHOUT a CUDA toolchain (like the current dev machine), the
  test below skips cleanly with an explanatory reason.
* On machines WITH a CUDA toolchain, it FAILS loudly, signalling that the
  environment is ready and the Tier 5 backend should now be implemented
  (see ROADMAP.md) -- it must never silently pass while no backend exists.

When the backend lands (e.g. ``tinynn/cuda.py`` with ``compile_graph_cuda``),
replace this sentinel with real compile-and-compare-to-interpreter tests.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def cuda_toolchain_available() -> bool:
    """Return True only if ``nvcc`` exists and runs."""
    if shutil.which("nvcc") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


@pytest.mark.skipif(
    not cuda_toolchain_available(),
    reason="CUDA toolchain (nvcc) not available: Tier 5 GPU backend is "
    "blocked on environment, not started (see ROADMAP.md)",
)
def test_cuda_backend_not_yet_implemented():
    pytest.fail(
        "A CUDA toolchain is available on this machine, but TinyNN has no "
        "CUDA backend yet. Implement Tier 5 (see ROADMAP.md) and replace "
        "this sentinel with real GPU-vs-interpreter correctness tests."
    )
