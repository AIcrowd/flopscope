"""Integration regression tests for the BudgetContext timing split.

The production bug: the client proxy reported flopscope_backend_time /
flopscope_overhead_time / residual_wall_time as 0 for every MLP. These tests run
a real FlopscopeServer in a subprocess and assert the split is non-zero,
decomposes wall, and assigns participant Python to residual (the billed bucket).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SERVER_URL = "tcp://127.0.0.1:15558"

_SERVER_SCRIPT = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    reset_connection()
    yield
    reset_connection()


import flopscope as flops  # noqa: E402
import flopscope as fnp  # noqa: E402  (flopscope IS the numpy-like API; flopscope.numpy is not a submodule)


def test_timing_nonzero_after_real_ops():
    with flops.BudgetContext(flop_budget=10**12) as ctx:
        a = fnp.ones((64, 64))
        for _ in range(5):
            a = fnp.dot(a, a)

    assert ctx.wall_time_s > 0
    assert ctx.flopscope_backend_time > 0
    assert ctx.flopscope_overhead_time > 0
    assert ctx.residual_wall_time >= 0
    # the prod bug was all-zero — this must never recur
    assert (ctx.flopscope_backend_time + ctx.flopscope_overhead_time) > 0
    total = (
        ctx.flopscope_backend_time
        + ctx.flopscope_overhead_time
        + ctx.residual_wall_time
    )
    assert abs(ctx.wall_time_s - total) < 0.05


def test_residual_reflects_participant_python():
    with flops.BudgetContext(flop_budget=10**12) as ctx:
        a = fnp.ones((16, 16))
        _ = fnp.dot(a, a)
        time.sleep(0.2)              # participant Python — must land in residual
        _ = fnp.dot(a, a)

    assert ctx.residual_wall_time >= 0.15
    # the sleep must NOT be billed as backend or overhead
    assert ctx.flopscope_backend_time < 0.15
    assert ctx.flopscope_overhead_time < 0.15


def test_backend_scales_with_compute():
    with flops.BudgetContext(flop_budget=10**13) as small:
        s = fnp.ones((8, 8))
        _ = fnp.dot(s, s)

    with flops.BudgetContext(flop_budget=10**15) as big:
        b = fnp.ones((512, 512))
        for _ in range(5):
            b = fnp.dot(b, b)

    assert big.flopscope_backend_time > small.flopscope_backend_time


def test_transport_lands_in_overhead_not_residual():
    with flops.BudgetContext(flop_budget=10**13) as many:
        a = fnp.ones((4, 4))
        for _ in range(30):          # many round-trips, ~no participant Python
            a = fnp.dot(a, a)

    assert many.flopscope_overhead_time > many.residual_wall_time


def test_empty_context_identity():
    with flops.BudgetContext(flop_budget=10**9) as ctx:
        pass

    assert ctx.wall_time_s is not None and ctx.wall_time_s > 0
    assert ctx.flopscope_backend_time >= 0
    assert ctx.residual_wall_time >= 0
    total = (
        ctx.flopscope_backend_time
        + ctx.flopscope_overhead_time
        + ctx.residual_wall_time
    )
    assert abs(ctx.wall_time_s - total) < 0.05


def test_getattr_end_to_end():
    with flops.BudgetContext(flop_budget=10**12) as ctx:
        a = fnp.ones((32, 32))
        _ = fnp.dot(a, a)

    # exactly how whestbench-evaluator reads them
    assert float(getattr(ctx, "flopscope_backend_time", 0.0)) > 0
    assert float(getattr(ctx, "wall_time_s", 0.0) or 0.0) > 0
