"""Real client+server regression for the array-handle leak (sub 310969).

A low server array-store cap + a Monte-Carlo-style loop that allocates far more
intermediates than the cap. With the free-on-GC fix the live count stays bounded
and the loop completes; without it the server raises MemoryError well before the
loop ends.
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
_SERVER_VENV_PYTHON = os.path.join(
    _WORKTREE, "flopscope-server", ".venv", "bin", "python"
)
_ROOT_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")
_VENV_PYTHON = (
    _SERVER_VENV_PYTHON if os.path.exists(_SERVER_VENV_PYTHON) else _ROOT_VENV_PYTHON
)

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SERVER_URL = "tcp://127.0.0.1:15561"  # distinct port from test_full_integration
_SERVER_SCRIPT = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="module", autouse=True)
def _low_cap_server():
    env = {
        **os.environ,
        "FLOPSCOPE_SERVER_URL": _SERVER_URL,
        "FLOPSCOPE_MAX_ARRAY_COUNT": "2000",
    }
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    line = proc.stdout.readline()
    if "SERVER_READY" not in line:
        # Surface the subprocess's stderr — a missing server venv / import error
        # otherwise shows up as an empty stdout line with no diagnostic.
        proc.kill()
        err = proc.stderr.read()
        pytest.fail(f"server failed to start: stdout={line!r} stderr={err!r}")
    time.sleep(0.3)  # allow the ZMQ socket to finish binding before clients connect
    yield proc
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    from flopscope._budget import _reset_global_default

    reset_connection()
    _reset_global_default()
    yield
    reset_connection()
    _reset_global_default()


def test_monte_carlo_loop_does_not_leak_handles():
    import flopscope as we
    import flopscope.numpy as fnp

    # 4000 iterations >> the 2000 cap. Each iter creates a transpose + matmul +
    # maximum (several handles) and drops the previous ones. Without free-on-GC,
    # live handles climb past 2000 and the server raises MemoryError; with it,
    # the live count stays at the working-set size and the loop completes.
    with we.BudgetContext(flop_budget=10**18):
        w = fnp.zeros((16, 16))
        acc = fnp.zeros(16)
        for _ in range(4000):
            acc = fnp.maximum(0.0, w.T @ acc)
        assert fnp.asarray(acc).tolist() == [0.0] * 16
