"""``BudgetContext.flops_used`` must reflect the server mid-session.

The client caches ``flops_used`` locally and used to refresh it only at context
open, context close, and an explicit ``summary()`` call. A participant inspecting
``ctx.flops_used`` *between* operations — to log progress or make a budget-aware
branching decision — therefore saw a stale value (0 until the first refresh),
with no error and no warning. That silently wrong number is the bug.

Every compute-op response already carries the server's authoritative budget, so
the cache can be kept current with no extra round trip. These tests pin that
``flops_used`` and ``flops_remaining`` are live after each op, and that they
still agree exactly with an explicit ``summary()`` refresh.
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

_SERVER_URL = "tcp://127.0.0.1:15559"

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


def test_flops_used_is_live_after_an_op_without_summary():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        assert ctx.flops_used == 0  # nothing run yet
        a = fl.ones((64, 64))
        a = fl.dot(a, a)
        # No summary() call: reading the cache directly must reflect the op.
        assert ctx.flops_used > 0, (
            "flops_used is stale mid-session; the server's per-op budget was dropped"
        )


def test_flops_used_advances_monotonically_across_ops():
    import flopscope as fl

    seen = []
    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((64, 64))
        for _ in range(4):
            a = fl.dot(a, a)
            seen.append(ctx.flops_used)
    assert seen == sorted(seen), f"flops_used not monotonic mid-session: {seen}"
    assert seen[0] > 0
    assert seen[-1] > seen[0]


def test_live_cache_agrees_with_summary_refresh():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((64, 64))
        a = fl.dot(a, a)
        live = ctx.flops_used
        # summary() forces an explicit budget_status round trip; the live cache
        # must already equal it, not merely be close.
        ctx.summary()
        assert ctx.flops_used == live


def test_flops_remaining_tracks_flops_used():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((64, 64))
        a = fl.dot(a, a)
        assert ctx.flops_remaining == ctx._flop_budget - ctx.flops_used
        assert ctx.flops_remaining < ctx._flop_budget
