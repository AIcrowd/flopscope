"""Catching an operation's failure must not hide what it already cost.

The kernel bills an operation before the server tries to pack its result, so an
operation can be charged and then fail because the result cannot be delivered.
The FLOPs are gone. A participant who wraps an op in ``try``/``except`` and then
reads ``ctx.flops_used`` to decide what to run next must see the charge; reading
a value from before it would have them plan against a budget they no longer
have.

The server is started here with a deliberately tiny ``FLOPSCOPE_MAX_ARRAY_BYTES``
so a modest broadcast multiply is billed and then refused: both operands fit
well under the limit and the array they broadcast to does not, which is the
shape of the bug without needing a genuinely enormous array.
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

_SERVER_URL = "tcp://127.0.0.1:15561"

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
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    env = dict(os.environ, FLOPSCOPE_MAX_ARRAY_BYTES="1024")
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
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


def test_a_caught_failure_leaves_flops_used_reflecting_the_charge():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        row = fl.ones((1, 64))  # 512 bytes each: both store fine
        col = fl.ones((64, 1))
        before = ctx.flops_used

        with pytest.raises(ValueError):
            # Broadcasts to 64x64 = 32768 bytes: the multiply is billed by
            # the kernel, and only then found to be unsendable.
            fl.multiply(row, col)

        assert ctx.flops_used > before, (
            "the failed op was still billed; flops_used must show it"
        )

        cached = ctx.flops_used
        ctx.summary()  # explicit refresh straight from the server
        assert ctx.flops_used == cached, (
            "the cache after a caught failure must already equal the "
            "server's own figure, not merely be closer to it"
        )


def test_flops_used_never_goes_backwards_across_a_caught_failure():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        row = fl.ones((1, 64))
        col = fl.ones((64, 1))
        fl.dot(fl.ones(64), fl.ones(64))  # returns a scalar: billed and delivered
        after_success = ctx.flops_used
        assert after_success > 0

        with pytest.raises(ValueError):
            fl.multiply(row, col)

        assert ctx.flops_used >= after_success
