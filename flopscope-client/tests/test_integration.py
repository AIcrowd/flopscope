"""End-to-end integration tests: real client talking to a real server.

Starts a FlopscopeServer in a subprocess, points the client at it,
and exercises the full request/response chain.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time

import pytest

import flopscope as flops
import flopscope.numpy as fnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
# Prefer the server's own venv (which has msgpack/pyzmq) for the server subprocess;
# fall back to the worktree root venv if it doesn't exist.
_SERVER_VENV_PYTHON = os.path.join(
    _WORKTREE, "flopscope-server", ".venv", "bin", "python"
)
_ROOT_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")
_VENV_PYTHON = (
    _SERVER_VENV_PYTHON if os.path.exists(_SERVER_VENV_PYTHON) else _ROOT_VENV_PYTHON
)

# Ensure the CLIENT flopscope is importable in this process
for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture — runs in a subprocess so it gets the REAL flopscope
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15556"

# Inline server launcher script
_SERVER_SCRIPT = f"""
import sys, os
# The server needs the REAL flopscope (src/) and the server package
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})

from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    """Start a real FlopscopeServer in a subprocess for the entire test session."""
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL

    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for the server to be ready
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    # Give the socket a moment to bind
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the client connection between tests."""
    from flopscope._connection import reset_connection

    reset_connection()
    yield
    reset_connection()


# ---------------------------------------------------------------------------
# TestBasicOps
# ---------------------------------------------------------------------------


class TestBasicOps:
    def test_zeros_and_fetch(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            z = we.zeros((3, 4))
            assert z.shape == (3, 4)
            assert z.dtype == "float64"
            values = z.tolist()
            assert len(values) == 3
            assert len(values[0]) == 4
            assert all(v == 0.0 for row in values for v in row)

    def test_ones(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            o = we.ones((5,))
            values = o.tolist()
            assert values == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_array_from_list(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            assert a.shape == (3,)
            assert a.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TestCountedOps
# ---------------------------------------------------------------------------


class TestCountedOps:
    def test_exp(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000) as ctx:
            o = we.ones((3,))
            result = we.exp(o)
            values = result.tolist()
            for v in values:
                assert abs(v - math.e) < 1e-6
            # Check that flops were tracked
            ctx.summary()  # updates flops_used from server
            assert ctx.flops_used > 0

    def test_add(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            o = we.ones((4,))
            result = we.add(o, o)
            values = result.tolist()
            assert values == [2.0, 2.0, 2.0, 2.0]

    def test_sum_reduction(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            s = we.sum(a)
            assert float(s) == 6.0


# ---------------------------------------------------------------------------
# TestOperators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_add_operator(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0])
            y = we.array([3.0, 4.0])
            z = x + y
            assert z.tolist() == [4.0, 6.0]

    def test_mul_scalar(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            z = x * 2.0
            assert z.tolist() == [2.0, 4.0, 6.0]

    def test_neg(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, -2.0, 3.0])
            z = -x
            assert z.tolist() == [-1.0, 2.0, -3.0]

    def test_matmul(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            # 2x2 identity matrix
            ident = we.eye(2)
            v = we.array([5.0, 7.0])
            result = ident @ v
            assert result.tolist() == [5.0, 7.0]


# ---------------------------------------------------------------------------
# TestTransparency
# ---------------------------------------------------------------------------


class TestTransparency:
    def test_print_shows_values(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            r = repr(x)
            assert "array(" in r
            assert "1.0" in r

    def test_iteration(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            values = list(x)
            assert values == [10.0, 20.0, 30.0]

    def test_indexing(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            assert x[0] == 10.0
            assert x[2] == 30.0

    def test_len(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            assert len(x) == 2

    def test_float_conversion(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([42.0])
            assert float(x) == 42.0


# ---------------------------------------------------------------------------
# TestEinsum
# ---------------------------------------------------------------------------


class TestEinsum:
    def test_einsum_matvec(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            ident = we.eye(3)
            v = we.array([1.0, 2.0, 3.0])
            result = we.einsum("ij,j->i", ident, v)
            assert result.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TestErrors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_budget_exhausted(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1):
            a = we.ones((100,))
            with pytest.raises(we.BudgetExhaustedError):
                we.exp(a)

    def test_no_budget_context(self):
        # Operations outside a BudgetContext should raise NoBudgetContextError
        with pytest.raises((flops.NoBudgetContextError, flops.FlopscopeServerError)):
            fnp.ones((3,))

    def test_blacklisted_function(self):
        import flopscope as we

        # save/load/savez are supported pickle-free I/O now; savetxt (text I/O)
        # remains blacklisted.
        with pytest.raises(AttributeError, match="blacklisted"):
            _ = we.savetxt

    def test_unknown_function(self):
        import flopscope as we

        with pytest.raises(AttributeError):
            _ = we.nonexistent_function_xyz


# ---------------------------------------------------------------------------
# TestBudgetTracking
# ---------------------------------------------------------------------------


class TestBudgetTracking:
    def test_flops_tracked(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000) as ctx:
            a = we.ones((10,))
            _ = we.exp(a)
            ctx.summary()  # updates flops_used from server
            assert ctx.flops_used > 0

    def test_summary(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000) as ctx:
            _ = we.ones((10,))
            _ = we.exp(we.ones((10,)))
            s = ctx.summary()
            assert isinstance(s, str)
            assert "FLOPs" in s or "flop" in s.lower()
