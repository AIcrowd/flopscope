"""End-to-end integration tests for all flopscope operation categories.

Starts a real server subprocess and tests pointwise ops, reductions, linalg,
random, stats, einsum, and error propagation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

import flopscope as flops

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

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15560"

_SERVER_SCRIPT = f"""
import sys, os
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


def _raw_server_control(op: str) -> dict:
    """Send a test-only lifecycle request outside the participant client."""
    import msgpack
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5_000)
    socket.setsockopt(zmq.SNDTIMEO, 5_000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(_SERVER_URL)
        socket.send(
            msgpack.packb(
                {"op": op, "kwargs": {}},
                use_bin_type=True,
            )
        )
        return msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close()


def _recover_remote_budget_context() -> None:
    """Close a leaked test session, tolerating an already-clean server."""
    response = _raw_server_control("budget_close")
    if response.get("status") == "ok":
        return
    assert response.get("status") == "error", response
    assert response.get("error_type") == "NoBudgetContextError", response


def _clear_private_client_budget_state() -> None:
    """Make local test proxies match the raw server-side close."""
    from flopscope._connection import reset_connection

    import flopscope._budget as budget_module

    contexts = {
        context
        for context in (
            budget_module._active_context,
            budget_module._global_default,
        )
        if context is not None
    }
    for context in contexts:
        context._is_open = False
        context._previous_context = None
    budget_module._active_context = None
    budget_module._global_default = None
    reset_connection()


def _reset_remote_summary_epoch() -> None:
    """Reset the test epoch without exposing a participant API."""
    response = _raw_server_control("budget_summary_reset")
    assert response.get("status") == "ok", response


def _isolate_client_and_remote_summary_state() -> None:
    try:
        _recover_remote_budget_context()
    finally:
        _clear_private_client_budget_state()
    _reset_remote_summary_epoch()


@pytest.fixture(autouse=True)
def _reset_client_and_remote_epoch():
    _isolate_client_and_remote_summary_state()
    yield
    _isolate_client_and_remote_summary_state()


# ===========================================================================
# Category 1: Pointwise operations
# ===========================================================================


class TestPointwise:
    def test_add_lists(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1, 2, 3])
            b = we.array([4, 5, 6])
            result = we.add(a, b)
            assert result.tolist() == [5, 7, 9]

    def test_add_floats(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            b = we.array([4.0, 5.0, 6.0])
            result = we.add(a, b)
            assert result.tolist() == [5.0, 7.0, 9.0]

    def test_subtract(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([10.0, 20.0, 30.0])
            b = we.array([1.0, 2.0, 3.0])
            result = we.subtract(a, b)
            assert result.tolist() == [9.0, 18.0, 27.0]

    def test_multiply(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([2.0, 3.0, 4.0])
            b = we.array([5.0, 6.0, 7.0])
            result = we.multiply(a, b)
            assert result.tolist() == [10.0, 18.0, 28.0]

    def test_exp(self):
        import math

        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([0.0, 1.0])
            result = we.exp(a)
            vals = result.tolist()
            assert abs(vals[0] - 1.0) < 1e-10
            assert abs(vals[1] - math.e) < 1e-10


# ===========================================================================
# Category 2: Reductions
# ===========================================================================


class TestReduction:
    def test_sum_1d(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            result = we.sum(a)
            assert float(result) == 6.0

    def test_sum_returns_scalar(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1, 2, 3])
            result = we.sum(a)
            assert float(result) == 6.0

    def test_sum_2d_axis0(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([[1.0, 2.0], [3.0, 4.0]])
            result = we.sum(a, axis=0)
            assert result.tolist() == [4.0, 6.0]

    def test_sum_2d_axis1(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([[1.0, 2.0], [3.0, 4.0]])
            result = we.sum(a, axis=1)
            assert result.tolist() == [3.0, 7.0]

    def test_mean(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0, 4.0])
            result = we.mean(a)
            assert float(result) == 2.5


# ===========================================================================
# Category 3: Linear algebra (linalg)
# ===========================================================================


class TestLinalg:
    def test_svd_diagonal(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 0.0], [0.0, 2.0]])
            U, S, Vh = we.linalg.svd(A)
            sv = sorted(S.tolist(), reverse=True)
            assert abs(sv[0] - 2.0) < 1e-10
            assert abs(sv[1] - 1.0) < 1e-10

    def test_svd_shapes(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 0.0], [0.0, 2.0]])
            U, S, Vh = we.linalg.svd(A)
            assert S.shape == (2,)
            assert U.shape == (2, 2)
            assert Vh.shape == (2, 2)

    def test_norm(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            # L2 norm of [3, 4] = 5
            a = we.array([3.0, 4.0])
            result = we.linalg.norm(a)
            assert abs(float(result) - 5.0) < 1e-10

    def test_dot_matmul(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            B = we.array([[5.0, 6.0], [7.0, 8.0]])
            C = we.linalg.matmul(A, B)
            assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]


# ===========================================================================
# Category 4: Random
# ===========================================================================


class TestRandom:
    def test_normal_shape(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            result = we.random.normal(size=[100])
            assert result.shape == (100,)

    def test_normal_values_are_floats(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            result = we.random.normal(size=[10])
            vals = result.tolist()
            assert len(vals) == 10
            assert all(isinstance(v, float) for v in vals)

    def test_uniform_shape(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            result = we.random.uniform(size=[50])
            assert result.shape == (50,)

    def test_uniform_range(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            result = we.random.uniform(size=[100])
            vals = result.tolist()
            assert all(0.0 <= v <= 1.0 for v in vals)


# ===========================================================================
# Category 5: Stats distributions
# ===========================================================================


class TestStats:
    def test_norm_pdf_at_zero(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([0.0])
            result = we.stats.norm.pdf(x)
            val = float(result)
            # PDF of standard normal at 0 = 1/sqrt(2*pi) ≈ 0.3989
            assert abs(val - 0.3989422804014327) < 1e-6

    def test_norm_cdf_at_zero(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([0.0])
            result = we.stats.norm.cdf(x)
            val = float(result)
            # CDF of standard normal at 0 = 0.5
            assert abs(val - 0.5) < 1e-10

    def test_expon_pdf_at_zero(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([0.0])
            result = we.stats.expon.pdf(x)
            val = float(result)
            # PDF of exponential(rate=1) at 0 = 1.0
            assert abs(val - 1.0) < 1e-10

    def test_norm_pdf_shape(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([0.0, 1.0, -1.0])
            result = we.stats.norm.pdf(x)
            assert result.shape == (3,)

    def test_norm_cdf_monotone(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([-1.0, 0.0, 1.0])
            result = we.stats.norm.cdf(x)
            vals = result.tolist()
            assert vals[0] < vals[1] < vals[2]


# ===========================================================================
# Category 6: Einsum
# ===========================================================================


class TestEinsum:
    def test_matmul_2x2(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            B = we.array([[5.0, 6.0], [7.0, 8.0]])
            C = we.einsum("ij,jk->ik", A, B)
            # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
            assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]

    def test_matmul_identity(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            eye = we.array([[1.0, 0.0], [0.0, 1.0]])
            C = we.einsum("ij,jk->ik", A, eye)
            assert C.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_dot_product(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            b = we.array([4.0, 5.0, 6.0])
            result = we.einsum("i,i->", a, b)
            # 1*4 + 2*5 + 3*6 = 32
            assert float(result) == 32.0

    def test_outer_product(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0])
            b = we.array([3.0, 4.0])
            result = we.einsum("i,j->ij", a, b)
            assert result.shape == (2, 2)
            assert result.tolist() == [[3.0, 4.0], [6.0, 8.0]]

    def test_trace(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            trace = we.einsum("ii->", A)
            assert float(trace) == 5.0


# ===========================================================================
# Category 7: Error propagation
# ===========================================================================


class TestErrorPropagation:
    def test_budget_exhausted_on_matmul(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            B = we.array([[5.0, 6.0], [7.0, 8.0]])
            with pytest.raises(we.BudgetExhaustedError):
                we.einsum("ij,jk->ik", A, B)

    def test_budget_exhausted_error_type(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1):
            a = we.array([1.0, 2.0, 3.0])
            b = we.array([4.0, 5.0, 6.0])
            with pytest.raises(we.BudgetExhaustedError):
                # Keep calling ops until budget is exhausted
                for _ in range(100):
                    we.add(a, b)

    def test_no_budget_context_raises(self):
        with pytest.raises((flops.NoBudgetContextError, flops.FlopscopeServerError)):
            # Use the top-level alias `flops.array` rather than `fnp.array`:
            # the client's ``flopscope.numpy`` JAX-style restructure is not
            # yet complete (the tracked ``flopscope/numpy/__init__.py`` is
            # absent), so `fnp.array` raises AttributeError before reaching
            # the budget-context check this test exercises.
            flops.array([1.0, 2.0, 3.0])

    def test_budget_context_isolates_errors(self):
        """Verify a new context works after a previous one exhausted budget."""
        import flopscope as we

        # First context: exhaust budget
        try:
            with we.BudgetContext(flop_budget=1):
                a = we.array([1.0] * 1000)
                we.exp(a)
        except we.BudgetExhaustedError:
            pass

        # Second context: should work normally
        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0])
            b = we.array([4.0, 5.0, 6.0])
            result = we.add(a, b)
            assert result.tolist() == [5.0, 7.0, 9.0]


# ===========================================================================
# Category 8: Version handshake (added in v0.3.0)
# ===========================================================================


class TestVersionHandshake:
    def test_handshake_completes_transparently_on_first_op(self):
        """A fresh Connection performs the version handshake on first send_recv."""
        from flopscope._connection import get_connection

        import flopscope as we

        conn = get_connection()
        assert conn._handshake_done is False

        with we.BudgetContext(flop_budget=1_000):
            a = we.array([1.0, 2.0, 3.0])
            b = we.array([4.0, 5.0, 6.0])
            we.add(a, b)

        assert conn._handshake_done is True

    def test_handshake_rejects_version_mismatch(self):
        """A client whose __version__ doesn't match the server fails fast."""
        from flopscope._connection import reset_connection

        import flopscope
        import flopscope as we

        reset_connection()
        original = flopscope.__version__
        flopscope.__version__ = "9.99.99"
        try:
            with pytest.raises(ConnectionError) as excinfo:
                with we.BudgetContext(flop_budget=1_000):
                    we.array([1.0, 2.0, 3.0])
            msg = str(excinfo.value)
            assert "9.99.99" in msg
            assert original.split("+", 1)[0] in msg
        finally:
            flopscope.__version__ = original
            reset_connection()


# ===========================================================================
# Category 9: Authoritative budget summaries
# ===========================================================================


STABLE_KEYS = {
    "flop_budget",
    "flops_used",
    "flops_remaining",
    "operations",
}


def _stable(summary):
    return {key: summary[key] for key in STABLE_KEYS}


def _assert_timing_contract(summary):
    wall = summary["wall_time_s"]
    backend = summary["flopscope_backend_time_s"]
    overhead = summary["flopscope_overhead_time_s"]
    residual = summary["residual_wall_time_s"]
    assert backend >= 0
    assert overhead >= 0
    if wall is None:
        assert residual is None
    else:
        assert residual >= 0
        assert wall == pytest.approx(backend + overhead + residual, abs=1e-9)


def _charge_add(length: int = 1) -> None:
    left = flops.array([1.0] * length)
    right = flops.array([2.0] * length)
    flops.add(left, right)


def test_real_wire_session_accumulates_two_closed_contexts() -> None:
    with flops.BudgetContext(100, namespace="first") as first:
        _charge_add(1)
    with flops.BudgetContext(100, namespace="second") as second:
        _charge_add(2)
    summary = flops.budget_summary_dict(by_namespace=True)
    assert summary["flop_budget"] == 200
    assert summary["flops_used"] == first.flops_used + second.flops_used
    assert summary["operations"]["add"]["calls"] == 2
    assert set(summary["by_namespace"]) == {"first", "second"}
    _assert_timing_contract(summary)


def test_real_wire_session_includes_active_context() -> None:
    with flops.BudgetContext(100) as closed:
        _charge_add(1)
    with flops.BudgetContext(100) as active:
        _charge_add(2)
        active_summary = active.summary_dict()
        session_summary = flops.budget_summary_dict()
        assert session_summary["flops_used"] == (
            closed.flops_used + active_summary["flops_used"]
        )
        _assert_timing_contract(active_summary)
        _assert_timing_contract(session_summary)


def test_real_wire_active_context_excludes_closed_history() -> None:
    with flops.BudgetContext(100) as closed:
        _charge_add(1)
    with flops.BudgetContext(100) as active:
        _charge_add(2)
        active_summary = active.summary_dict()
        session_summary = flops.budget_summary_dict()
        assert active_summary["flops_used"] == active.flops_used
        assert session_summary["flops_used"] > active_summary["flops_used"]
        assert closed.flops_used == (
            session_summary["flops_used"] - active_summary["flops_used"]
        )


def test_real_wire_namespace_flag_controls_only_optional_key() -> None:
    with flops.BudgetContext(100, namespace="phase"):
        _charge_add(1)
    flat = flops.budget_summary_dict(False)
    namespaced = flops.budget_summary_dict(True)
    assert "by_namespace" not in flat
    assert set(namespaced) == set(flat) | {"by_namespace"}
    assert _stable(namespaced) == _stable(flat)
    phase = namespaced["by_namespace"]["phase"]
    assert phase["flops_used"] == namespaced["flops_used"]
    assert phase["operations"] == namespaced["operations"]
    assert phase["calls"] == sum(
        operation["calls"] for operation in namespaced["operations"].values()
    )


def test_real_wire_closed_context_cache_survives_later_session() -> None:
    with flops.BudgetContext(100, namespace="first") as first:
        _charge_add(1)
    cached = first.summary_dict(True)
    with flops.BudgetContext(100, namespace="second"):
        _charge_add(3)
        assert first.summary_dict(True) == cached
    assert first.summary_dict(True) == cached


def test_real_wire_summary_after_budget_close() -> None:
    with flops.BudgetContext(100, namespace="only") as context:
        _charge_add(2)
        live = context.summary_dict(True)
    closed = context.summary_dict(True)
    session = flops.budget_summary_dict(True)
    assert _stable(closed) == _stable(live)
    assert _stable(session) == _stable(closed)
    assert session["by_namespace"] == closed["by_namespace"]
    _assert_timing_contract(closed)
    _assert_timing_contract(session)
    assert closed["wall_time_s"] == context.wall_time_s
    assert closed["flopscope_backend_time_s"] == context.flopscope_backend_time_s
    assert closed["flopscope_overhead_time_s"] == context.flopscope_overhead_time_s
    assert closed["residual_wall_time_s"] == context.residual_wall_time_s
    # The open/op/close RPC spans make transport visibly part of the
    # client-owned context overhead; this is not a server timing claim.
    assert context.flopscope_overhead_time_s > 0


def test_fixture_recovers_a_deliberately_failed_close(monkeypatch) -> None:
    """Recovery is self-contained and clears local state on transport failure."""
    import sys

    from flopscope._connection import get_connection

    import flopscope._budget as budget_module

    context = flops.BudgetContext(100, namespace="deliberately-leaked")
    context.__enter__()
    _charge_add(1)
    connection = get_connection()

    def fail_close(_request: bytes):
        raise TimeoutError("deliberate close failure")

    monkeypatch.setattr(connection, "send_recv", fail_close)
    with pytest.raises(TimeoutError, match="deliberate close failure"):
        context.__exit__(None, None, None)
    assert context._is_open is True

    def fail_raw_close(_op: str):
        raise TimeoutError("deliberate raw-close transport failure")

    with monkeypatch.context() as raw_close_failure:
        raw_close_failure.setattr(
            sys.modules[__name__], "_raw_server_control", fail_raw_close
        )
        with pytest.raises(
            TimeoutError, match="deliberate raw-close transport failure"
        ):
            _isolate_client_and_remote_summary_state()
    assert context._is_open is False
    assert budget_module._active_context is None
    assert budget_module._global_default is None

    # Once transport is available, the same fixture helper recovers the still-open
    # server context, resets its epoch, and permits a fresh client context.
    _isolate_client_and_remote_summary_state()
    with flops.BudgetContext(100, namespace="after-recovery") as fresh:
        _charge_add(1)
    summary = flops.budget_summary_dict(True)
    assert summary["flops_used"] == fresh.flops_used
    assert set(summary["by_namespace"]) == {"after-recovery"}
