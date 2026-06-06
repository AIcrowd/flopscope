"""Integration regression tests for the BudgetContext timing split (Option-3 semantics).

Option-3: backend = pure server numpy kernel; overhead = all flopscope machinery
(client dispatch + wire + server marshaling, including .tolist() and implicit
fetches such as repr/bool); residual = participant's own Python only.

The production bug: the client proxy reported flopscope_backend_time /
flopscope_overhead_time / residual_wall_time as 0 for every MLP. These tests run
a real FlopscopeServer in a subprocess and assert the split is non-zero,
decomposes wall, and correctly isolates participant Python in residual.
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


def test_timing_nonzero_and_identity():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((64, 64))
        for _ in range(5):
            a = fl.dot(a, a)

    assert ctx.wall_time_s > 0
    assert ctx.flopscope_backend_time > 0  # pure kernel
    assert ctx.flopscope_overhead_time > 0  # dispatch + wire
    assert ctx.residual_wall_time >= 0
    total = (
        ctx.flopscope_backend_time
        + ctx.flopscope_overhead_time
        + ctx.residual_wall_time
    )
    assert abs(ctx.wall_time_s - total) < 0.05


def test_tolist_is_overhead_not_residual():
    """Heavy fetch+reconstruct with NO participant Python ⇒ residual ≈ 0."""
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**13) as ctx:
        a = fl.ones((128, 128))
        for _ in range(10):
            _ = a.tolist()

    assert ctx.residual_wall_time < 0.01  # reconstruction is overhead now
    assert ctx.flopscope_overhead_time > ctx.residual_wall_time


def test_implicit_fetch_is_overhead():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((16, 16))
        for _ in range(5):
            _ = repr(a)

    assert ctx.residual_wall_time < 0.01
    assert ctx.flopscope_overhead_time > ctx.residual_wall_time


def test_residual_is_only_python():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((8, 8))
        _ = fl.dot(a, a)
        time.sleep(0.2)  # the only non-flopscope wall
        _ = fl.dot(a, a)

    assert ctx.residual_wall_time >= 0.15
    assert ctx.flopscope_backend_time < 0.15
    assert ctx.flopscope_overhead_time < 0.15
    assert ctx.flopscope_overhead_time > 0


def test_worker_tolist_not_billed():
    """Reproduce the worker's preds.tolist() — must land in overhead, not residual."""
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**13) as ctx:
        preds = fl.ones((256, 256))
        for _ in range(3):
            preds = fl.dot(preds, preds)
        _ = preds.tolist()  # harness serialization of participant output

    assert ctx.residual_wall_time < 0.05


def test_flops_cost_query_round_trip_is_overhead_not_residual():
    """flops.einsum_cost / flops.svd_cost round-trip to the server; that round-trip
    is framework work and must land in overhead, never the participant's billed
    residual bucket.

    Regression: these two helpers previously issued a bare send_recv outside any
    dispatch span, so every advisory cost query leaked into residual. (The server
    rejects the op, but the request still goes out and back, and the @timed_dispatch
    span counts that round-trip regardless of the raise.)
    """
    import contextlib

    import flopscope as fl
    from flopscope import flops
    from flopscope.errors import FlopscopeError

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        for _ in range(30):
            with contextlib.suppress(FlopscopeError):
                flops.einsum_cost("ij,jk->ik", [(64, 64), (64, 64)])
            with contextlib.suppress(FlopscopeError):
                flops.svd_cost(64, 64)

    # No participant compute ran inside the context, so residual must stay tiny;
    # the 60 cost-query round-trips are all overhead.
    assert ctx.flopscope_overhead_time > 0
    assert ctx.residual_wall_time < 0.01
    assert ctx.flopscope_overhead_time > ctx.residual_wall_time


def test_backend_scales_with_compute():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**13) as small:
        s = fl.ones((8, 8))
        _ = fl.dot(s, s)
    with fl.BudgetContext(flop_budget=10**13) as big:
        b = fl.ones((512, 512))
        for _ in range(5):
            b = fl.dot(b, b)

    assert big.flopscope_backend_time > small.flopscope_backend_time


def test_getattr_end_to_end():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((32, 32))
        _ = fl.dot(a, a)

    # exactly how whestbench-evaluator reads them
    assert float(getattr(ctx, "flopscope_backend_time", 0.0)) > 0
    assert float(getattr(ctx, "wall_time_s", 0.0) or 0.0) > 0


def test_every_op_family_increments_dispatch():
    import flopscope._dispatch as d

    import flopscope as fl

    def _grew(fn):
        before = d.total_dispatch_ns()
        result = fn()
        assert d.total_dispatch_ns() > before, (
            "op did not increment dispatch accumulator"
        )
        return result

    with fl.BudgetContext(flop_budget=10**13):
        a = _grew(lambda: fl.ones((8, 8)))  # module-level proxy
        b = _grew(lambda: a + a)  # _dispatch_op (arithmetic)
        c = _grew(lambda: fl.dot(b, b))  # module-level proxy
        _grew(lambda: c[0])  # __getitem__
        _grew(lambda: c.tolist())  # _fetch_data + reconstruct
        _grew(lambda: repr(c))  # implicit fetch (repr→tolist→_fetch_data)
        g = _grew(lambda: fl.random.default_rng(0))  # random submodule proxy
        _grew(lambda: g.standard_normal((4, 4)))  # RemoteGenerator._call
        _grew(lambda: fl.linalg.qr(fl.ones((4, 4))))  # linalg submodule proxy
        _grew(lambda: fl.stats.norm.pdf(fl.ones((4,))))  # stats _DistributionProxy.pdf
        _grew(lambda: fl.array([1.0, 2.0, 3.0]))  # array() special-case
        _grew(lambda: fl.einsum("ij,jk->ik", a, b))  # einsum() special-case
        # fl.flops.einsum_cost / fl.flops.svd_cost are @timed_dispatch but send
        # "flops.einsum_cost" / "flops.svd_cost" to the server — neither op is
        # in the server whitelist (flopscope._registry.REGISTRY has no "flops.*"
        # keys; cost helpers live under flopscope.accounting, not proxied).
        # Calling them always raises FlopscopeServerError; skip rather than
        # assert a call that is guaranteed to fail. Their overhead-not-residual
        # behavior is covered by
        # test_flops_cost_query_round_trip_is_overhead_not_residual.
        _grew(lambda: fl.stats.norm.cdf(fl.ones((4,))))  # stats.norm.cdf
        _grew(lambda: fl.stats.norm.ppf(fl.array([0.25, 0.5, 0.75])))  # stats.norm.ppf


def test_empty_context_identity():
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**9) as ctx:
        pass

    assert ctx.wall_time_s is not None and ctx.wall_time_s > 0
    assert ctx.flopscope_backend_time >= 0
    assert ctx.flopscope_overhead_time >= 0  # budget_open/close round-trips
    assert ctx.residual_wall_time >= 0
    total = (
        ctx.flopscope_backend_time
        + ctx.flopscope_overhead_time
        + ctx.residual_wall_time
    )
    assert abs(ctx.wall_time_s - total) < 0.05


def test_flops_used_refreshed_on_close_without_summary():
    """A plain `with` block must report the server's authoritative flops_used on
    exit — no `bctx.summary()` workaround required.

    Regression: __exit__ called `_update_budget(response)` on the raw budget_close
    response (top-level), but the server nests the count at
    `result.budget_breakdown.flops_used`, so the cache stayed at 0. The downstream
    worker papered over this with a best-effort `bctx.summary()` before close.
    """
    import flopscope as fl

    with fl.BudgetContext(flop_budget=10**12) as ctx:
        a = fl.ones((128, 128))
        for _ in range(5):
            a = fl.dot(a, a)  # ~10.5M FLOPs counted server-side

    # ~2M FLOPs per 128³ matmul × 5 ⇒ ~10M server-side; was exactly 0 pre-fix.
    assert ctx.flops_used > 1_000_000, "flops_used not refreshed from server on close"
