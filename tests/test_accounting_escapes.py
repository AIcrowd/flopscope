"""Regression tests for three accounting escapes (see PR description).

Each test asserts an INVARIANT rather than a magic number, so they keep their
value as the cost model evolves:

1. ``ufunc.at`` must bill per element WRITTEN, not per index supplied, so it
   agrees with the equivalent elementwise op on the same element count.
2. Every wrapper that invokes participant code must route it through
   ``_call_user_code``, so the callback's wall time books to residual rather
   than to the free ``flopscope_overhead_time_s`` bucket.

NOT included, deliberately: single-operand ``einsum`` ('ij->ij', 'ij->ji')
billing 0. That looks like an under-bill but is INTENTIONAL and locked by
``test_complex_einsum_transpose_charges_zero`` and
``test_empty_symmetric_contraction_does_not_refund_budget``; clamping it to 1
breaks both. Left alone.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp


def _billed(fn):
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        before = ctx.flops_used
        fn()
        return ctx.flops_used - before


# --------------------------------------------------------------------------
# 1. ufunc.at bills the cells it writes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 4096), (1, 64, 64), (1, 16, 16, 16)])
def test_ufunc_at_bills_written_cells_not_index_count(shape):
    """A length-1 index into an N-d array writes prod(shape[1:]) cells."""
    dst = fnp.asarray(np.zeros(shape, np.float32))
    vals = fnp.asarray(np.random.randn(*shape[1:]).astype(np.float32))
    idx = np.array([0], np.intp)

    at_cost = _billed(lambda: np.add.at(dst, idx, vals[None]))
    written = int(np.prod(shape[1:]))
    equivalent = _billed(lambda: fnp.add(fnp.asarray(np.zeros(shape[1:], np.float32)), vals))

    assert at_cost == written, f"add.at billed {at_cost}, wrote {written} cells"
    assert at_cost == equivalent, "add.at must agree with the equivalent fnp.add"


def test_ufunc_at_scatter_bills_every_accumulate():
    """Repeated fancy indices each perform an operation; tuple form included."""
    n_idx = 50_000
    dst = fnp.asarray(np.zeros((1000, 1), np.float32))
    vals = fnp.asarray(np.random.randn(n_idx).astype(np.float32))
    rows = fnp.asarray(np.random.randint(0, 1000, n_idx).astype(np.intp))
    cols = fnp.asarray(np.zeros(n_idx, np.intp))

    tuple_cost = _billed(lambda: np.add.at(dst, (rows, cols), vals))
    assert tuple_cost == n_idx, (
        f"tuple-index scatter billed {tuple_cost} for {n_idx} accumulates "
        "(the old `a.size` upper bound is NOT conservative for repeated indices)"
    )

    list_cost = _billed(
        lambda: np.add.at(dst, ([0] * 1000, [0] * 1000), fnp.asarray(np.zeros(1000, np.float32)))
    )
    assert list_cost == 1000, f"list-index scatter billed {list_cost} for 1000 accumulates"


def test_ufunc_at_matmul_route_is_not_cheaper_than_matmul():
    """The composed at-matmul must not undercut the honest contraction."""
    k = 32
    X = fnp.asarray(np.random.randn(k, k).astype(np.float32))
    W = fnp.asarray(np.random.randn(k, k).astype(np.float32))
    honest = _billed(lambda: fnp.matmul(X, W))

    one = np.array([0], np.intp)

    def at_route():
        D = fnp.asarray(np.zeros((1, k, k, k), np.float32))
        np.add.at(D, one, W[None, None, :, :])
        np.multiply.at(D, one, X[None, :, :, None])
        V = D[0]
        m = k
        while m > 1:
            h = m // 2
            np.add.at(V[:, :h, :][None], one, V[:, h : 2 * h, :][None])
            V = V[:, :h, :]
            m = h

    assert _billed(at_route) >= honest, "at-composed matmul must not be cheaper than matmul"


# --------------------------------------------------------------------------
# 2. participant callbacks bill to residual, not to free overhead
# --------------------------------------------------------------------------

CALLBACK_WRAPPERS = [
    ("mask_indices", lambda cb: fnp.mask_indices(4, cb)),
    ("fromfunction", lambda cb: fnp.fromfunction(cb, (4, 4))),
    ("apply_along_axis", lambda cb: fnp.apply_along_axis(
        cb, 0, fnp.asarray(np.zeros((4, 4), np.float32)))),
]


@pytest.mark.parametrize("name,invoke", CALLBACK_WRAPPERS, ids=[n for n, _ in CALLBACK_WRAPPERS])
def test_callback_time_books_to_residual(name, invoke):
    """User-code wall time must not land in the free overhead bucket."""
    sleep_s = 0.20

    def callback(*args, **kwargs):
        time.sleep(sleep_s)
        if name == "mask_indices":
            return np.zeros((4, 4), bool)
        if name == "apply_along_axis":
            return np.float32(0.0)
        return np.zeros(np.shape(args[0]), np.float32) if args else np.zeros((4, 4), np.float32)

    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        invoke(callback)
    summary = ctx.summary_dict()
    residual = float(summary.get("residual_wall_time_s") or 0.0)
    overhead = float(summary.get("flopscope_overhead_time_s") or 0.0)

    assert residual >= 0.8 * sleep_s, (
        f"{name}: callback slept {sleep_s}s but only {residual:.3f}s billed as residual "
        f"({overhead:.3f}s went to the free overhead bucket)"
    )
