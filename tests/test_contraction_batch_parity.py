"""Contraction ops must count batch/broadcast axes exactly (== equivalent einsum)."""

from __future__ import annotations

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope import BudgetContext

W, N = 64, 512
rng = np.random.default_rng(0)

requires_matvec = pytest.mark.skipif(
    not hasattr(np, "matvec"), reason="requires numpy >= 2.2"
)
requires_vecmat = pytest.mark.skipif(
    not hasattr(np, "vecmat"), reason="requires numpy >= 2.2"
)


def _cost(fn, *args):
    with BudgetContext(flop_budget=int(1e20)) as bc:
        fn(*args)
    return bc.flops_used


def _einsum_cost(subs, *arrs):
    with BudgetContext(flop_budget=int(1e20)) as bc:
        fnp.einsum(subs, *arrs)
    return bc.flops_used


def arr(*shape):
    return fnp.asarray(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "op, subs, shapes",
    [
        pytest.param(
            fnp.vecmat, "...n,...nm->...m", [(N, W), (W, W)], marks=requires_vecmat
        ),
        pytest.param(
            fnp.vecmat, "...n,...nm->...m", [(N, W), (N, W, W)], marks=requires_vecmat
        ),
        pytest.param(
            fnp.matvec, "...mn,...n->...m", [(W, W), (N, W)], marks=requires_matvec
        ),
        pytest.param(
            fnp.matvec, "...mn,...n->...m", [(N, W, W), (N, W)], marks=requires_matvec
        ),
        (fnp.vecdot, "...n,...n->...", [(N, W), (N, W)]),
        (fnp.vecdot, "...n,...n->...", [(N, W), (W,)]),
        (fnp.matmul, "...ij,...jk->...ik", [(N, W, W), (N, W, W)]),
        (fnp.matmul, "...ij,...jk->...ik", [(N, W, W), (W, W)]),
    ],
)
def test_cost_equals_equivalent_einsum(op, subs, shapes):
    arrs = [arr(*s) for s in shapes]
    assert _cost(op, *arrs) == _einsum_cost(subs, *arrs)


@pytest.mark.parametrize(
    "op, subs, shapes",
    [
        # A size-1 batch axis must broadcast (numpy semantics), not raise; regression
        # for the accumulation cost model rejecting a shared label with sizes {1, N}.
        (fnp.matmul, "...ij,...jk->...ik", [(3, W, W), (1, W, W)]),
        (fnp.matmul, "...ij,...jk->...ik", [(1, W, W), (3, W, W)]),
        (fnp.vecdot, "...n,...n->...", [(3, W), (1, W)]),
        pytest.param(
            fnp.vecmat, "...n,...nm->...m", [(3, W), (1, W, W)], marks=requires_vecmat
        ),
        pytest.param(
            fnp.matvec, "...mn,...n->...m", [(1, W, W), (3, W)], marks=requires_matvec
        ),
    ],
)
def test_cost_equals_einsum_with_size1_broadcast(op, subs, shapes):
    arrs = [arr(*s) for s in shapes]
    assert _cost(op, *arrs) == _einsum_cost(subs, *arrs)


def test_size1_broadcast_cost_uses_broadcast_extent():
    """A size-1 batch axis broadcasts to the other operand's extent for cost."""
    from flopscope._flops import matmul_cost

    # matmul (3,W,W) @ (1,W,W): batch 1 -> 3; cost = 3 * matmul_cost(W,W,W)
    assert _cost(fnp.matmul, arr(3, W, W), arr(1, W, W)) == 3 * matmul_cost(W, W, W)
    # vecdot (3,W) . (1,W): batch 1 -> 3; cost = 3 * (2W-1)
    assert _cost(fnp.vecdot, arr(3, W), arr(1, W)) == 3 * (2 * W - 1)


@requires_vecmat
def test_vecmat_scales_with_batch_no_unbilled_compute():
    w = arr(W, W)
    c_small = _cost(fnp.vecmat, arr(8, W), w)
    c_big = _cost(fnp.vecmat, arr(800, W), w)
    assert c_big > 50 * c_small
    assert _cost(fnp.vecmat, arr(N, W), w) == N * W * (2 * W - 1)


@requires_matvec
def test_matvec_mirror_scales_with_batch():
    w = arr(W, W)
    assert _cost(fnp.matvec, w, arr(N, W)) == N * W * (2 * W - 1)
