"""``linalg.cond(x, 'nuc')`` bills the two nuclear norms it computes (#329).

For ``p`` outside ``{None, 2, -2}`` numpy returns
``norm(x, p) * norm(inv(x), p)``. For ``p='nuc'`` both factors are nuclear
norms, which numpy evaluates with a values-only SVD each and which
``linalg.norm(x, 'nuc')`` already bills at that rate, but ``cond_cost``
priced them as two elementwise passes, the ``'fro'`` price.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._flops import svd_cost


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _well_conditioned(shape, seed=0):
    n = shape[-1]
    return np.random.default_rng(seed).standard_normal(shape) + 8 * np.eye(n)


def _written_out_bill(x, p):
    """What numpy's own steps bill: inv(x), then norm(x, p) and norm(inv(x), p)."""
    inv_bill, inv_x = _billed(lambda: fnp.linalg.inv(x))
    norm_bill, _ = _billed(lambda: fnp.linalg.norm(x, p, axis=(-2, -1)))
    inv_norm_bill, _ = _billed(lambda: fnp.linalg.norm(inv_x, p, axis=(-2, -1)))
    return inv_bill + norm_bill + inv_norm_bill


def test_issue_repro_nuc_bills_two_values_only_svds():
    A = fnp.asarray(_well_conditioned((64, 64)))
    billed, got = _billed(lambda: fnp.linalg.cond(A, "nuc"))
    _, ref = _billed(
        lambda: fnp.linalg.norm(A, "nuc") * fnp.linalg.norm(fnp.linalg.inv(A), "nuc")
    )
    assert float(got) == np.linalg.cond(np.asarray(A), "nuc")
    assert np.isclose(float(got), float(ref))
    assert billed == _written_out_bill(A, "nuc") + 1
    assert billed == 2 * 64**3 + 2 * svd_cost(64, 64, with_vectors=False) + 1


SHAPES = [(64, 64), (6, 6), (8, 32, 32), (2, 3, 5, 5)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", ["nuc", "fro", 1, -1, np.inf, -np.inf])
def test_inverse_based_orders_bill_their_written_out_steps(p, shape):
    # One multiply per matrix joins the two norms; nuc was billed as fro.
    x = fnp.asarray(_well_conditioned(shape))
    batch = int(np.prod(shape[:-2], dtype=np.int64))
    billed, got = _billed(lambda: fnp.linalg.cond(x, p))
    assert np.array_equal(np.asarray(got), np.linalg.cond(np.asarray(x), p))
    assert billed == _written_out_bill(x, p) + batch


@pytest.mark.parametrize("p", [None, 2, -2])
def test_svd_based_orders_are_unchanged(p):
    x = fnp.asarray(_well_conditioned((3, 7, 7)))
    billed, got = _billed(lambda: fnp.linalg.cond(x, p))
    assert np.array_equal(np.asarray(got), np.linalg.cond(np.asarray(x), p))
    assert billed == 3 * (svd_cost(7, 7, with_vectors=False) + 1)
