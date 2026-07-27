"""``mask_indices`` must be priced like the scan it performs.

numpy's body is ``a = mask_func(ones((n, n)), k); return nonzero(a != 0)``, so
the honest cost is what ``nonzero`` costs on that n x n array. Pricing it off
the returned index count instead made it an arbitrarily cheap substitute for
``nonzero``.
"""

from __future__ import annotations

import numpy as np

import flopscope
import flopscope.numpy as fnp


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        before = ctx.flops_used
        fn()
        return ctx.flops_used - before


def test_mask_indices_costs_what_nonzero_costs_on_the_same_array():
    n = 200
    sparse = np.zeros((n, n), bool)
    sparse[0, :3] = True
    via_mask = billed(lambda: fnp.mask_indices(n, lambda m, k: sparse))
    via_nonzero = billed(lambda: fnp.nonzero(fnp.asarray(sparse)))
    assert via_mask == via_nonzero


def test_mask_indices_costs_what_nonzero_costs_on_a_dense_mask():
    """A sparse mask alone does not pin the invariant: a mask that keeps just
    over half its elements (the default ``triu``/``tril`` case) is exactly
    where a returned-index-count formula and a mask-scan formula diverge.
    Assert parity for that case too, at more than one size.
    """
    for n in (8, 200):
        dense = np.triu(np.ones((n, n), int))
        via_mask = billed(lambda n=n: fnp.mask_indices(n, np.triu))
        via_nonzero = billed(lambda dense=dense: fnp.nonzero(fnp.asarray(dense)))
        assert via_mask == via_nonzero


def test_mask_indices_scales_with_the_probe_not_the_output():
    """A tiny result must not buy a large scan."""
    small = billed(lambda: fnp.mask_indices(50, lambda m, k: np.zeros((50, 50), bool)))
    large = billed(
        lambda: fnp.mask_indices(200, lambda m, k: np.zeros((200, 200), bool))
    )
    assert large > small


def test_tri_indices_helpers_are_unchanged():
    """These do not route through the counted mask_indices wrapper, so
    repricing mask_indices must not move them. Pinned to the measured
    pre-change values (n*(n+1)) so a regression here is caught."""
    for n in (10, 100):
        assert billed(lambda n=n: fnp.triu_indices(n)) == n * (n + 1)
        assert billed(lambda n=n: fnp.tril_indices(n)) == n * (n + 1)
