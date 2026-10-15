"""Cumulative scans keep the symmetry of the axes they do not scan (#279).

``cumsum``/``cumprod`` (and ``cumulative_*``, ``nancum*``) return the input's
shape. Only the scanned axis loses its symmetry partners; the remaining group
keeps its axis positions. Deriving the tag as if the scanned axis had been
removed shifted the group one axis to the left, which under-billed later
reductions over the last axis and made valid contractions raise.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

_OPS = ["cumsum", "cumprod", "nancumsum", "nancumprod"] + [
    name for name in ("cumulative_sum", "cumulative_prod") if hasattr(np, name)
]


def _fully_symmetric(n: int, ndim: int, seed: int = 0) -> np.ndarray:
    t = np.random.default_rng(seed).uniform(0.5, 1.5, (n,) * ndim)
    perms = list(itertools.permutations(range(ndim)))
    total = np.zeros_like(t)
    for perm in perms:
        total += np.transpose(t, perm)
    return total / len(perms)


def _tag_axes(arr):
    """Axes of ``arr``'s symmetry tag, or None when it carries none."""
    group = getattr(arr, "symmetry", None)
    return None if group is None else tuple(group.axes)


def _tagged(values: np.ndarray):
    return flops.as_symmetric(fnp.asarray(values), symmetry=tuple(range(values.ndim)))


def _bill(fn):
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        fn()
    return int(bc.flops_used)


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("axis", "kept"), [(0, (1, 2)), (1, (0, 2)), (-1, (0, 1))])
def test_scan_keeps_the_unscanned_axes_tagged(op, axis, kept):
    values = _fully_symmetric(6, 3)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        out = getattr(fnp, op)(_tagged(values), axis=axis)
    assert _tag_axes(out) == kept
    raw = np.asarray(out)
    assert np.allclose(raw, np.swapaxes(raw, *kept))


def test_issue_repro_bills_like_the_untagged_twin():
    # The issue's 12x12x12 repro: with the tag on (0, 1) the reduction and
    # the contraction were billed 1.85x below the same values untagged.
    t = np.random.default_rng(0).standard_normal((12, 12, 12))
    T = np.zeros_like(t)
    for perm in itertools.permutations(range(3)):
        T += np.transpose(t, perm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            W = fnp.cumsum(_tagged(T), axis=0)
            U = fnp.cumsum(fnp.asarray(T), axis=0)
        assert _tag_axes(W) == (1, 2)
        # numel(in) - numel(out) for the dense sum over the last axis.
        assert _bill(lambda: fnp.sum(W, axis=-1)) == 12 * 12 * 11
        assert _bill(lambda: fnp.sum(U, axis=-1)) == 12 * 12 * 11
        tagged = _bill(lambda: fnp.einsum("ijk,ijz->kz", W, W))
        assert tagged == _bill(lambda: fnp.einsum("ijk,ijz->kz", U, U))
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            fnp.einsum("ijk->ij", W)  # raised SymmetryError before the fix


def test_four_d_scan_keeps_the_three_unscanned_axes():
    values = _fully_symmetric(5, 4, seed=1)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        out = fnp.cumsum(_tagged(values), axis=0)
    assert _tag_axes(out) == (1, 2, 3)


def test_two_d_and_flattened_scans_are_unchanged():
    # A 2-D scan leaves one unscanned axis, which has no partner, and
    # axis=None scans the flattened array: neither result is tagged.
    values = _fully_symmetric(6, 2)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        assert _tag_axes(fnp.cumsum(_tagged(values), axis=0)) is None
        flat = fnp.cumsum(_tagged(_fully_symmetric(4, 3)))
    assert flat.shape == (64,)
    assert _tag_axes(flat) is None


def test_reductions_keep_their_keepdims_behaviour():
    # The fix is scoped to the scans: sum still renumbers without keepdims.
    values = _fully_symmetric(6, 3)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        reduced = fnp.sum(_tagged(values), axis=0)
        kept = fnp.sum(_tagged(values), axis=0, keepdims=True)
    assert _tag_axes(reduced) == (0, 1)
    assert _tag_axes(kept) == (1, 2)
