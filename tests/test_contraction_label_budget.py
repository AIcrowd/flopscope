"""The 52-letter subscript budget: allocation, fallback pricing, invariants."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._pointwise import _dense_accumulation_cost


def billed(fn) -> int:
    """Billed FLOPs for `fn`, warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as b:
            fn()
            return b.flops_used


# `tests/conftest.py` has an autouse fixture calling `reset_weights()`, which
# puts the suite in unit-weight mode: every op weight and dtype rate is 1.0, so
# `billed()` returns the raw FLOP cost. Do NOT introduce a dtype-weight divisor
# here — production weights get re-tuned, and a test that hardcodes one would
# break on every re-calibration. Compare raw costs.


TENSORDOT_SHAPES = [
    ((8, 6), (6, 5), ([1], [0])),
    ((4, 3, 5), (5, 7), ([2], [0])),
    ((2, 3, 4), (3, 4, 6), ([1, 2], [0, 1])),
    ((9,), (9,), ([0], [0])),
    ((5, 4), (4,), ([1], [0])),
    ((3, 3, 3), (3, 3, 3), ([2], [0])),
    ((7, 2, 3), (2, 3, 11), ([1, 2], [0, 1])),
    ((6, 1, 4), (4, 1, 6), ([2], [0])),
    ((2, 2), (2, 2), ([0, 1], [0, 1])),
    ((10, 3), (3, 10), ([1], [0])),
    ((1, 5), (5, 1), ([1], [0])),
    ((4, 5, 6), (6,), ([2], [0])),
]


def _geometry(a_shape, b_shape, axes):
    a_ax, b_ax = axes
    contracted = math.prod(a_shape[i] for i in a_ax)
    output_shape = tuple(s for i, s in enumerate(a_shape) if i not in a_ax) + tuple(
        s for j, s in enumerate(b_shape) if j not in b_ax
    )
    return contracted, output_shape


@pytest.mark.parametrize("a_shape,b_shape,axes", TENSORDOT_SHAPES)
def test_dense_cost_matches_einsum_path(a_shape, b_shape, axes):
    """The label-free formula must equal what the einsum path charges."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(a_shape)
    b = rng.standard_normal(b_shape)
    einsum_path = billed(
        lambda: fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=axes)
    )
    contracted, output_shape = _geometry(a_shape, b_shape, axes)
    assert (
        _dense_accumulation_cost(a.size, b.size, contracted, output_shape).total
        == einsum_path
    )


@pytest.mark.parametrize(
    "a_shape,b_shape,axes",
    [
        ((8, 0), (0, 5), ([1], [0])),  # zero-length contracted axis
        ((0, 6), (6, 5), ([1], [0])),  # zero-size output via a
        ((8, 6), (6, 0), ([1], [0])),  # zero-size output via b
    ],
)
def test_empty_domain_bills_zero_never_negative(a_shape, b_shape, axes):
    """A zero-sized contraction bills 0. It must never refund budget."""
    contracted, output_shape = _geometry(a_shape, b_shape, axes)
    a_size = math.prod(a_shape)
    b_size = math.prod(b_shape)
    cost = _dense_accumulation_cost(a_size, b_size, contracted, output_shape)
    assert cost.total == 0
    assert cost.total >= 0


def test_dense_cost_exposes_multiply_add_split():
    """mu is the multiply count, so complex billing can derive its exact ratio."""
    # (8,6)x(6,5): alpha = 8*5*6 = 240 multiplies, M = 40 cells, 200 adds.
    cost = _dense_accumulation_cost(48, 30, 6, (8, 5))
    assert cost.mu == 240
    assert cost.total == 2 * 240 - 40
    assert cost.num_terms == 2
    assert cost.fallback_used is False


from flopscope._pointwise import _contraction_subscripts


def test_subscripts_ties_contracted_pairs():
    """Contracted axis pairs share a label; free axes get distinct ones.

    b's labels start at offset a_ndim, so for two rank-2 operands b is
    initially 'cd'; tying b's axis 0 to a's axis 1 rewrites it to 'bd'.
    """
    assert _contraction_subscripts(2, 2, (1,), (0,)) == "ab,bd->ad"


def test_subscripts_handles_multiple_contracted_axes():
    assert _contraction_subscripts(3, 3, (1, 2), (0, 1)) == "abc,bcf->af"


def test_subscripts_normalises_negative_axes():
    """Negative axis indices mean the same thing as their positive form."""
    assert _contraction_subscripts(2, 2, (-1,), (-2,)) == _contraction_subscripts(
        2, 2, (1,), (0,)
    )


def test_subscripts_returns_none_above_budget():
    """52 letters exist, so a rank sum above 52 has no representation."""
    assert _contraction_subscripts(26, 26, (1,), (0,)) is not None
    assert _contraction_subscripts(27, 26, (1,), (0,)) is None
    assert _contraction_subscripts(27, 27, (1,), (0,)) is None


def test_subscripts_full_contraction_to_scalar():
    """All axes contracted on both sides gives an empty output."""
    assert _contraction_subscripts(3, 3, (0, 1, 2), (0, 1, 2)) == "abc,abc->"


def _pad_end(arr, n):
    """Append n singleton axes via basic indexing — a free, unmetered view."""
    return arr[(slice(None),) * arr.ndim + (None,) * n]


def _pad_front(arr, n):
    return arr[(None,) * n + (slice(None),) * arr.ndim]


@pytest.mark.parametrize("n_pad", [0, 10, 24, 25, 26, 30])
def test_tensordot_padding_does_not_change_bill(n_pad):
    """Singleton axes carry no arithmetic, so they must carry no price.

    Padding to 25+ per operand pushes the rank sum past 52, which is where
    the subscript budget runs out. Rank must not be a discount.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    baseline = billed(
        lambda: fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([1], [0]))
    )
    padded = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), n_pad),
            _pad_end(fnp.asarray(b), n_pad),
            axes=([1], [0]),
        )
    )
    assert padded == baseline


def test_tensordot_above_budget_bills_fma_not_multiplies_only():
    """Above the budget the bill is 2*alpha - M, not the multiply count."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    got = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), 25),
            _pad_end(fnp.asarray(b), 25),
            axes=([1], [0]),
        )
    )
    alpha, m = 32 * 8 * 16, 32 * 8
    assert got == 2 * alpha - m
    assert got != alpha  # the old multiply-only price


@pytest.mark.parametrize("n_pad", [0, 25])
def test_tensordot_padding_preserves_values(n_pad):
    """The fix must not change what the operation computes."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fnp.tensordot(
                _pad_end(fnp.asarray(a), n_pad),
                _pad_end(fnp.asarray(b), n_pad),
                axes=([1], [0]),
            )
    assert np.allclose(np.squeeze(np.asarray(got)), a @ b)


@pytest.mark.parametrize("n_pad", [0, 25])
def test_complex_tensordot_above_budget_bills_exactly(n_pad):
    """Complex operands must bill, not raise fail-closed, above the budget."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))
    b = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
    got = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), n_pad),
            _pad_end(fnp.asarray(b), n_pad),
            axes=([1], [0]),
        )
    )
    assert got > 0
