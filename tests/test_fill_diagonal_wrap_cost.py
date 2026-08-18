"""fill_diagonal must bill the cells numpy actually writes, including under wrap.

Billing is ``flop_cost x dtype_rate`` (measured: float64 rate = 2.0, float32
rate = 1.0), so assertions here derive the per-cell rate empirically from the
already-correct ``wrap=False`` case rather than hardcoding it. See the
CONTROLLER CORRECTION at the top of
``.superpowers/sdd/2026-08-18-phase2-prelaunch-sprint/task-1-brief.md``.
"""

import numpy as np
import pytest
from numpy.typing import DTypeLike

import flopscope as flops
import flopscope.numpy as fnp


def billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


def cells_written(shape, wrap):
    probe = np.zeros(shape)
    np.fill_diagonal(probe, 1.0, wrap=wrap)
    return int(probe.sum())


def per_cell_rate(shape, dtype: DTypeLike = np.float64):
    # Operand built OUTSIDE the budget: fnp.array conversion is itself billed.
    a = fnp.array(np.zeros(shape, dtype=dtype))
    return billed(lambda: fnp.fill_diagonal(a, 7.0, wrap=False)) / cells_written(
        shape, wrap=False
    )


@pytest.mark.parametrize("shape", [(30_000, 3), (300_000, 3), (1500, 3), (9, 3)])
def test_wrap_true_bills_every_written_cell(shape):
    # Operand built OUTSIDE the budget: fnp.array conversion is itself billed.
    a = fnp.array(np.zeros(shape))
    rate = per_cell_rate(shape)
    expected = cells_written(shape, wrap=True) * rate
    assert billed(lambda: fnp.fill_diagonal(a, 7.0, wrap=True)) == expected


def test_wrap_true_bills_every_written_cell_float32():
    """Pin that the fix is rate-agnostic, not just correct at float64."""
    shape = (30_000, 3)
    a = fnp.array(np.zeros(shape, dtype=np.float32))
    rate = per_cell_rate(shape, dtype=np.float32)
    assert rate == 1.0  # measured float32 rate, sanity-checked
    expected = cells_written(shape, wrap=True) * rate
    assert billed(lambda: fnp.fill_diagonal(a, 7.0, wrap=True)) == expected


def test_wrap_true_is_not_a_constant():
    """The defect was a FLAT 6 FLOPs at every size. Pin that it scales."""
    small = fnp.array(np.zeros((30_000, 3)))
    large = fnp.array(np.zeros((300_000, 3)))
    b_small = billed(lambda: fnp.fill_diagonal(small, 7.0, wrap=True))
    b_large = billed(lambda: fnp.fill_diagonal(large, 7.0, wrap=True))
    assert b_large > b_small * 5


@pytest.mark.parametrize("shape", [(1000, 1000), (1_500_000, 3), (3, 1500)])
def test_wrap_false_unchanged(shape):
    """wrap=False was already correct — it must not move."""
    a = fnp.array(np.zeros(shape))
    assert billed(lambda: fnp.fill_diagonal(a, 7.0, wrap=False)) == cells_written(
        shape, wrap=False
    )


def test_square_is_identical_under_both_wraps():
    a = fnp.array(np.zeros((100, 100)))
    assert billed(lambda: fnp.fill_diagonal(a, 7.0, wrap=True)) == billed(
        lambda: fnp.fill_diagonal(a, 7.0, wrap=False)
    )
