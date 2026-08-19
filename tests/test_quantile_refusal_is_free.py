"""A refused percentile/quantile call must cost nothing.

Reject-before-billing is the model's stated pattern (svd's invalid k, index
reductions' out= refusal). An out-of-range q does no work, so it must not
consume budget.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _charged_on_refusal(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        before = b.flops_used
        with pytest.raises(ValueError):
            fn()
        return b.flops_used - before


@pytest.mark.parametrize("bad_q", [150, -5, 101])
def test_percentile_refusal_charges_nothing(bad_q):
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: fnp.percentile(x, bad_q)) == 0


@pytest.mark.parametrize("bad_q", [2.0, -0.5, 1.5])
def test_quantile_refusal_charges_nothing(bad_q):
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: fnp.quantile(x, bad_q)) == 0


def test_valid_calls_still_bill():
    """The guard must not make legitimate calls free."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    with flops.budget(10**15, quiet=True) as b:
        fnp.percentile(x, 50)
        assert b.flops_used > 0


def test_array_q_with_one_bad_entry_is_refused_free():
    """An array q containing an out-of-range entry is refused, and costs 0."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: fnp.percentile(x, [50, 150])) == 0
