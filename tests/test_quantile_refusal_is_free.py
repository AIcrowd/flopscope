"""A refused percentile/quantile call must cost nothing.

Reject-before-billing is the model's stated pattern (svd's invalid k, index
reductions' out= refusal). An out-of-range q does no work, so it must not
consume budget. Covers both the plain and nan-prefixed forms: nanpercentile/
nanquantile route to the same numpy ValueError and had the identical leak
(Ruling R8).
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
@pytest.mark.parametrize(
    "func", [fnp.percentile, fnp.nanpercentile], ids=lambda f: f.__name__
)
def test_percentile_refusal_charges_nothing(func, bad_q):
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: func(x, bad_q)) == 0


@pytest.mark.parametrize("bad_q", [2.0, -0.5, 1.5])
@pytest.mark.parametrize(
    "func", [fnp.quantile, fnp.nanquantile], ids=lambda f: f.__name__
)
def test_quantile_refusal_charges_nothing(func, bad_q):
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: func(x, bad_q)) == 0


@pytest.mark.parametrize(
    "func",
    [fnp.percentile, fnp.nanpercentile, fnp.quantile, fnp.nanquantile],
    ids=lambda f: f.__name__,
)
def test_valid_calls_still_bill(func):
    """The guard must not make legitimate calls free."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    q = 50 if func in (fnp.percentile, fnp.nanpercentile) else 0.5
    with flops.budget(10**15, quiet=True) as b:
        func(x, q)
        assert b.flops_used > 0


@pytest.mark.parametrize(
    "func", [fnp.percentile, fnp.nanpercentile], ids=lambda f: f.__name__
)
def test_array_q_with_one_bad_entry_is_refused_free(func):
    """An array q containing an out-of-range entry is refused, and costs 0."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: func(x, [50, 150])) == 0
