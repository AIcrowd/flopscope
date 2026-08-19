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


@pytest.mark.parametrize("boundary_q", [0, 100])
@pytest.mark.parametrize(
    "func", [fnp.percentile, fnp.nanpercentile], ids=lambda f: f.__name__
)
def test_percentile_inclusive_boundary_is_accepted_and_billed(func, boundary_q):
    """q=0 and q=100 are the inclusive edges of a valid percentile, not
    refusals. Pins the other half of the range-check property: a future
    `>` -> `>=` (or `<` -> `<=`) typo in the guard would wrongly refuse
    these and escape the refusal-only tests above."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    with flops.budget(10**15, quiet=True) as b:
        func(x, boundary_q)
        assert b.flops_used > 0


@pytest.mark.parametrize("boundary_q", [0.0, 1.0])
@pytest.mark.parametrize(
    "func", [fnp.quantile, fnp.nanquantile], ids=lambda f: f.__name__
)
def test_quantile_inclusive_boundary_is_accepted_and_billed(func, boundary_q):
    """q=0 and q=1 are the inclusive edges of a valid quantile, not
    refusals. Same boundary-typo guard as the percentile-family test above."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    with flops.budget(10**15, quiet=True) as b:
        func(x, boundary_q)
        assert b.flops_used > 0


@pytest.mark.parametrize(
    "func",
    [fnp.percentile, fnp.nanpercentile, fnp.quantile, fnp.nanquantile],
    ids=lambda f: f.__name__,
)
def test_nan_q_is_refused_free(func):
    """A NaN q is out of range for numpy, so it must cost nothing here too.

    NaN is the one input on which ``q.min() < 0 or q.max() > hi`` and numpy's
    own ``not (q.min() >= 0 and q.max() <= hi)`` disagree: every comparison
    against NaN is False, so the positive form does not fire, the full cost is
    deducted, and numpy raises the identical ValueError immediately after --
    the exact charge-then-refuse leak this guard exists to close. Measured
    through the real client before the fix: 20008 FLOPs on ``quantile``,
    40008 on ``nanquantile``.
    """
    x = fnp.array(np.arange(100, dtype=np.float64))
    assert _charged_on_refusal(lambda: func(x, float("nan"))) == 0


@pytest.mark.parametrize(
    "func",
    [fnp.percentile, fnp.nanpercentile, fnp.quantile, fnp.nanquantile],
    ids=lambda f: f.__name__,
)
def test_array_q_with_a_nan_entry_is_refused_free(func):
    """One NaN among otherwise valid entries is still a refusal, still free."""
    x = fnp.array(np.arange(100, dtype=np.float64))
    valid = 50.0 if func in (fnp.percentile, fnp.nanpercentile) else 0.5
    assert _charged_on_refusal(lambda: func(x, [valid, float("nan")])) == 0


@pytest.mark.parametrize(
    "func, hi",
    [
        (fnp.percentile, 100),
        (fnp.nanpercentile, 100),
        (fnp.quantile, 1),
        (fnp.nanquantile, 1),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_nan_q_refusal_message_matches_numpy(func, hi):
    """The refusal must stay indistinguishable from numpy's own."""
    x = np.arange(100, dtype=np.float64)
    numpy_func = getattr(np, func.__name__)
    with pytest.raises(ValueError) as numpy_exc:
        numpy_func(x, float("nan"))
    with flops.budget(10**15, quiet=True):
        with pytest.raises(ValueError) as ours:
            func(fnp.array(x), float("nan"))
    assert str(ours.value) == str(numpy_exc.value)
