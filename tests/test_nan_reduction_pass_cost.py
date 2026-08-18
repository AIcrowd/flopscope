"""``nan*`` reductions run an extra isnan pass and must be charged for it.

Each nan-prefixed reduction tests every element for NaN before reducing -- work
its plain sibling does not do. The cost model charges every other value test
(count_nonzero, 1-arg where, isclose), so these must be charged too.

Beyond the eleven factory-built ops (`_counted_reduction`, `_counted_mean`,
`_counted_variance`), `nanmedian`, `nanpercentile`, and `nanquantile` are
hand-written functions in `_pointwise.py` with the identical defect -- they
are not reached by a factory-level `op_name.startswith("nan")` rule, so they
are covered here too.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

# (nan op, plain sibling) -- every nan* reduction on the counted surface that
# shares its plain sibling's single-array call signature.
_PAIRS = [
    ("nansum", "sum"),
    ("nanprod", "prod"),
    ("nanmean", "mean"),
    ("nanvar", "var"),
    ("nanstd", "std"),
    ("nanmax", "max"),
    ("nanmin", "min"),
    ("nanargmax", "argmax"),
    ("nanargmin", "argmin"),
    ("nancumsum", "cumsum"),
    ("nancumprod", "cumprod"),
    ("nanmedian", "median"),
]

# (nan op, plain sibling, q) -- the quantile-family ops need a second
# positional argument, so they cannot share _PAIRS's single-arg call shape.
_Q_PAIRS = [
    ("nanpercentile", "percentile", 50),
    ("nanquantile", "quantile", 0.5),
]


def _billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.mark.parametrize("nan_name, plain_name", _PAIRS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128])
def test_nan_variant_costs_more_than_its_plain_sibling(nan_name, plain_name, dtype):
    """The isnan pass is real work; the nan* form must never bill the same."""
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x)) > _billed(lambda: plain_op(x))


@pytest.mark.parametrize("nan_name, plain_name", _PAIRS)
def test_nan_pass_surcharge_is_one_per_element(nan_name, plain_name):
    """The surcharge is exactly one pass over the input: numel(input)."""
    n = 10_000
    x = fnp.array(np.ones(n, dtype=np.float64))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    delta = _billed(lambda: nan_op(x)) - _billed(lambda: plain_op(x))
    # Weight-independent: the surcharge scales linearly with n at a fixed rate,
    # so doubling the input doubles the surcharge.
    x2 = fnp.array(np.ones(2 * n, dtype=np.float64))
    delta2 = _billed(lambda: nan_op(x2)) - _billed(lambda: plain_op(x2))
    assert delta > 0
    assert delta2 == 2 * delta


@pytest.mark.parametrize("nan_name, plain_name, q", _Q_PAIRS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nan_quantile_variant_costs_more_than_its_plain_sibling(
    nan_name, plain_name, q, dtype
):
    """Same invariant as the single-array family, for the quantile ops.

    No complex128 case here: numpy itself rejects complex input for both
    percentile and quantile ("a must be an array of real numbers"), so the
    registry marks both families complex_factor="illegal" -- that is a
    pre-existing, unrelated restriction, not part of this defect.
    """
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x, q)) > _billed(lambda: plain_op(x, q))


@pytest.mark.parametrize("nan_name, plain_name, q", _Q_PAIRS)
def test_nan_quantile_pass_surcharge_is_one_per_element(nan_name, plain_name, q):
    """Same one-pass-per-element invariant, for the quantile ops."""
    n = 10_000
    x = fnp.array(np.ones(n, dtype=np.float64))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    delta = _billed(lambda: nan_op(x, q)) - _billed(lambda: plain_op(x, q))
    x2 = fnp.array(np.ones(2 * n, dtype=np.float64))
    delta2 = _billed(lambda: nan_op(x2, q)) - _billed(lambda: plain_op(x2, q))
    assert delta > 0
    assert delta2 == 2 * delta


def test_plain_reductions_are_unchanged():
    """Only the nan* family moves; plain reductions keep their price."""
    x = fnp.array(np.ones(1000, dtype=np.float64))
    y = fnp.array(np.ones(2000, dtype=np.float64))
    # A plain reduction's cost still scales with its own element count only.
    assert _billed(lambda: fnp.sum(y)) > _billed(lambda: fnp.sum(x))
