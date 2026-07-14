"""End-to-end production-weight billing tests.

Most of the cost suite runs under UNIT weights: conftest's autouse
``reset_global_budget`` fixture calls ``reset_weights()``, clearing the table so
every op falls back to weight 1.0 and dtype_rate 1.0 — which pins each op's raw
``flop_cost``. Weight TIERS are checked separately by ``test_weight_tier_policy.py``
(only that each weight is a legal tier value). Neither pins what production
actually *bills* — ``flop_cost x dtype_rate x complex_factor x weight`` — which is
what participants are charged.

This module closes that gap: it loads the packaged production weights
(``data/default_weights.json``) and pins the billed cost for one representative
op per weight tier {0, 1, 8, 16}. A silent weight regression (e.g. a
transcendental sampler dropping from 16x to 1x) or a tier mislabel now fails
here — not only in the (unenforced) ``docs/reference/cost-model.md`` table.

Note: the former 4.0 "gather" tier (take/put/take_along_axis/put_along_axis etc.)
was replaced by the data-movement free tier (weight=0.0) in the cost-model
data-movement-free-tier change.
"""

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._weights import get_weight, load_weights, reset_weights

# Inputs are built once at import, OUTSIDE any BudgetContext, so only the
# measured op bills. (Under unit weights, array creation has its own non-zero
# flop_cost — see test_polynomial.py::test_polyfit_flopscope_array_inputs.)
_RNG = np.random.default_rng(0)
_A = fnp.asarray(_RNG.standard_normal(100))
_B = fnp.asarray(_RNG.standard_normal(100))
_IDX = fnp.asarray(_RNG.integers(0, 100, 100))


def _billed(call):
    with BudgetContext(flop_budget=10**12, quiet=True) as budget:
        call()
    return budget.flops_used


# label, weight_key, call, expected_billed (= flop_cost x dtype_rate x weight),
# expected_weight.
# One op per tier; expected_billed verified against the live model at 100 elems.
# Tiers: {0, 1, 8, 16}. The old 4.0 gather tier is gone (data-movement free).
#
# _A/_B are float64 (np.random.default_rng(...).standard_normal's default
# dtype), so add/exp resolve dtype_rate 2.0. random.randn has no dtype=
# parameter and always draws float64, so it is also dtype_rate 2.0. reshape
# and take are weight 0.0 (billed 0 regardless of dtype_rate). hanning takes
# no array operand and declares dtypes=() (dtype-neutral), so it stays at
# dtype_rate 1.0 and is unaffected by the dtype-aware billing migration.
_TIER_CASES = [
    ("free: reshape", "reshape", lambda: fnp.reshape(_A, (10, 10)), 0, 0.0),
    ("free: take (was gather)", "take", lambda: fnp.take(_A, _IDX), 0, 0.0),
    (
        "arithmetic: add",
        "add",
        lambda: fnp.add(_A, _B),
        200,
        1.0,
    ),  # 100 * 2.0(f64) * 1.0
    ("half: hanning", "hanning", lambda: fnp.hanning(100), 1600, 8.0),
    (
        "transcendental: exp",
        "exp",
        lambda: fnp.exp(_A),
        3200,
        16.0,
    ),  # 100 * 2.0(f64) * 16.0
    (
        "transcendental: random.randn",
        "random.randn",
        lambda: fnp.random.randn(100),
        3200,  # 100 * 2.0(f64, no dtype= param) * 16.0
        16.0,
    ),
]

# dtype_rate that applies to each case above (keyed by weight_key), used by
# the invariant test below. See the derivations in the _TIER_CASES comment.
_DTYPE_RATE_BY_WEIGHT_KEY = {
    "reshape": 1.0,
    "take": 1.0,
    "add": 2.0,
    "hanning": 1.0,
    "exp": 2.0,
    "random.randn": 2.0,
}


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weight table for the test body.

    The autouse ``reset_global_budget`` fixture clears weights (-> unit 1.0)
    around every test; this loads the real ``default_weights.json`` for this
    test only. ``FLOPSCOPE_WEIGHTS_FILE`` is cleared first so the packaged
    default is used regardless of the environment.
    """
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


@pytest.mark.parametrize(
    "label, weight_key, call, expected_billed, expected_weight", _TIER_CASES
)
def test_production_billed_cost_per_tier(
    production_weights, label, weight_key, call, expected_billed, expected_weight
):
    # The op carries its documented tier weight ...
    assert get_weight(weight_key) == expected_weight, (
        f"{label}: weight is {get_weight(weight_key)}, expected tier {expected_weight}"
    )
    # ... and production bills flop_cost x dtype_rate x weight (what
    # participants are charged) -- see _TIER_CASES for the per-case
    # dtype_rate derivation baked into expected_billed.
    assert _billed(call) == expected_billed, label


def test_production_billed_equals_unit_flop_cost_times_weight(monkeypatch):
    """Invariant: production billed cost == raw flop_cost x dtype_rate x weight,
    end to end. Under reset_weights() dtype_rate is unit (1.0), so `flop_cost`
    measured there is the raw, dtype-unadjusted cost; production billing then
    reapplies both the dtype_rate (_DTYPE_RATE_BY_WEIGHT_KEY) and the weight.
    """
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    for label, weight_key, call, _, _ in _TIER_CASES:
        reset_weights()  # unit weights and unit dtype rates -> raw flop_cost
        flop_cost = _billed(call)
        load_weights()  # production weights -> billed
        weight = get_weight(weight_key)
        dtype_rate = _DTYPE_RATE_BY_WEIGHT_KEY[weight_key]
        assert _billed(call) == flop_cost * dtype_rate * weight, (
            f"{label}: billed != flop_cost {flop_cost} x dtype_rate {dtype_rate} "
            f"x weight {weight}"
        )
    reset_weights()
