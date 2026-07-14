"""Four-factor billing: charged = int(flop_cost * dtype_rate * complex_factor * weight)."""

import numpy as np
import pytest

import flopscope as f
from flopscope._weights import load_weights

# Inputs built outside any BudgetContext (input construction is billed).
_f32 = np.ones(10, dtype=np.float32)
_f64 = np.ones(10, dtype=np.float64)
_c128 = np.ones(10, dtype=np.complex128)


def _charge(op_name, flop_cost, dtypes, override=None) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            op_name,
            flop_cost=flop_cost,
            subscripts=None,
            shapes=(),
            dtypes=dtypes,
            complex_factor_override=override,
        ):
            pass
        return b.flops_used


def test_unit_mode_real_dtypes_bill_flop_cost():
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 10


def test_unit_mode_complex_factor_still_applies():
    # complex_factor is math, not policy: active even under unit rates/weights
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 60
    assert _charge("add", 10, (np.dtype("complex128"),)) == 20


def test_production_rates_compose():
    load_weights()
    assert _charge("multiply", 10, (np.dtype("float32"),)) == 10
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 20
    assert _charge("multiply", 10, (np.dtype("complex64"),)) == 60
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 120
    assert _charge("add", 10, (np.dtype("complex128"),)) == 40


def test_override_bypasses_registry_factor():
    assert (
        _charge("einsum", 960, (np.dtype("complex128"),), override=3968 / 960) == 3968
    )


def test_dtype_neutral_and_unmigrated():
    assert _charge("einsum_path", 1, ()) == 1  # declared neutral
    assert _charge("einsum_path", 1, None) == 1  # unmigrated site (until Task 9)


def test_resolved_dtype_recorded():
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            "multiply",
            flop_cost=1,
            subscripts=None,
            shapes=(),
            dtypes=(np.dtype("float32"), np.dtype("float64")),
        ):
            pass
        assert b._op_log[-1].resolved_dtype == "float64"


def test_unsupported_dtype_fails_closed_before_charging():
    load_weights()
    from flopscope.errors import UnsupportedDtypeError

    # float128 is unavailable on this platform; object exercises the same fail-closed path
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with pytest.raises(UnsupportedDtypeError):
            with b.deduct(
                "multiply",
                flop_cost=1,
                subscripts=None,
                shapes=(),
                dtypes=(np.dtype("object"),),
            ):
                pass
        assert b.flops_used == 0


# ---------------------------------------------------------------------------
# Task 6: pointwise/reduction factories declare their billing dtypes
# ---------------------------------------------------------------------------

import flopscope.numpy as fnp  # noqa: E402

_a32 = fnp.asarray(np.ones(10, dtype=np.float32))
_b32 = fnp.asarray(np.ones(10, dtype=np.float32))
_z = fnp.asarray(np.ones(10, dtype=np.complex128))


def _cost(fn) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def test_elementwise_multiply_complex_bills_six_per_element():
    assert _cost(lambda: fnp.multiply(_z, _z)) == 60  # unit rates; factor 6


def test_elementwise_add_complex_bills_two_per_element():
    assert _cost(lambda: fnp.add(_z, _z)) == 20


def test_nep50_scalar_keeps_float32_billing():
    load_weights()  # production rates: f32=1.0, f64=2.0
    assert _cost(lambda: fnp.multiply(_a32, 2.0)) == 10  # stays float32
    assert _cost(lambda: fnp.multiply(_a32, _b32)) == 10
    _a64 = fnp.asarray(np.ones(10, dtype=np.float64))
    assert _cost(lambda: fnp.multiply(_a64, _a64)) == 20


def test_explicit_dtype_kwarg_raises_billing_dtype():
    load_weights()
    assert _cost(lambda: fnp.add(_a32, _b32, dtype=np.float64)) == 20


def test_reduction_sum_complex_bills_factor_two():
    # ratio-based: robust to the reduction skeleton's exact flop_cost formula
    _r = fnp.asarray(np.ones(10, dtype=np.float64))
    assert _cost(lambda: fnp.sum(_z)) == 2 * _cost(lambda: fnp.sum(_r))


# ---------------------------------------------------------------------------
# Task 6b: generic ufunc-method dispatch (np.<ufunc>.<method> on a
# FlopscopeArray) inherits its base ufunc's complex factor via fallback.
#
# ``multiply``/``add`` are routed to a dedicated fast path
# (FlopscopeArray._REDUCE_TO_WHEST) for ``.reduce``/``.accumulate``, so
# those methods on those two ufuncs do NOT exercise the generic dispatch
# sites this task changes. ``.outer`` always uses the generic path
# regardless of ufunc, and ``subtract`` is never whest-routed — so
# ``multiply.outer`` and ``subtract.reduce``/``subtract.accumulate`` are
# the calls that genuinely reach _counted_ufunc_outer /
# _counted_ufunc_reduce_generic / _counted_ufunc_accumulate_generic.
# ---------------------------------------------------------------------------

_r10 = fnp.asarray(np.ones(10, dtype=np.float64))


def test_generic_ufunc_outer_complex_bills_base_ufunc_factor():
    # multiply.outer always uses the generic path; base factor is multiply's (6).
    assert _cost(lambda: np.multiply.outer(_z, _z)) == 6 * _cost(
        lambda: np.multiply.outer(_r10, _r10)
    )


def test_generic_ufunc_reduce_complex_runs_and_bills_base_ufunc_factor():
    # subtract is not whest-routed, so subtract.reduce reaches the generic
    # fallback. Before this task's fix, dtypes=None there billed complex
    # inputs as dtype-neutral (factor 1) instead of raising OR correctly
    # scaling — this is the regression the naive "just add dtypes=" would
    # have caused (fail-closed RAISE on every complex generic-ufunc-method
    # call). Asserting it both runs AND bills factor 2 (subtract's factor)
    # covers that regression.
    assert _cost(lambda: np.subtract.reduce(_z)) == 2 * _cost(
        lambda: np.subtract.reduce(_r10)
    )


def test_generic_ufunc_accumulate_complex_runs_and_bills_base_ufunc_factor():
    assert _cost(lambda: np.subtract.accumulate(_z)) == 2 * _cost(
        lambda: np.subtract.accumulate(_r10)
    )
