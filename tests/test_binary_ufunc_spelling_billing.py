import warnings

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import get_dtype_rate, get_weight, load_weights


def _delta(ctx, call):
    before = ctx.flops_used
    result = call()
    return result, ctx.flops_used - before, ctx.op_log[-1].resolved_dtype


_AFFECTED = (
    "arctan2",
    "atan2",
    "copysign",
    "heaviside",
    "hypot",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "nextafter",
)

_NARROW = {
    np.bool_: np.float16,
    np.int8: np.float16,
    np.uint8: np.float16,
    np.int16: np.float32,
    np.uint16: np.float32,
}


@pytest.mark.parametrize("name", ("equal", "less", "logical_and"))
@pytest.mark.parametrize("explicit_bool_dtype", (False, True))
def test_complex64_predicates_preserve_input_kind_in_billing(name, explicit_bool_dtype):
    load_weights()
    left = np.array([1 + 2j, 0 + 0j], dtype=np.complex64)
    right = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    kwargs = {"dtype": np.bool_} if explicit_bool_dtype else {}
    np_func = getattr(np, name)
    expected_direct = np_func(left[:, None], right[None, :], **kwargs)
    expected_outer = np_func.outer(left, right, **kwargs)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: getattr(fnp, name)(
                fnp.asarray(left[:, None]),
                fnp.asarray(right[None, :]),
                **kwargs,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: np_func.outer(fnp.asarray(left), fnp.asarray(right), **kwargs),
            )

    expected_bill = 4 * get_weight(name) * get_dtype_rate("complex64") * 2
    assert np.array_equal(direct, expected_direct)
    assert np.array_equal(outer, expected_outer)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "complex64"


@pytest.mark.parametrize(
    ("mixed_dtype", "expected_dtype"),
    ((np.dtype(object), "object"), (np.dtype("timedelta64[D]"), "bool")),
)
@pytest.mark.parametrize("reverse_operands", (False, True))
def test_logical_and_mixed_kind_bills_full_numpy_loop_signature(
    mixed_dtype, expected_dtype, reverse_operands
):
    load_weights()
    mixed = np.array([0, 1], dtype=mixed_dtype)
    complex_values = np.array([0 + 0j, 1 + 2j], dtype=np.complex64)
    left, right = (
        (complex_values, mixed) if reverse_operands else (mixed, complex_values)
    )
    expected_direct = np.logical_and(left[:, None], right[None, :])
    expected_outer = np.logical_and.outer(left, right)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: fnp.logical_and(
                fnp.asarray(left[:, None]), fnp.asarray(right[None, :])
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: np.logical_and.outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_bill = 4 * get_weight("logical_and")
    assert np.array_equal(direct, expected_direct)
    assert np.array_equal(outer, expected_outer)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == expected_dtype


@pytest.mark.parametrize("name", _AFFECTED)
@pytest.mark.parametrize(("dtype", "expected_loop"), _NARROW.items())
def test_narrow_binary_direct_matches_outer_spelling(name, dtype, expected_loop):
    load_weights()
    left = np.arange(1, 9).astype(dtype)
    right = np.arange(8, 0, -1).astype(dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: getattr(fnp, name)(
                fnp.asarray(left[:, None]), fnp.asarray(right[None, :])
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: getattr(np, name).outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_dtype = np.dtype(expected_loop).name
    expected_bill = 64 * get_weight(name) * get_dtype_rate(expected_dtype)
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == expected_dtype


@pytest.mark.parametrize("name", ("divide", "true_divide"))
@pytest.mark.parametrize("dtype", (np.int8, np.int16))
def test_integer_division_direct_and_outer_stay_float64(name, dtype):
    load_weights()
    left = np.arange(1, 9).astype(dtype)
    right = np.arange(8, 0, -1).astype(dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: getattr(fnp, name)(
                fnp.asarray(left[:, None]), fnp.asarray(right[None, :])
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: getattr(np, name).outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_bill = 64 * get_weight(name) * get_dtype_rate("float64")
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "float64"


@pytest.mark.parametrize(
    ("dtype", "expected_loop"),
    (
        (np.int32, np.float64),
        (np.int64, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ),
)
def test_hypot_wide_inputs_keep_numpy_loop(dtype, expected_loop):
    load_weights()
    left = np.arange(1, 9).astype(dtype)
    right = np.arange(8, 0, -1).astype(dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: fnp.hypot(fnp.asarray(left[:, None]), fnp.asarray(right[None, :])),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: np.hypot.outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_dtype = np.dtype(expected_loop).name
    expected_bill = 64 * get_weight("hypot") * get_dtype_rate(expected_dtype)
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == expected_dtype


@pytest.mark.parametrize(
    ("left_dtype", "right_dtype", "expected_billing_dtype"),
    (
        (np.int8, np.int8, np.float16),
        (np.int8, np.int32, np.float16),
        (np.int16, np.int32, np.float32),
        (np.int8, np.int64, np.int64),
    ),
)
def test_ldexp_asymmetric_loop_and_input_floor(
    left_dtype, right_dtype, expected_billing_dtype
):
    load_weights()
    left = np.arange(1, 9).astype(left_dtype)
    right = np.arange(8).astype(right_dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: fnp.ldexp(fnp.asarray(left[:, None]), fnp.asarray(right[None, :])),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: np.ldexp.outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_dtype = np.dtype(expected_billing_dtype).name
    expected_bill = 64 * get_weight("ldexp") * get_dtype_rate(expected_dtype)
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == expected_dtype


def test_ldexp_float32_int32_promoted_input_floor_matches_outer():
    load_weights()
    mantissas = np.arange(1, 9, dtype=np.float32)
    exponents = np.arange(8, dtype=np.int32)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: fnp.ldexp(
                fnp.asarray(mantissas[:, None]), fnp.asarray(exponents[None, :])
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: np.ldexp.outer(fnp.asarray(mantissas), fnp.asarray(exponents)),
            )

    expected_bill = 64 * get_weight("ldexp") * get_dtype_rate("float64")
    assert np.array_equal(direct, outer)
    assert direct_bill == expected_bill
    assert direct_dtype == "float64"
    assert outer_bill == expected_bill
    assert outer_dtype == "float64"


@pytest.mark.parametrize(
    ("dtype", "scalar"),
    ((np.int8, 1), (np.int16, 1), (np.float32, 1.0)),
)
def test_direct_hypot_preserves_weak_scalar_loop(dtype, scalar):
    load_weights()
    values = np.arange(1, 9).astype(dtype)
    expected = np.hypot(values, scalar)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        actual, bill, resolved_dtype = _delta(
            ctx, lambda: fnp.hypot(fnp.asarray(values), scalar)
        )

    expected_dtype = expected.dtype.name
    assert np.array_equal(actual, expected, equal_nan=True)
    assert actual.dtype == expected.dtype
    assert resolved_dtype == expected_dtype
    assert bill == 8 * get_weight("hypot") * get_dtype_rate(expected_dtype)


@pytest.mark.parametrize(("dtype", "expected_loop"), _NARROW.items())
def test_direct_hypot_matches_two_row_reduce(dtype, expected_loop):
    load_weights()
    values = np.arange(1, 9).astype(dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx, lambda: fnp.hypot(fnp.asarray(values), fnp.asarray(values))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reduced, reduced_bill, reduced_dtype = _delta(
                ctx,
                lambda: np.hypot.reduce(
                    fnp.asarray(np.stack((values, values))), axis=0
                ),
            )

    expected_dtype = np.dtype(expected_loop).name
    expected_bill = 8 * get_weight("hypot") * get_dtype_rate(expected_dtype)
    assert np.array_equal(direct, reduced, equal_nan=True)
    assert direct_bill == reduced_bill == expected_bill
    assert direct_dtype == reduced_dtype == expected_dtype
