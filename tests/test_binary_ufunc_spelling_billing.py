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


@pytest.mark.parametrize("name", ("logical_and", "logical_or", "logical_xor"))
@pytest.mark.parametrize(
    ("left_dtype", "right_dtype"),
    (
        (np.complex128, np.bool_),
        (np.bool_, np.complex128),
        (np.complex64, np.int64),
        (np.int64, np.complex64),
    ),
)
def test_logical_mixed_inputs_use_complex_rate_floor_without_complex_factor(
    name, left_dtype, right_dtype
):
    load_weights()
    left = np.array([0, 1], dtype=left_dtype)
    right = np.array([1, 0], dtype=right_dtype)
    np_func = getattr(np, name)
    expected_direct = np_func(left[:, None], right[None, :])
    expected_outer = np_func.outer(left, right)

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
                lambda: np_func.outer(fnp.asarray(left), fnp.asarray(right)),
            )

    expected_bill = 4 * get_weight(name) * get_dtype_rate("complex128") * 1.0
    assert np.array_equal(direct, expected_direct)
    assert np.array_equal(outer, expected_outer)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "complex128"


def test_logical_and_native_complex_loop_keeps_registry_complex_factor():
    load_weights()
    left = np.array([0 + 0j, 1 + 2j], dtype=np.complex128)
    right = np.array([1 + 0j, 0 + 0j], dtype=np.complex128)
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

    expected_bill = 4 * get_weight("logical_and") * get_dtype_rate("complex128") * 2.0
    assert np.array_equal(direct, expected_direct)
    assert np.array_equal(outer, expected_outer)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "complex128"


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


@pytest.mark.parametrize("name", ("add", "hypot"))
def test_explicit_narrow_dtype_bills_wider_out_for_direct_and_outer(name):
    load_weights()
    left = np.arange(1, 9, dtype=np.int8)
    right = np.arange(8, 0, -1, dtype=np.int8)
    direct_out = np.empty((8, 8), dtype=np.float64)
    outer_out = np.empty((8, 8), dtype=np.float64)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: getattr(fnp, name)(
                fnp.asarray(left[:, None]),
                fnp.asarray(right[None, :]),
                dtype=np.float16,
                out=direct_out,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: getattr(np, name).outer(
                    fnp.asarray(left),
                    fnp.asarray(right),
                    dtype=np.float16,
                    out=outer_out,
                ),
            )

    expected_bill = 64 * get_weight(name) * get_dtype_rate("float64")
    assert direct is direct_out
    assert outer is outer_out
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "float64"


@pytest.mark.parametrize(
    ("name", "dtype"),
    (("hypot", np.float16), ("bitwise_and", np.int8)),
)
@pytest.mark.parametrize("spelling", ("direct", "outer"))
def test_explicit_real_loop_with_complex_out_bills_store_only_factor(
    name, dtype, spelling
):
    load_weights()
    left = np.array([1, 2], dtype=dtype)
    right = np.array([3, 4], dtype=dtype)
    np_func = getattr(np, name)
    output_shape = (2,) if spelling == "direct" else (2, 2)
    expected_out = np.empty(output_shape, dtype=np.complex64)
    out = np.empty(output_shape, dtype=np.complex64)

    if spelling == "direct":
        expected = np_func(left, right, dtype=dtype, out=expected_out)

        def call():
            return getattr(fnp, name)(
                fnp.asarray(left), fnp.asarray(right), dtype=dtype, out=out
            )

    else:
        expected = np_func.outer(left, right, dtype=dtype, out=expected_out)

        def call():
            return np_func.outer(
                fnp.asarray(left), fnp.asarray(right), dtype=dtype, out=out
            )

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual, bill, resolved_dtype = _delta(ctx, call)

    expected_bill = out.size * get_weight(name) * get_dtype_rate("complex64") * 2.0
    assert expected is expected_out
    assert actual is out
    assert np.array_equal(actual, expected)
    assert bill == expected_bill
    assert resolved_dtype == "complex64"


@pytest.mark.parametrize("spelling", ("direct", "outer"))
def test_binary_out_subclass_uses_underlying_dtype_for_billing(spelling):
    load_weights()

    class LiesAboutDtype(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(np.float16)

    left = np.array([3.0, 5.0], dtype=np.float32)
    right = np.array([4.0, 12.0], dtype=np.float32)
    output_shape = (2,) if spelling == "direct" else (2, 2)
    out = np.empty(output_shape, dtype=np.complex128).view(LiesAboutDtype)
    expected_out = np.empty(output_shape, dtype=np.complex128)
    if spelling == "direct":
        expected = np.hypot(left, right, out=expected_out)
    else:
        expected = np.hypot.outer(left, right, out=expected_out)

    def call():
        if spelling == "direct":
            return fnp.hypot(fnp.asarray(left), fnp.asarray(right), out=out)  # pyright: ignore[reportArgumentType]
        return np.hypot.outer(fnp.asarray(left), fnp.asarray(right), out=out)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual, bill, resolved_dtype = _delta(ctx, call)

    expected_bill = out.size * get_weight("hypot") * get_dtype_rate("complex128") * 2.0
    assert actual is out
    assert np.asarray(out).dtype == np.dtype(np.complex128)
    assert np.array_equal(np.asarray(actual), expected)
    assert bill == expected_bill
    assert resolved_dtype == "complex128"


@pytest.mark.parametrize("spelling", ("direct", "outer"))
def test_binary_billing_never_reads_out_subclass_dtype_override(spelling):
    load_weights()
    reads = 0

    class StatefulLyingDtype(np.ndarray):
        @property
        def dtype(self):
            nonlocal reads
            reads += 1
            return np.dtype(np.float16 if reads % 2 else np.complex128)

    left = np.array([3.0, 5.0], dtype=np.float32)
    right = np.array([4.0, 12.0], dtype=np.float32)
    output_shape = (2,) if spelling == "direct" else (2, 2)
    out = np.empty(output_shape, dtype=np.complex128).view(StatefulLyingDtype)
    expected_out = np.empty(output_shape, dtype=np.complex128)
    if spelling == "direct":
        expected = np.hypot(left, right, out=expected_out)
    else:
        expected = np.hypot.outer(left, right, out=expected_out)

    def call():
        if spelling == "direct":
            return fnp.hypot(fnp.asarray(left), fnp.asarray(right), out=out)  # pyright: ignore[reportArgumentType]
        return np.hypot.outer(fnp.asarray(left), fnp.asarray(right), out=out)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual, bill, resolved_dtype = _delta(ctx, call)

    expected_bill = out.size * get_weight("hypot") * get_dtype_rate("complex128") * 2.0
    assert reads == 0
    assert actual is out
    assert np.asarray(out).dtype == np.dtype(np.complex128)
    assert np.array_equal(np.asarray(actual), expected)
    assert bill == expected_bill
    assert resolved_dtype == "complex128"


def test_direct_refreshes_out_billing_view_after_operand_dtype_resolution():
    load_weights()
    dtype_reads = 0

    class OwningDestination(np.ndarray):
        def __new__(cls):
            return super().__new__(cls, (8,), dtype=np.float16)

    out = OwningDestination()

    class MutatingOperandDtype(np.ndarray):
        @property
        def dtype(self):
            nonlocal dtype_reads
            dtype_reads += 1
            if dtype_reads == 1:
                out.resize(16, refcheck=False)
                out.dtype = np.complex128  # pyright: ignore[reportAttributeAccessIssue]
            return np.dtype(np.float32)

    left = np.array([3.0, 5.0], dtype=np.float32).view(MutatingOperandDtype)
    right = np.array([4.0, 12.0], dtype=np.float32)
    expected = np.hypot(np.asarray(left), right).astype(np.complex128)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        actual, bill, resolved_dtype = _delta(
            ctx,
            lambda: fnp.hypot(left, right, out=out),  # pyright: ignore[reportArgumentType]
        )

    expected_bill = 2 * get_weight("hypot") * get_dtype_rate("complex128") * 2.0
    assert dtype_reads > 0
    assert actual is out
    assert np.asarray(out).dtype == np.dtype(np.complex128)
    assert out.shape == (2,)
    assert np.array_equal(np.asarray(actual), expected)
    assert bill == expected_bill
    assert resolved_dtype == "complex128"


@pytest.mark.parametrize("name", ("add", "hypot"))
def test_explicit_narrow_dtype_without_out_stays_narrow_for_direct_and_outer(name):
    load_weights()
    left = np.arange(1, 9, dtype=np.int8)
    right = np.arange(8, 0, -1, dtype=np.int8)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        direct, direct_bill, direct_dtype = _delta(
            ctx,
            lambda: getattr(fnp, name)(
                fnp.asarray(left[:, None]),
                fnp.asarray(right[None, :]),
                dtype=np.float16,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outer, outer_bill, outer_dtype = _delta(
                ctx,
                lambda: getattr(np, name).outer(
                    fnp.asarray(left), fnp.asarray(right), dtype=np.float16
                ),
            )

    expected_bill = 64 * get_weight(name) * get_dtype_rate("float16")
    assert np.array_equal(direct, outer, equal_nan=True)
    assert direct_bill == outer_bill == expected_bill
    assert direct_dtype == outer_dtype == "float16"


@pytest.mark.parametrize("spelling", ("direct", "outer"))
def test_explicit_dtype_object_is_resolved_once_for_binary_spellings(spelling):
    load_weights()
    reads = 0

    class StatefulDtype:
        @property
        def dtype(self):
            nonlocal reads
            reads += 1
            return np.dtype(np.float16 if reads == 1 else np.float64)

    left = np.array([1, 2], dtype=np.int8)
    right = np.array([3, 4], dtype=np.int8)
    dtype = StatefulDtype()
    if spelling == "direct":
        expected = np.add(left[:, None], right[None, :], dtype=np.float16)
    else:
        expected = np.add.outer(left, right, dtype=np.float16)

    def call():
        if spelling == "direct":
            return fnp.add(
                fnp.asarray(left[:, None]),
                fnp.asarray(right[None, :]),
                dtype=dtype,
            )
        return np.add.outer(fnp.asarray(left), fnp.asarray(right), dtype=dtype)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual, bill, resolved_dtype = _delta(ctx, call)

    expected_bill = 4 * get_weight("add") * get_dtype_rate("float16")
    assert reads == 1
    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype == np.dtype(np.float16)
    assert bill == expected_bill
    assert resolved_dtype == "float16"


def test_direct_dtype_resolution_precedes_operand_shape_snapshot():
    load_weights()

    class OwningFloat64(np.ndarray):
        def __new__(cls, values):
            obj = super().__new__(cls, (len(values),), dtype=np.float64)
            obj[...] = values
            return obj

    values_after_resize = np.arange(1.0, 7.0)
    left = OwningFloat64(values_after_resize[:2])
    reads = 0

    class ResizingDtype:
        @property
        def dtype(self):
            nonlocal reads
            reads += 1
            if reads == 1:
                left.resize(values_after_resize.size, refcheck=False)
                left[...] = values_after_resize
            return np.dtype(np.float64)

    right = fnp.asarray(np.array([10.0], dtype=np.float64))
    expected = np.add(values_after_resize, np.asarray(right), dtype=np.float64)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        actual, bill, resolved_dtype = _delta(
            ctx, lambda: fnp.add(left, right, dtype=ResizingDtype())
        )

    expected_bill = (
        values_after_resize.size * get_weight("add") * get_dtype_rate("float64")
    )
    assert reads == 1
    assert left.shape == actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    assert bill == expected_bill
    assert resolved_dtype == "float64"


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
