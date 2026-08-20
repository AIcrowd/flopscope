"""A ufunc call NumPy has no loop for must cost nothing.

The established doctrine, stated in two places in the cost model and enforced
for ``out=`` by ``test_out_arg_wrapped_destination.py``, is that a refusal
decided before any work happens bills ``0``:

- ``cost-model.md``: an invalid ``k`` for ``svd`` "is rejected before any
  billing"; an index reduction's ``out=`` refusal "is decided before the
  reduction runs, so it costs 0".
- #177 item 1, fixed in #241: ``percentile``/``quantile`` charged 208 FLOPs
  for an out-of-range ``q`` they then refused.

Dtype refusals were the remaining hole. ``fnp.bitwise_and(f32, f32)``,
``fnp.ldexp(f32, uint64)`` and ``np.ldexp.reduce(f32)`` all deducted in full
and then raised, because the guard was NumPy's own dispatch, which runs inside
the deduct block. A budget could be drained by calls that did no arithmetic.

The refusal is now taken from ``ufunc.resolve_dtypes`` above the deduct site.
That is the same question NumPy's dispatcher asks, so it raises the identical
exception -- these tests assert the type AND the message text, because
intercepting a failure earlier than NumPy does must not change what the
failure looks like to a caller.
"""

import warnings

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp

N = 100


def _charged(ctx, call):
    """FLOPs deducted by ``call``, and the exception it raised."""
    before = ctx.flops_used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            call()
        except Exception as exc:  # noqa: BLE001 -- the exception IS the subject
            return ctx.flops_used - before, exc
    raise AssertionError("call was expected to raise, but succeeded")


# (op, left dtype, right dtype) -- every one is a pair NumPy has no loop for.
_REFUSED_BINARY = (
    ("bitwise_and", "float32", "float32"),
    ("bitwise_or", "float64", "int32"),
    ("bitwise_xor", "float32", "bool"),
    ("left_shift", "float32", "int32"),
    ("right_shift", "int32", "float64"),
    ("gcd", "float32", "float32"),
    ("lcm", "float64", "float64"),
    ("ldexp", "float32", "float32"),
    ("ldexp", "float32", "uint64"),
    ("subtract", "bool", "bool"),
)


@pytest.mark.parametrize(("name", "left_dtype", "right_dtype"), _REFUSED_BINARY)
def test_a_binary_loop_numpy_lacks_costs_nothing(name, left_dtype, right_dtype):
    left = np.ones(N, dtype=left_dtype)
    right = np.ones(N, dtype=right_dtype)
    # NumPy must really refuse this pair, or the test is measuring nothing.
    with pytest.raises(Exception) as numpy_refusal:  # noqa: B017
        getattr(np, name)(left, right)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        # Operands built BEFORE the measurement: asarray bills on its own.
        x, y = fnp.asarray(left), fnp.asarray(right)
        charged, exc = _charged(ctx, lambda: getattr(fnp, name)(x, y))

    assert charged == 0, f"{name}({left_dtype}, {right_dtype}) billed {charged}"
    assert type(exc) is type(numpy_refusal.value)
    assert str(exc) == str(numpy_refusal.value)


_REFUSED_UNARY = (
    ("invert", "float32"),
    ("bitwise_count", "float32"),
    ("negative", "bool"),
    ("sign", "bool"),
)


@pytest.mark.parametrize(("name", "dtype"), _REFUSED_UNARY)
def test_a_unary_loop_numpy_lacks_costs_nothing(name, dtype):
    operand = np.ones(N, dtype=dtype)
    with pytest.raises(Exception) as numpy_refusal:  # noqa: B017
        getattr(np, name)(operand)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x = fnp.asarray(operand)
        charged, exc = _charged(ctx, lambda: getattr(fnp, name)(x))

    assert charged == 0, f"{name}({dtype}) billed {charged}"
    assert type(exc) is type(numpy_refusal.value)
    assert str(exc) == str(numpy_refusal.value)


@pytest.mark.parametrize("method", ("reduce", "accumulate", "reduceat"))
@pytest.mark.parametrize("name", ("ldexp", "gcd", "bitwise_and"))
def test_a_refused_ufunc_method_costs_nothing(name, method):
    """``reduce``/``accumulate``/``reduceat`` need a same-dtype loop.

    ``ldexp`` is ``(float, int) -> float`` and has none at all, so every one of
    these is refused whatever the operand dtype -- ``gcd``/``bitwise_and`` are
    refused for a float operand. ``gcd.reduce`` billed 1584 FLOPs for a call
    that never ran.
    """
    operand = np.ones(N, dtype=np.float32)
    ufunc = getattr(np, name)
    args = ([0, 4],) if method == "reduceat" else ()
    with pytest.raises(Exception) as numpy_refusal:  # noqa: B017
        getattr(ufunc, method)(operand, *args)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x = fnp.asarray(operand)
        charged, exc = _charged(ctx, lambda: getattr(ufunc, method)(x, *args))

    assert charged == 0, f"{name}.{method} billed {charged}"
    assert type(exc) is type(numpy_refusal.value)


@pytest.mark.parametrize(("name", "left_dtype", "right_dtype"), _REFUSED_BINARY)
def test_a_refused_outer_costs_nothing(name, left_dtype, right_dtype):
    left = np.ones(8, dtype=left_dtype)
    right = np.ones(8, dtype=right_dtype)
    ufunc = getattr(np, name)
    with pytest.raises(Exception):  # noqa: B017
        ufunc.outer(left, right)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x, y = fnp.asarray(left), fnp.asarray(right)
        charged, _ = _charged(ctx, lambda: ufunc.outer(x, y))

    assert charged == 0, f"{name}.outer({left_dtype}, {right_dtype}) billed {charged}"


def test_a_refused_call_does_not_reach_the_op_log():
    """Costing nothing is not the same as billing zero: it must not log an op.

    A zero-FLOP record would still show up in ``summary_dict()`` as an
    operation that ran, and every consumer of the log would count it.
    """
    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x = fnp.asarray(np.ones(N, dtype=np.float32))
        depth = len(ctx.op_log)
        with pytest.raises(TypeError):
            fnp.bitwise_and(x, x)
        assert len(ctx.op_log) == depth


def test_a_supported_call_next_to_a_refused_one_still_bills():
    """The guard must refuse the unsupported pair and nothing else."""
    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        floats = fnp.asarray(np.ones(N, dtype=np.float32))
        ints = fnp.asarray(np.ones(N, dtype=np.int32))
        with pytest.raises(TypeError):
            fnp.bitwise_and(floats, floats)
        before = ctx.flops_used
        fnp.bitwise_and(ints, ints)
        assert ctx.flops_used - before > 0


# ---------------------------------------------------------------------------
# Moving the dtype refusal above the deduct site also moves it above the
# refusals ``deduct`` itself performs. Those have to keep winning. Both are
# TypeError subclasses, so ``except TypeError`` is unaffected either way --
# what changes is the specific class and the message. UnsupportedDtypeError
# names the op and says why the dtype is not billable; NumPy's bare loop error
# says only that no loop matched. Letting the bare error through first swapped
# the class on 1,918 measured operand combinations.
# ---------------------------------------------------------------------------

_COMPLEX_ILLEGAL = ("arctan2", "atan2", "gcd", "lcm", "hypot", "copysign")


@pytest.mark.parametrize("name", _COMPLEX_ILLEGAL)
@pytest.mark.parametrize("spelling", ("direct", "outer", "reduce"))
def test_complex_refusal_outranks_the_numpy_loop_refusal(name, spelling):
    """A complex operand is flopscope's refusal to make, not NumPy's.

    Both cost 0, so nothing is mis-metered either way -- what matters is that
    the error stays inside the shipped hierarchy.
    """
    operand = np.ones(8, dtype=np.complex64)
    ufunc = getattr(np, name)

    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x = fnp.asarray(operand)
        before = ctx.flops_used
        with pytest.raises(f.errors.UnsupportedDtypeError) as caught:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if spelling == "direct":
                    getattr(fnp, name)(x, x)
                elif spelling == "outer":
                    ufunc.outer(x, x)
                else:
                    ufunc.reduce(x)
        assert ctx.flops_used == before

    # The specific class, not merely "some TypeError" -- that is the part a
    # bare NumPy loop error would have taken away.
    assert type(caught.value) is f.errors.UnsupportedDtypeError
    assert "complex" in str(caught.value)
    # The direct spelling is billed under the name it was called by, while the
    # ufunc-method spellings report ``ufunc.__name__`` -- for the array-API
    # aliases those differ, since ``np.atan2 is np.arctan2``.
    expected_name = name if spelling == "direct" else ufunc.__name__
    assert expected_name in str(caught.value)


def test_non_numeric_refusal_outranks_the_numpy_loop_refusal():
    """Same precedence for the non-numeric allowlist, the other deduct guard."""
    with f.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
        x = fnp.asarray(np.ones(8, dtype=np.float32))
        before = ctx.flops_used
        with pytest.raises(f.errors.UnsupportedDtypeError):
            # bitwise_and has no float loop AND a non-numeric operand: the
            # flopscope refusal is the one that must surface.
            fnp.bitwise_and(x, np.array(["a"] * 8))
        assert ctx.flops_used == before
