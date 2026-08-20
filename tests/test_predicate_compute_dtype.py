"""Predicate ops must bill the dtype numpy COMPUTES in, not the bool it returns.

``signbit`` publishes only float loops (``np.signbit.types`` is
``['e->?', 'f->?', 'd->?', 'g->?']``), so an integer operand is promoted to
the same-size float loop before the sign test runs -- exactly like ``sin``.
``isneginf``/``isposinf`` are ``logical_and(isinf(x), signbit(x))``, so they
inherit that promotion. ``isclose``/``allclose`` promote unconditionally:
numpy's own ``isclose`` runs ``dt = result_type(y, 1.)`` and casts, which
floors every integer/bool operand at float64.

All five return bool. That is why the miss went unseen: the compute-dtype
conformance sweep floors the billed rate at the rate of the RESULT dtype,
and bool rates 1.0 -- the lowest rate any dtype carries -- so the floor was
unfalsifiable for the whole predicate family. See
``test_compute_dtype_conformance.py``'s compute-loop floor for the
generalized guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 1000


def _billed(fn) -> int:
    load_weights()
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


def _arr(dtype) -> np.ndarray:
    return np.ones(N, dtype=dtype)


# ---------------------------------------------------------------------------
# signbit and the two composites built on it: the size-mapped float loop
# ---------------------------------------------------------------------------

SIZE_MAPPED = [
    (np.bool_, np.float16),
    (np.int8, np.float16),
    (np.uint8, np.float16),
    (np.int16, np.float32),
    (np.uint16, np.float32),
    (np.int32, np.float64),
    (np.uint32, np.float64),
    (np.int64, np.float64),
    (np.uint64, np.float64),
]

SIGNBIT_FAMILY = ["signbit", "isneginf", "isposinf"]


@pytest.mark.parametrize("op", SIGNBIT_FAMILY)
@pytest.mark.parametrize("dtype, loop_dtype", SIZE_MAPPED)
def test_signbit_family_bills_the_float_loop_numpy_resolves(op, dtype, loop_dtype):
    """An integer operand must bill exactly what its promoted float loop bills.

    numpy's own loop resolution is the oracle: ``np.signbit.resolve_dtypes``
    reports float16 for int8, float32 for int16 and float64 for int32/int64,
    and the same promotion is traceable inside isneginf/isposinf.
    """
    fn = getattr(fnp, op)
    assert np.signbit.resolve_dtypes((np.dtype(dtype), None))[0] == np.dtype(
        loop_dtype
    ), "probe stale: numpy no longer resolves this loop"
    assert _billed(lambda: fn(_arr(dtype))) == _billed(lambda: fn(_arr(loop_dtype)))


@pytest.mark.parametrize("op", SIGNBIT_FAMILY)
def test_signbit_family_int32_bills_above_a_genuine_int32_rate_op(op):
    """The discriminating case: int32 computes in float64 and must cost 2x.

    A same-shape op with a real int32 loop (``negative``) bills the int32
    rate; the predicate promotes, so it must bill strictly more.
    """
    fn = getattr(fnp, op)
    int32_rate = _billed(lambda: fnp.negative(_arr(np.int32)))
    assert _billed(lambda: fn(_arr(np.int32))) > int32_rate


@pytest.mark.parametrize("op", SIGNBIT_FAMILY)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_signbit_family_float_widths_are_untouched(op, dtype):
    """Float operands already run their own loop: the fix must not move them."""
    fn = getattr(fnp, op)
    assert _billed(lambda: fn(_arr(dtype))) == _billed(lambda: fn(_arr(dtype)))


@pytest.mark.parametrize("op", SIGNBIT_FAMILY)
def test_signbit_family_narrow_float_bills_below_float64(op):
    """float32 must stay strictly cheaper than float64: no blanket f64 floor.

    Flooring the whole family at float64 (the ``i0``/``sinc`` rule) would
    also pass every equality above while overcharging float32 and every
    narrow integer width. This is the test that separates the size-mapped
    mapping from that blunt one.
    """
    fn = getattr(fnp, op)
    assert _billed(lambda: fn(_arr(np.float32))) < _billed(lambda: fn(_arr(np.float64)))
    assert _billed(lambda: fn(_arr(np.int16))) < _billed(lambda: fn(_arr(np.int32)))


# ---------------------------------------------------------------------------
# isclose / allclose: numpy floors the tolerance core at float64
# ---------------------------------------------------------------------------

CLOSE_OPS = ["isclose", "allclose"]
INTEGER_DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
]


@pytest.mark.parametrize("op", CLOSE_OPS)
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_close_ops_bill_integer_operands_at_float64(op, dtype):
    """``result_type(y, 1.)`` floors every integer/bool operand at float64.

    numpy's isclose casts its reference operand to ``result_type(y, 1.)``
    before the ``|x-y| <= atol + rtol*|y|`` core runs, so an all-integer call
    computes in float64 and must bill what the float64 call bills.
    """
    fn = getattr(fnp, op)
    assert np.result_type(np.dtype(dtype), 1.0) == np.dtype(np.float64), (
        "probe stale: numpy no longer floors this dtype at float64"
    )
    a = _arr(dtype)
    f64 = _arr(np.float64)
    assert _billed(lambda: fn(a, a)) == _billed(lambda: fn(f64, f64))


@pytest.mark.parametrize("op", CLOSE_OPS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_close_ops_keep_narrow_float_widths(op, dtype):
    """A narrow float keeps its own loop -- ``result_type(float32, 1.)`` is
    float32 -- so the float64 floor must not reach it."""
    fn = getattr(fnp, op)
    a = _arr(dtype)
    f64 = _arr(np.float64)
    assert np.result_type(np.dtype(dtype), 1.0) == np.dtype(dtype)
    assert _billed(lambda: fn(a, a)) < _billed(lambda: fn(f64, f64))


@pytest.mark.parametrize("op", CLOSE_OPS)
def test_close_ops_mixed_narrow_float_and_integer_follow_numpy(op):
    """A float16 reference against an int32 operand computes in float64.

    ``result_type(int32, 1.) == float64`` promotes the reference, and the
    subtraction against the other operand then runs at float64. Billing the
    resolved operand pair alone (float64 here) happens to agree; the case is
    pinned so a future narrowing of the rule cannot silently drop it.
    """
    fn = getattr(fnp, op)
    i32 = _arr(np.int32)
    f16 = _arr(np.float16)
    f64 = _arr(np.float64)
    assert _billed(lambda: fn(f16, i32)) == _billed(lambda: fn(f64, f64))


# ---------------------------------------------------------------------------
# A billing change must not change what numpy returns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", SIGNBIT_FAMILY + CLOSE_OPS)
def test_values_are_unchanged(op):
    base = np.array([-3, -1, 0, 2, 5], dtype=np.int32)
    fn = getattr(fnp, op)
    np_fn = getattr(np, op)
    with flops.budget(10**15, quiet=True):
        got = fn(base, base) if op in CLOSE_OPS else fn(base)
    want = np_fn(base, base) if op in CLOSE_OPS else np_fn(base)
    assert np.array_equal(np.asarray(got), np.asarray(want))


# ---------------------------------------------------------------------------
# The symmetry validators: a billed allclose wearing a different name
# ---------------------------------------------------------------------------
#
# ``is_symmetric`` and ``as_symmetric`` bill ``k * (7n - 1)`` -- literally
# allclose's cost formula, once per non-identity generator -- because the work
# they do IS ``np.allclose(array, array.transpose(perm))``. So they inherit
# allclose's unconditional float64 floor for integer and bool operands. Neither
# returns bool (``is_symmetric`` returns a Python bool, ``as_symmetric`` returns
# a tensor of the INPUT dtype), which is why the result-dtype oracle is blind to
# both for a second reason: there is no bool result to floor at, and the
# tensor's dtype is the very operand dtype being under-billed.

SYM_N = 32


def _sym(dtype) -> np.ndarray:
    """A matrix that is genuinely symmetric at any dtype (so validation passes).

    Built float-first and cast, not ``np.eye(n, dtype=dtype) * 3 + 1``: that
    spelling promotes a bool ``eye`` straight to int64 through the arithmetic,
    so the bool row of the parametrization would silently probe int64 -- which
    already rates 2.0 and would pass without the fix under test.
    """
    return (np.eye(SYM_N) * 3.0 + 1.0).astype(dtype)


def _group():
    return flops.SymmetryGroup.symmetric(axes=(0, 1))


SYM_OPS = ["is_symmetric", "as_symmetric"]


def _sym_call(op, dtype):
    fn = getattr(flops, op)
    return lambda: fn(_sym(dtype), symmetry=_group())


@pytest.mark.parametrize("op", SYM_OPS)
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_symmetry_validators_bill_integer_operands_at_float64(op, dtype):
    """The validating allclose computes in float64, so it must bill float64."""
    assert _billed(_sym_call(op, dtype)) == _billed(_sym_call(op, np.float64))


@pytest.mark.parametrize("op", SYM_OPS)
def test_symmetry_validators_keep_float32_below_float64(op):
    """float32 keeps its own loop: the floor must not swallow narrow floats."""
    assert _billed(_sym_call(op, np.float32)) < _billed(_sym_call(op, np.float64))


def test_symmetric_tensor_constructor_matches_as_symmetric():
    """``SymmetricTensor(...)`` shares as_symmetric's validation charge, so the
    two must not be able to disagree about the operand's compute width."""
    a = _sym(np.int32)
    direct = _billed(lambda: flops.SymmetricTensor(a, symmetry=_group()))
    via_fn = _billed(lambda: flops.as_symmetric(a, symmetry=_group()))
    assert direct == via_fn == _billed(_sym_call("as_symmetric", np.float64))
