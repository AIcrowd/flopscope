"""Lock in the client-vs-native gaps the 2026-06-23 prod audit found.

These are OPERATOR / METHOD / NAMESPACE gaps on a real ``RemoteArray`` — the kind
the numpy-suite harness does NOT surface (that harness operates on native
ndarrays, so RemoteArray dunders/methods are never exercised). Each test builds a
real client RemoteArray and asserts the gap reproduces on the CURRENT client, so
the inventory is grounded in live behavior rather than the stale prod sample.

These are deliberately written as "the gap currently reproduces" assertions:
when Phase 2 closes a gap, the corresponding ``pytest.raises`` stops raising and
the test FAILS — a loud signal to flip it into a positive parity assertion.

All tests rely on the ambient ``BudgetContext`` (autouse fixture); they must NOT
open their own.
"""

from __future__ import annotations

import pytest

import flopscope as fnp  # the CLIENT

# --- Bitwise / shift operators on RemoteArray (audit's #1, ~112 subs) ---
# Native FlopscopeArray defines __and__/__or__/__xor__/__invert__/__lshift__/
# __rshift__; RemoteArray defines none. The boolean-mask idiom (a > 0) & (b < 1)
# is the dominant failure.


def test_bitwise_and_gap():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a & b


def test_bitwise_or_gap():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a | b


def test_bitwise_xor_gap():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a ^ b


def test_bitwise_invert_gap():
    a = fnp.array([1, 0, 1])
    with pytest.raises(TypeError, match="bad operand type for unary"):
        _ = ~a


def test_left_shift_gap():
    a, b = fnp.array([1, 2, 3]), fnp.array([1, 1, 1])
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a << b


def test_right_shift_gap():
    a, b = fnp.array([2, 4, 6]), fnp.array([1, 1, 1])
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a >> b


# --- ndarray methods missing on RemoteArray (audit P2) ---


def test_argsort_method_gap():
    a = fnp.array([3.0, 1.0, 2.0])
    with pytest.raises(AttributeError, match="argsort"):
        _ = a.argsort()


def test_diagonal_method_gap():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(AttributeError, match="diagonal"):
        _ = a.diagonal()


# --- dtype= constructor parity (audit P1) ---
# String aliases work; Python/numpy TYPE OBJECTS do not.


def test_dtype_string_alias_works():
    # Positive control: the string-alias path is already fine on the client.
    out = fnp.array([1, 2, 3], dtype="float64")
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_dtype_type_object_gap():
    with pytest.raises(TypeError, match="dtype"):
        _ = fnp.array([1, 2, 3], dtype=float)


# --- flopscope.numpy is a module, not a package (audit P1, ~7 subs) ---


def test_flopscope_numpy_submodule_import_gap():
    with pytest.raises(ModuleNotFoundError, match="not a package"):
        __import__("flopscope.numpy.linalg")


# --- scalar conversion: NO LONGER a gap on main (audit's __float__ bug) ---


def test_float_of_reduced_scalar_works_now():
    # The audit's struct.unpack scalar-conversion bug does not reproduce on the
    # current client for this path; lock that in so a regression is caught.
    a = fnp.array([1.0, 2.0, 3.0])
    assert float(fnp.max(a)) == 3.0
