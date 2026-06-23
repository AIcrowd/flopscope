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


def test_bitwise_and_parity():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    assert (a & b).tolist() == [1, 0, 0]


def test_bitwise_or_parity():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    assert (a | b).tolist() == [1, 1, 1]


def test_bitwise_xor_parity():
    a, b = fnp.array([1, 0, 1]), fnp.array([1, 1, 0])
    assert (a ^ b).tolist() == [0, 1, 1]


def test_bitwise_invert_parity():
    a = fnp.array([0, 1, 2])
    assert (~a).tolist() == [-1, -2, -3]


def test_left_shift_parity():
    a, b = fnp.array([1, 2, 3]), fnp.array([1, 1, 1])
    assert (a << b).tolist() == [2, 4, 6]


def test_right_shift_parity():
    a, b = fnp.array([2, 4, 6]), fnp.array([1, 1, 1])
    assert (a >> b).tolist() == [1, 2, 3]


# --- ndarray methods missing on RemoteArray (audit P2) ---


def test_argsort_method_parity():
    a = fnp.array([3.0, 1.0, 2.0])
    assert a.argsort().tolist() == [1, 2, 0]


def test_diagonal_method_parity():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a.diagonal().tolist() == [1.0, 4.0]


# --- dtype= constructor parity (audit P1) ---
# String aliases work; Python/numpy TYPE OBJECTS do not.


def test_dtype_string_alias_works():
    # Positive control: the string-alias path is already fine on the client.
    out = fnp.array([1, 2, 3], dtype="float64")
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_dtype_type_object_parity():
    # rc3: dtype= now accepts Python type-objects (float/int) — parity with native.
    assert fnp.array([1, 2, 3], dtype=float).tolist() == [1.0, 2.0, 3.0]
    assert fnp.array([1, 2, 3], dtype=int).tolist() == [1, 2, 3]


# --- flopscope.numpy is now a package (audit P1, ~7 subs) ---


def test_flopscope_numpy_is_a_package():
    import flopscope.numpy.linalg as fnl

    assert hasattr(fnl, "svd")


# --- scalar conversion: NO LONGER a gap on main (audit's __float__ bug) ---


def test_float_of_reduced_scalar_works_now():
    # The audit's struct.unpack scalar-conversion bug does not reproduce on the
    # current client for this path; lock that in so a regression is caught.
    a = fnp.array([1.0, 2.0, 3.0])
    assert float(fnp.max(a)) == 3.0


def test_raw_buffer_pointers_raise_clear_error():
    a = fnp.array([1.0, 2.0, 3.0])
    for attr in ("data", "ctypes", "__array_interface__", "__array_struct__"):
        with pytest.raises(AttributeError, match="remote"):
            getattr(a, attr)


# --- in-place operators: immutable, parity with native (Codex review) ---
# Native FlopscopeArray raises on `a += b`; without explicit __i* on the client,
# Python would fall back to __add__ and silently rebind `a`, so `a += b` would
# work on the eval client while raising locally. The client must raise too.


def test_inplace_add_raises_on_client():
    a, b = fnp.array([1.0, 2.0, 3.0]), fnp.array([1.0, 1.0, 1.0])
    with pytest.raises(TypeError, match="immutable"):
        a += b


def test_inplace_mul_raises_on_client():
    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="immutable"):
        a *= fnp.array([2.0, 2.0, 2.0])


# --- __complex__ preserves complex values (Codex review) ---
# Must not route through float(self): float() raises for a genuinely complex
# size-1 array, unlike NumPy's complex(arr).


def test_complex_of_real_scalar_works():
    assert complex(fnp.array([3.0])) == (3 + 0j)


def test_complex_preserves_complex_value():
    try:
        a = fnp.array([1 + 2j])
    except Exception:
        pytest.skip("client/server does not support complex dtype")
    assert complex(a) == (1 + 2j)
