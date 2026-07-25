"""Unit tests for fingerprinting and observation records. Pure — no backend."""

from __future__ import annotations

import math

from tests.parity.observe import (
    fingerprint,
    observe_exception,
    observe_result,
    observe_timeout,
    observe_worker_died,
)


def test_distinguishes_int_float_and_bool():
    assert fingerprint(1) != fingerprint(1.0)
    assert fingerprint(1) != fingerprint(True)
    assert fingerprint(1.0) != fingerprint(True)


def test_distinguishes_signed_zero():
    assert fingerprint(0.0) != fingerprint(-0.0)


def test_nan_fingerprints_equal_to_itself():
    # float('nan') != float('nan'), but their bit patterns match, which is what
    # we need: two backends both returning NaN must compare equal.
    assert fingerprint(float("nan")) == fingerprint(float("nan"))


def test_distinguishes_last_ulp():
    # This is Family 1: allclose passes these, we must not.
    a = 0.30000000447034836
    b = 0.30000001192092896
    assert math.isclose(a, b, rel_tol=1e-7)  # allclose would accept
    assert fingerprint(a) != fingerprint(b)  # we do not


def test_fingerprints_nested_lists_elementwise():
    assert fingerprint([[1.0, 2.0]]) == fingerprint([[1.0, 2.0]])
    assert fingerprint([[1.0, 2.0]]) != fingerprint([[1.0, 2.0000000000000004]])


def test_distinguishes_complex():
    assert fingerprint(complex(1, 2)) != fingerprint(complex(1, 3))
    assert fingerprint(complex(1, 0)) != fingerprint(1.0)


def test_observe_result_records_type_dtype_shape_container():
    class FakeArray:
        dtype = "float32"
        shape = (2, 3)

        def tolist(self):
            return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    obs = observe_result(FakeArray(), flops=42)
    assert obs["outcome"] == "returned"
    assert obs["pytype"] == "FakeArray"
    assert obs["dtype"] == "float32"
    assert obs["shape"] == [2, 3]
    assert obs["container"] == "array"
    assert obs["flops"] == 42
    assert obs["value"] == fingerprint([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_observe_result_detects_namedtuple_container():
    import collections

    Pair = collections.namedtuple("Pair", "U S")
    obs = observe_result(Pair(1, 2), flops=0)
    assert obs["container"] == "namedtuple:Pair(U,S)"


def test_observe_result_distinguishes_tuple_from_list():
    assert observe_result((1, 2), flops=0)["container"] == "tuple"
    assert observe_result([1, 2], flops=0)["container"] == "list"


def test_observe_exception_records_class_and_builtin_ancestry():
    obs = observe_exception(IndexError("out of bounds"), flops=7)
    assert obs["outcome"] == "raised"
    assert obs["exc_type"] == "IndexError"
    assert "LookupError" in obs["exc_bases"]
    assert obs["flops"] == 7
    assert obs["exc_msg"] == "out of bounds"


def test_observe_exception_records_only_builtin_ancestors():
    class Custom(ValueError):
        pass

    obs = observe_exception(Custom("x"), flops=0)
    assert obs["exc_type"] == "Custom"
    assert "ValueError" in obs["exc_bases"]
    assert "Custom" not in obs["exc_bases"]


def test_terminal_outcomes():
    assert observe_timeout(flops=3)["outcome"] == "timeout"
    assert observe_worker_died()["outcome"] == "worker_died"


def test_container_fingerprints_do_not_collide_on_separator_chars():
    # Structural separators appearing inside string content must not make two
    # different values fingerprint identically.
    assert fingerprint(["s:x", "y"]) != fingerprint(["s:x,s:y"])
    assert fingerprint(["a", "b"]) != fingerprint(["a,b"])
    assert fingerprint([""]) != fingerprint([])


def test_materialize_unwraps_item_style_scalars():
    class ItemScalar:
        def item(self):
            return 1.5

    assert observe_result(ItemScalar(), flops=0)["value"] == fingerprint(1.5)


def test_array_wrapper_classes_do_not_diverge_on_pytype():
    # inproc's `FlopscopeArray` and the client's `RemoteArray` are two
    # different classes by construction (two implementations of the same
    # array type); raw class names would report a spurious divergence on
    # every single array-returning case. They must collapse to one token.
    class FlopscopeArray:
        dtype = "float32"
        shape = ()

        def tolist(self):
            return 1.0

    class RemoteArray:
        dtype = "float32"
        shape = ()

        def tolist(self):
            return 1.0

    inproc = observe_result(FlopscopeArray(), flops=0)
    client = observe_result(RemoteArray(), flops=0)
    assert inproc["pytype"] == client["pytype"]


def test_plain_bool_against_a_scalar_wrapper_still_diverges_on_pytype():
    # A plain Python bool on one backend against a wrapper object on the
    # other means a value failed to unwrap the way it should have — this
    # is the real defect class "pytype" exists to catch, and the array-
    # wrapper collapse above must not blind the harness to it.
    class RemoteScalar:
        dtype = "bool"
        shape = ()

        def tolist(self):
            return True

    inproc = observe_result(True, flops=0)
    client = observe_result(RemoteScalar(), flops=0)
    assert inproc["pytype"] != client["pytype"]
    assert inproc["pytype"] == "bool"
