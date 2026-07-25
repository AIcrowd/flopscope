"""Meta-tests: prove the comparator can SEE each dimension.

A parity harness that silently detects nothing is worse than none, because it
manufactures confidence. Each test below breaks exactly one dimension and
asserts the comparator reports exactly that dimension.
"""

from __future__ import annotations

from tests.parity.compare import compare_observations
from tests.parity.observe import observe_exception, observe_result


def _returned(**overrides) -> dict:
    base = observe_result(1.0, flops=10)
    base.update(overrides)
    return base


def _dims(inproc: dict, client: dict) -> set[str]:
    return {d.dimension for d in compare_observations("x/y", inproc, client)}


def test_identical_observations_produce_no_divergence():
    assert compare_observations("x/y", _returned(), _returned()) == []


def test_detects_outcome():
    inproc = _returned()
    client = observe_exception(TypeError("nope"), flops=10)
    assert _dims(inproc, client) == {"outcome"}


def test_detects_value():
    assert _dims(_returned(), _returned(value="f:different")) == {"value"}


def test_detects_dtype():
    assert _dims(_returned(dtype="float32"), _returned(dtype="float64")) == {"dtype"}


def test_detects_shape():
    assert _dims(_returned(shape=[2, 3]), _returned(shape=[6])) == {"shape"}


def test_detects_container():
    assert _dims(_returned(container="tuple"), _returned(container="list")) == {
        "container"
    }


def test_detects_pytype():
    assert _dims(_returned(pytype="bool"), _returned(pytype="RemoteScalar")) == {
        "pytype"
    }


def test_detects_flops():
    assert _dims(_returned(flops=3198), _returned(flops=0)) == {"flops"}


def test_detects_exception_class():
    inproc = observe_exception(IndexError("a"), flops=1)
    client = observe_exception(RuntimeError("b"), flops=1)
    dims = _dims(inproc, client)
    assert "exc_type" in dims


def test_ignores_exception_message_differences():
    inproc = observe_exception(IndexError("index 10 is out of bounds"), flops=1)
    client = observe_exception(IndexError("out of range"), flops=1)
    assert _dims(inproc, client) == set()


def test_flops_compared_even_when_both_raise():
    # The billing family: both raise, but only one charged for it.
    inproc = observe_exception(TypeError("x"), flops=0)
    client = observe_exception(TypeError("x"), flops=341648)
    assert _dims(inproc, client) == {"flops"}


def test_value_dimensions_skipped_when_both_raise():
    inproc = observe_exception(TypeError("x"), flops=0)
    client = observe_exception(TypeError("x"), flops=0)
    assert _dims(inproc, client) == set()


def test_reports_multiple_dimensions_independently():
    inproc = _returned(dtype="float32", flops=10)
    client = _returned(dtype="float64", flops=20)
    assert _dims(inproc, client) == {"dtype", "flops"}


def test_divergence_carries_both_sides():
    (div,) = compare_observations("x/y", _returned(dtype="a"), _returned(dtype="b"))
    assert div.case_id == "x/y"
    assert div.dimension == "dtype"
    assert div.inproc == "a"
    assert div.client == "b"
