"""Tests for in-process ↔ remote (client/server) divergence feedback."""

from __future__ import annotations

import numpy as np

import flopscope as flops
from flopscope._registry import REGISTRY

# Derived from REGISTRY, never hand-maintained: a literal copy of this set is
# exactly how `mask_indices` went missing for so long (its `local_callback`
# flag was added to the registry but the copy here wasn't updated to match).
# Deriving it means the dispatch table in `_call()` below is the only thing
# that still needs a manual entry per op.
_CALLBACK_OPS = frozenset(
    name for name, entry in REGISTRY.items() if entry.get("local_callback")
)


def test_remote_unsupported_ops_returns_frozenset():
    ops = flops.remote_unsupported_ops()
    assert isinstance(ops, frozenset)


def test_remote_unsupported_ops_matches_registry_flag():
    assert flops.remote_unsupported_ops() == frozenset(
        name for name, entry in REGISTRY.items() if entry.get("local_callback")
    )


import warnings

import pytest

import flopscope.numpy as fnp
from flopscope.errors import ConfigureNoOpWarning, RemoteCallbackWarning


class _NumericUfuncDuck:
    """A successful numeric NEP 13 participant for local-warning tests."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls += 1
        inputs = tuple(self.values if value is self else value for value in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)


class _ForeignIndexArray(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise AssertionError("indices must not dispatch __array_ufunc__")


class _NestedProtocolArray(np.ndarray):
    """An ndarray protocol participant hidden inside a sequence operand."""

    calls = 0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        type(self).calls += 1
        return NotImplemented


def _call(op: str) -> None:
    if op == "apply_along_axis":
        fnp.apply_along_axis(lambda r: r.sum(), 0, fnp.ones((3, 3)))
    elif op == "apply_over_axes":
        fnp.apply_over_axes(lambda a, ax: a.sum(axis=ax), fnp.ones((3, 3)), [0])
    elif op == "piecewise":
        fnp.piecewise(
            fnp.array([-2.0, 2.0]),
            [fnp.array([True, False]), fnp.array([False, True])],
            [lambda v: -v, lambda v: v],
        )
    elif op == "fromfunction":
        fnp.fromfunction(lambda i, j: i + j, (3, 3))
    elif op == "fromiter":
        fnp.fromiter((x for x in range(5)), dtype=float)
    elif op == "mask_indices":
        fnp.mask_indices(3, np.triu)
    else:  # pragma: no cover
        raise AssertionError(op)


@pytest.mark.parametrize("op", sorted(_CALLBACK_OPS))
def test_callback_op_warns_in_process(op):
    with flops.BudgetContext(flop_budget=10**9):
        with pytest.warns(RemoteCallbackWarning):
            _call(op)


def test_callback_warning_suppressed_by_config():
    flops.configure(callback_warnings=False)
    try:
        with flops.BudgetContext(flop_budget=10**9):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RemoteCallbackWarning)
                _call("fromfunction")
    finally:
        flops.configure(callback_warnings=True)


def test_non_callback_op_does_not_warn():
    with flops.BudgetContext(flop_budget=10**9):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RemoteCallbackWarning)
            fnp.matmul(fnp.ones((3, 3)), fnp.ones((3, 3)))


def test_numeric_ufunc_protocol_warns_locally():
    duck = _NumericUfuncDuck([3.0, 4.0])
    with flops.BudgetContext(flop_budget=10**9):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RemoteCallbackWarning)
            result = fnp.add(fnp.array([1.0, 2.0]), duck)
    np.testing.assert_array_equal(result, [4.0, 6.0])
    assert duck.calls == 1
    assert [warning.category for warning in caught].count(RemoteCallbackWarning) == 1


def test_numeric_ufunc_protocol_warning_respects_configuration():
    duck = _NumericUfuncDuck([3.0, 4.0])
    flops.configure(callback_warnings=False)
    try:
        with flops.BudgetContext(flop_budget=10**9):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RemoteCallbackWarning)
                fnp.add(fnp.array([1.0, 2.0]), duck)
    finally:
        flops.configure(callback_warnings=True)
    assert duck.calls == 1


@pytest.mark.parametrize(
    "op_name, invoke",
    [
        ("abs", lambda duck: fnp.abs(duck)),
        ("atan", lambda duck: fnp.atan(duck)),
        ("acos", lambda duck: fnp.acos(duck)),
        ("true_divide", lambda duck: fnp.true_divide(fnp.array([1.0, 2.0]), duck)),
    ],
)
def test_ufunc_protocol_warning_uses_charged_alias_name(op_name, invoke):
    duck = _NumericUfuncDuck([3.0, 4.0])
    with flops.BudgetContext(flop_budget=10**9):
        with pytest.warns(RemoteCallbackWarning, match=rf"^{op_name}\(\)"):
            invoke(duck)
    assert duck.calls == 1


@pytest.mark.parametrize(
    "operand, expect_error",
    [
        (np.array([3.0, 4.0]), None),
        (np.array([3.0, 4.0]).view(type("Inherited", (np.ndarray,), {})), None),
        (
            np.array([3.0, 4.0]).view(
                type("ProtocolOptOut", (np.ndarray,), {"__array_ufunc__": None})
            ),
            TypeError,
        ),
    ],
    ids=["plain-ndarray", "inherited-ndarray-protocol", "protocol-opt-out"],
)
def test_non_foreign_ufunc_protocol_operands_do_not_warn(operand, expect_error):
    with flops.BudgetContext(flop_budget=10**9):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RemoteCallbackWarning)
            if expect_error is None:
                fnp.add(fnp.array([1.0, 2.0]), operand)
            else:
                with pytest.raises(expect_error):
                    fnp.add(fnp.array([1.0, 2.0]), operand)


def test_at_indices_protocol_is_not_treated_as_a_ufunc_callback_operand():
    indices = np.array([0, 1]).view(_ForeignIndexArray)
    target = fnp.zeros(2)
    with flops.BudgetContext(flop_budget=10**9):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RemoteCallbackWarning)
            np.add.at(target, indices, [3.0, 4.0])
    np.testing.assert_array_equal(target, [3.0, 4.0])


def test_nested_sequence_protocol_does_not_warn_or_dispatch():
    nested = np.array([3.0, 4.0]).view(_NestedProtocolArray)
    expected = np.add(np.array([[1.0, 2.0]]), [nested])
    _NestedProtocolArray.calls = 0
    with flops.BudgetContext(flop_budget=10**9):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RemoteCallbackWarning)
            actual = fnp.add(fnp.array([[1.0, 2.0]]), [nested])

    np.testing.assert_array_equal(actual, expected)
    assert _NestedProtocolArray.calls == 0


def test_configure_warns_it_is_a_noop_remotely():
    # configure() works in-process but is a no-op on the client/eval servers;
    # it warns so participants don't expect it to affect a graded submission.
    with pytest.warns(ConfigureNoOpWarning):
        flops.configure(symmetry_warnings=True)
