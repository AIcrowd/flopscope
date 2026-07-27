"""Tests for in-process ↔ remote (client/server) divergence feedback."""

from __future__ import annotations

import numpy as np

import flopscope as flops

_CALLBACK_OPS = frozenset(
    {
        "apply_along_axis",
        "apply_over_axes",
        "fromfunction",
        "fromiter",
        "mask_indices",
        "piecewise",
    }
)


def test_remote_unsupported_ops_returns_frozenset_of_callback_ops():
    ops = flops.remote_unsupported_ops()
    assert isinstance(ops, frozenset)
    assert ops == _CALLBACK_OPS


def test_remote_unsupported_ops_matches_registry_flag():
    from flopscope._registry import REGISTRY

    assert flops.remote_unsupported_ops() == frozenset(
        name for name, entry in REGISTRY.items() if entry.get("local_callback")
    )


import warnings

import pytest

import flopscope.numpy as fnp
from flopscope.errors import ConfigureNoOpWarning, RemoteCallbackWarning


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


def test_configure_warns_it_is_a_noop_remotely():
    # configure() works in-process but is a no-op on the client/eval servers;
    # it warns so participants don't expect it to affect a graded submission.
    with pytest.warns(ConfigureNoOpWarning):
        flops.configure(symmetry_warnings=True)
