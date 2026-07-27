"""The op resolver must not expose flopscope's internals as callable ops.

``_get_flopscope_func`` walks ``getattr`` over ``flopscope.numpy`` and then
``flopscope``, so without a scope check any private helper reachable from those
namespaces becomes a remotely callable op. That matters for billing: helpers
such as ``_array_ops._wrap_constant_fill`` mint a symmetry tag -- a claim the
cost model prices on -- without going through a counted wrapper, so a caller
could tag arbitrary asymmetric data for 0 FLOPs and collect the symmetric-rate
discount on everything downstream.
"""

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler, _get_flopscope_func
from flopscope_server._session import Session


@pytest.fixture
def handler():
    session = Session(flop_budget=int(1e12))
    yield RequestHandler(session)
    if session.is_open:
        session.close()


@pytest.mark.parametrize(
    "op",
    [
        "_array_ops._wrap_constant_fill",  # 0-FLOP symmetry-tag mint
        "_array_ops._np.matmul",  # raw numpy module
        "_symmetry_utils.wrap_with_trusted_symmetry",
        "SymmetricTensor",  # tag mint via the class itself
        # Public, but not ops: configure mutates process-global settings that
        # steer later cost decisions (symmetry budgets, einsum path cache).
        "configure",
        "BudgetContext",
    ],
)
def test_non_op_names_are_not_callable(op):
    with pytest.raises(AttributeError):
        _get_flopscope_func(op)


def test_public_ops_still_resolve():
    for op in ("matmul", "einsum", "linalg.inv", "fft.fft", "random.default_rng"):
        assert callable(_get_flopscope_func(op))


def test_registered_symmetry_ops_still_resolve():
    """as_symmetric and symmetrize are counted ops -- as_symmetric validates the
    data and charges for it before tagging, so it is not a free mint."""
    for op in ("as_symmetric", "symmetrize"):
        assert callable(_get_flopscope_func(op))


def test_rng_constructors_still_resolve():
    for op in ("random.RandomState", "random.SeedSequence"):
        assert callable(_get_flopscope_func(op))


def test_tag_mint_op_is_refused_over_the_wire(handler):
    a = np.random.default_rng(0).random((8, 8))
    created = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [8, 8],
            "dtype": "float64",
        }
    )
    handle = created["result"]["id"]
    resp = handler.handle(
        {"op": "_array_ops._wrap_constant_fill", "args": [{"__handle__": handle}]}
    )
    assert resp["status"] == "error"
