"""Native flopscope features must be reachable + behave through the client.

One test per public native feature the 2026-06-23 prod audit saw participants
use. These do NOT go through the numpy-suite harness (which only exercises the
patched numpy *functions* against native ndarrays); they call the CLIENT's
public flopscope API directly, so feature-level gaps — a native feature missing
or broken on the client — surface here.

No xfails on real gaps: a missing-but-native feature should FAIL loudly (that is
the Phase-1 measurement signal). Only behaviour that is intentionally
client-unavailable is asserted as such.

All tests rely on the ambient ``BudgetContext`` opened by the autouse
``_fresh_connection_and_budget`` fixture (the client rejects nested contexts);
they must NOT open their own.
"""

from __future__ import annotations

import pytest

import flopscope as fnp  # the CLIENT (conftest puts flopscope-client/src first)


def _sym2():
    """A 2-axis symmetric group for 2-D test matrices."""
    return fnp.SymmetryGroup.symmetric(axes=(0, 1))


def test_symmetrize_through_client():
    out = fnp.symmetrize(fnp.array([[1.0, 2.0], [0.0, 1.0]]), symmetry=_sym2())
    # The client returns a RemoteArray proxy for the symmetrized tensor.
    assert out.tolist() is not None


def test_as_symmetric_through_client():
    out = fnp.as_symmetric(fnp.array([[1.0, 2.0], [2.0, 1.0]]), symmetry=_sym2())
    assert out.tolist() is not None


def test_is_symmetric_through_client():
    out = fnp.is_symmetric(fnp.array([[1.0, 2.0], [2.0, 1.0]]), symmetry=_sym2())
    assert bool(out) is True


def test_configure_through_client():
    # configure(**kwargs) -> None; must be callable on the client.
    assert fnp.configure() is None


def test_accounting_namespace_reachable():
    # REAL GAP (audit's CHILD_INTERNAL_ERROR class): flopscope.accounting exists
    # natively but is missing on the client. No xfail — this failure IS the
    # measurement. When Phase 2 adds it to the client, this goes green.
    assert hasattr(fnp, "accounting"), "client missing flopscope.accounting"


def test_accounting_cost_helpers_work_end_to_end():
    # rc3 regression caught by the prod re-grade: flopscope.accounting cost
    # helpers IMPORTED but failed at CALL time with
    # "FlopscopeServerError: unknown op: 'flops.einsum_cost'" — the client
    # proxied einsum_cost/svd_cost to a server op that didn't exist. Exercise
    # the real client->server path so a call-time regression can't slip through
    # (the import-only check above could not catch it).
    ec = fnp.accounting.einsum_cost("ij,jk->ik", [(4, 5), (5, 6)])
    assert isinstance(ec, int) and ec > 0

    svd_full = fnp.accounting.svd_cost(128, 64)  # k=0 -> FULL svd (not "top-0")
    svd_topk = fnp.accounting.svd_cost(128, 64, k=8)
    assert isinstance(svd_full, int) and svd_full > 0
    assert isinstance(svd_topk, int) and svd_topk > 0
    assert svd_full > svd_topk  # full decomposition costs more than rank-8

    # pointwise/reduction were already local; confirm they still work too.
    assert fnp.accounting.pointwise_cost("add", shape=(16, 16)) > 0
    assert fnp.accounting.reduction_cost("sum", input_shape=(16, 16)) > 0


def test_symmetric_tensor_is_intentionally_server_side():
    # NOT a gap: native exposes flopscope.SymmetricTensor, but the client
    # deliberately withholds it (a server-side / analysis API) with a helpful
    # error. Document that contract so it is not mistaken for a parity gap.
    with pytest.raises(AttributeError, match="server-side"):
        _ = fnp.SymmetricTensor


def test_symmetrize_result_carries_symmetry_attribute():
    # Positive control / partial parity: native symmetrize() returns a
    # SymmetricTensor with a .symmetry attribute; the client returns a RemoteArray
    # whose .symmetry DOES forward to the server. This part has parity.
    out = fnp.symmetrize(fnp.array([[1.0, 2.0], [2.0, 1.0]]), symmetry=_sym2())
    assert type(out.symmetry).__name__ == "SymmetryGroup"


def test_symmetrize_result_is_symmetric_method_gap():
    # PARTIAL-PARITY GAP: native SymmetricTensor has .is_symmetric (the type is a
    # numpy.ndarray subclass with extra symmetric API); the client's RemoteArray
    # proxy does NOT forward .is_symmetric. A participant who develops against
    # native symmetrize().is_symmetric breaks on the client. No xfail — this
    # failure IS the measurement; flip to a positive assert when Phase 2 closes it.
    out = fnp.symmetrize(fnp.array([[1.0, 2.0], [2.0, 1.0]]), symmetry=_sym2())
    assert hasattr(out, "is_symmetric"), (
        "client symmetrize() result missing .is_symmetric (native SymmetricTensor has it)"
    )
