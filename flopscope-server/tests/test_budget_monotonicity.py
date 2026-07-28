"""The security invariant: nothing may give FLOP budget back.

A charged-then-undeliverable result is fixed by not doing the work, never by
crediting the caller. These tests pin that a refusal is free and that a genuine
computation failure is still charged exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server import _array_store
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session


@pytest.fixture()
def session():
    s = Session(flop_budget=10**15)
    yield s
    s.close()


def test_a_full_store_refuses_before_charging(session, monkeypatch):
    handler = RequestHandler(session)
    a = session.store_array(np.ones((256, 256), dtype="float64"))
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", 1)
    before = session.budget_context.flops_used

    response = handler.handle(
        {"op": "matmul", "args": [{"__handle__": a}, {"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert session.budget_context.flops_used == before, (
        "a refusal must be free: the work was never delivered"
    )


def test_a_multi_array_result_refuses_before_charging(session, monkeypatch):
    """linalg.eigh stores two handles (eigenvalues + eigenvectors) for one
    call. With only one array-store slot free, the whole request must be
    refused before dispatch -- not run, charged, and left with one handle
    stranded when the second store hits the limit mid-loop.
    """
    handler = RequestHandler(session)
    a = session.store_array(np.eye(32, dtype="float64"))
    # array_count is now 1; MAX_ARRAY_COUNT=2 means exactly one slot free.
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", 2)
    before_flops = session.budget_context.flops_used
    before_count = session.array_count

    response = handler.handle(
        {"op": "linalg.eigh", "args": [{"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert session.budget_context.flops_used == before_flops, (
        "a refusal must be free even when the result needs multiple handles"
    )
    assert session.array_count == before_count, (
        "a refusal must leave nothing behind, including a partial multi-array store"
    )


def test_a_shape_dependent_arity_op_is_charged_but_never_strands_a_handle(
    session, monkeypatch
):
    """nonzero returns one array per dimension of its input -- an arity the
    static _FIXED_MULTI_HANDLE_COUNTS table does not (and by design should
    not) enumerate, since it depends on the input's ndim rather than the op
    name alone. The pre-dispatch gate therefore reserves only 1 slot for it,
    same as any untabulated op.

    With exactly one free slot, dispatch proceeds, the op runs and is
    charged, and a 2-D input needs 2 handles. The exact count computed in
    _pack_result before the first store must still refuse and leave the
    store untouched -- the second, table-independent layer is what prevents
    the first handle from being stranded here, not the pre-dispatch table.
    Unlike the free pre-dispatch refusal, this one is NOT free: the op
    already ran, so it is still charged. That charge is the accepted
    trade-off, not a bug -- assert it explicitly so it stays pinned.
    """
    handler = RequestHandler(session)
    a = session.store_array(np.array([[0, 1], [1, 0]], dtype="float64"))
    # array_count is now 1; MAX_ARRAY_COUNT=2 means exactly one slot free.
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", 2)
    before_flops = session.budget_context.flops_used
    before_count = session.array_count

    response = handler.handle(
        {"op": "nonzero", "args": [{"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert session.array_count == before_count, (
        "the exact-count check in _pack_result must refuse before storing "
        "anything, leaving no handle stranded"
    )
    assert session.budget_context.flops_used > before_flops, (
        "this refusal is not free: the op already ran and was charged "
        "before _pack_result could know its exact handle count -- only the "
        "pre-dispatch table (which does not cover shape-dependent arity "
        "ops like nonzero) can make a refusal free"
    )


def test_a_genuine_failure_is_still_charged(session):
    handler = RequestHandler(session)
    singular = session.store_array(np.zeros((64, 64), dtype="float64"))
    before = session.budget_context.flops_used

    response = handler.handle(
        {"op": "linalg.inv", "args": [{"__handle__": singular}], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert session.budget_context.flops_used > before, (
        "genuine computation failures are charged, unchanged by this work"
    )


def test_flops_used_never_decreases(session):
    handler = RequestHandler(session)
    a = session.store_array(np.ones((32, 32), dtype="complex128"))
    seen = [session.budget_context.flops_used]
    for op in ("linalg.det", "trace", "matmul", "linalg.inv", "sum"):
        args = [{"__handle__": a}]
        if op == "matmul":
            args.append({"__handle__": a})
        handler.handle({"op": op, "args": args, "kwargs": {}})
        seen.append(session.budget_context.flops_used)
    assert seen == sorted(seen), f"budget decreased somewhere: {seen}"


def test_no_op_record_carries_a_negative_flop_cost(session, monkeypatch):
    """The aggregate flops_used never decreasing is not the same guarantee as
    no individual record being negative -- a previously-shipped bug billed
    zero-sized contractions a negative flop_cost that still summed to a
    monotonic total. Pin the per-record shape directly, across a sequence
    that includes a capacity refusal.
    """
    handler = RequestHandler(session)
    a = session.store_array(np.ones((32, 32), dtype="complex128"))
    for op in ("linalg.det", "trace", "matmul", "linalg.inv", "sum"):
        args = [{"__handle__": a}]
        if op == "matmul":
            args.append({"__handle__": a})
        handler.handle({"op": op, "args": args, "kwargs": {}})

    # A refusal in the mix: it must not append a negative (or any) record.
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", session.array_count)
    handler.handle(
        {"op": "matmul", "args": [{"__handle__": a}, {"__handle__": a}], "kwargs": {}}
    )

    op_log = session.budget_context.op_log
    assert op_log, "expected at least one recorded op"
    assert all(record.flop_cost >= 0 for record in op_log), (
        f"a negative flop_cost slipped into the op log: {op_log}"
    )


def test_a_full_store_still_serves_ops_that_mint_no_array_handle(session, monkeypatch):
    # The pre-dispatch gate reserves a slot per prospective handle, which
    # would refuse an op that needs none. `random.default_rng` returns a
    # Generator, and _pack_result puts that in the connection's separate
    # generator store -- the array store being full says nothing about
    # whether this response can be delivered, so it must not be refused.
    session.store_array(np.ones(4, dtype="float64"))
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", 1)

    response = RequestHandler(session).handle(
        {"op": "random.default_rng", "args": [7], "kwargs": {}}
    )

    assert response["status"] == "ok", response.get("message")
    assert "gen_id" in response["result"]


def test_a_full_store_refuses_an_op_whose_handle_count_is_argument_dependent(
    session, monkeypatch
):
    # The other side of the exemption: `sum` is NOT exempt, because its
    # handle count cannot be read off the op name. `sum(a)` returns a
    # msgpack-native scalar and stores nothing, but `sum(a, axis=0)` returns
    # an array and stores one, so the gate stays conservative for it. The
    # cost of that conservatism is a refusal the server could have served --
    # but the refusal is free, which is the property being protected here.
    a = session.store_array(np.ones((4, 4), dtype="float64"))
    monkeypatch.setattr(_array_store, "MAX_ARRAY_COUNT", 1)
    before = session.budget_context.flops_used

    response = RequestHandler(session).handle(
        {"op": "sum", "args": [{"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert response["error_type"] == "MemoryError"
    assert session.budget_context.flops_used == before


def test_an_unrepresentable_operand_is_refused_before_any_charge(session):
    # A dict that survives _resolve_arg names nothing the protocol defines,
    # so NumPy can only coerce it to an `object` array -- a dtype the client
    # has no decoder for. Running the kernel would charge for a result that
    # could never be sent back, so the refusal has to happen before dispatch.
    handler = RequestHandler(session)
    v = session.store_array(np.array([True, False, True]))
    before = session.budget_context.flops_used

    response = handler.handle(
        {"op": "where", "args": [{"__handle__": v}, {"k": 1}, 0.0], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert response["error_type"] == "TypeError"
    assert "dict" in response["message"]
    assert session.budget_context.flops_used == before, (
        "an undeliverable operand must cost the caller nothing"
    )


def test_an_unrepresentable_operand_is_found_inside_a_sequence(session):
    # concatenate([a, {...}]) hides the dict one level down, where
    # _resolve_arg recurses -- so the check has to recurse too, or the
    # nested case falls through to the kernel and gets charged.
    handler = RequestHandler(session)
    a = session.store_array(np.ones(3, dtype="float64"))
    before = session.budget_context.flops_used

    response = handler.handle(
        {"op": "concatenate", "args": [[{"__handle__": a}, {"k": 1}]], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert response["error_type"] == "TypeError"
    assert session.budget_context.flops_used == before


def test_handle_envelopes_are_not_mistaken_for_unrepresentable_operands(session):
    # The check runs on *resolved* arguments precisely so that the protocol's
    # own dict envelopes -- which arrive as dicts and leave as arrays -- are
    # never caught by it. Without that ordering this refuses every op.
    handler = RequestHandler(session)
    a = session.store_array(np.ones((4, 4), dtype="float64"))

    response = handler.handle(
        {"op": "matmul", "args": [{"__handle__": a}, {"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "ok", response.get("message")
