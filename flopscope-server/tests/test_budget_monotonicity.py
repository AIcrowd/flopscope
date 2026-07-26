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
