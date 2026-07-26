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
