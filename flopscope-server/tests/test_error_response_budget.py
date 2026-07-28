"""An error response has to carry the budget, because failing can still cost.

An operation is billed by the kernel that runs it, and only afterwards does the
server try to pack the result -- where it may find the result is too large to
send. The FLOPs are spent by then and cannot come back. The client folds the
``budget`` of every response into the ``flops_used`` its callers read, so an
error response that omitted the budget would leave a caller that caught the
exception reading a value from before the charge.
"""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server import _request_handler
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session


@pytest.fixture()
def session():
    s = Session(flop_budget=10**15)
    yield s
    s.close()


def test_a_billed_then_undeliverable_op_reports_the_advanced_budget(
    session, monkeypatch
):
    handler = RequestHandler(session)
    a = session.store_array(np.ones((256, 256), dtype="float64"))
    # Small enough that the operands stored fine, small enough that the
    # result cannot be returned: the matmul runs and charges, then packing
    # discovers the result exceeds what may be sent.
    monkeypatch.setattr(_request_handler, "MAX_ARRAY_BYTES", 1024)
    before = session.budget_context.flops_used

    response = handler.handle(
        {"op": "matmul", "args": [{"__handle__": a}, {"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "error"
    charged = session.budget_context.flops_used
    assert charged > before, "precondition: this op must actually be billed"
    assert response["budget"]["flops_used"] == charged, (
        "an error response must report the budget as it stands after the charge"
    )


def test_an_error_raised_from_the_dispatch_path_also_reports_the_budget(session):
    # The budget is attached once, around the whole dispatch, rather than at
    # each error return -- so an error raised out of the op itself carries it
    # too, without that path having to remember.
    response = RequestHandler(session).handle(
        {"op": "definitely.not.an.op", "args": [], "kwargs": {}}
    )

    assert response["status"] == "error"
    assert response["budget"]["flops_used"] == session.budget_context.flops_used


def test_a_successful_response_keeps_the_budget_its_own_handler_computed(session):
    # The attachment must only fill a gap, never overwrite: success responses
    # already carry the budget their handler read at pack time.
    a = session.store_array(np.ones((8, 8), dtype="float64"))

    response = RequestHandler(session).handle(
        {"op": "matmul", "args": [{"__handle__": a}, {"__handle__": a}], "kwargs": {}}
    )

    assert response["status"] == "ok"
    assert response["budget"]["flops_used"] == session.budget_context.flops_used
