"""A result the server cannot deliver must not be charged for.

The server pre-charges an operation and then executes it, so any result it
cannot describe has already cost the caller. These tests pin the two ways that
is avoided: making the result deliverable, and refusing it before dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session


@pytest.fixture()
def handler():
    session = Session(flop_budget=10**15)
    yield RequestHandler(session)
    session.close()


def test_complex_scalar_is_delivered_as_a_handle(handler):
    packed = handler._pack_result(np.complex128(1 + 2j))
    assert packed["status"] == "ok"
    assert "id" in packed["result"], packed
    assert packed["result"]["dtype"] == "complex128"
    assert packed["result"]["shape"] == []


def test_complex_scalar_handle_holds_the_right_value(handler):
    packed = handler._pack_result(np.complex64(3 - 4j))
    stored = handler._session.get_array(packed["result"]["id"])
    assert stored.dtype == np.complex64
    assert stored.shape == ()
    assert stored == np.complex64(3 - 4j)


def test_native_scalars_still_come_back_by_value(handler):
    packed = handler._pack_result(np.float64(2.5))
    assert packed["result"] == {"value": 2.5, "dtype": "float64"}
    assert "id" not in packed["result"]
