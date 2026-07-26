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


def test_unencodable_result_is_refused_with_a_named_error(handler):
    class Opaque:
        pass

    packed = handler._pack_result(Opaque())
    assert packed["status"] == "error"
    assert packed["error_type"] == "UnsupportedReturnType"
    assert "Opaque" in packed["message"]


def test_undecodable_dtype_is_refused_not_stored(handler):
    # np.longdouble is not usable here: on some platforms (e.g. Apple Silicon)
    # it has no extended precision and reports dtype "float64", which IS
    # client-decodable, so it wouldn't exercise the refusal path at all.
    # datetime64 has no client-side decoder on any platform, so it reliably
    # takes the refusal branch everywhere.
    before = len(handler._session._conn.arrays._arrays)
    packed = handler._pack_result(np.datetime64("2024-01-01"))
    assert packed["status"] == "error"
    assert packed["error_type"] == "UnsupportedReturnType"
    # Refused means refused: nothing was minted for the caller to fetch.
    assert len(handler._session._conn.arrays._arrays) == before


def test_tuple_with_complex_elements_are_delivered_as_handles(handler):
    packed = handler._pack_result((np.complex128(1 + 2j), np.float64(3.5)))
    assert packed["status"] == "ok"
    items = packed["result"]["multi"]
    assert "id" in items[0]
    assert items[0]["dtype"] == "complex128"
    assert items[0]["shape"] == []
    stored = handler._session.get_array(items[0]["id"])
    assert stored == np.complex128(1 + 2j)
    assert items[1] == {"value": 3.5, "dtype": "float64"}


def test_tuple_with_undecodable_element_is_refused_not_stored(handler):
    before = len(handler._session._conn.arrays._arrays)
    packed = handler._pack_result((np.datetime64("2024-01-01"), np.float64(1.0)))
    assert packed["status"] == "error"
    assert packed["error_type"] == "UnsupportedReturnType"
    assert len(handler._session._conn.arrays._arrays) == before


def test_tuple_with_array_then_undecodable_element_leaks_no_handle(handler):
    # The array element would normally be stored before the later element is
    # even inspected; the whole result must still be refused with no handle
    # left over from the array.
    before = len(handler._session._conn.arrays._arrays)
    packed = handler._pack_result((np.arange(3.0), np.datetime64("2024-01-01")))
    assert packed["status"] == "error"
    assert packed["error_type"] == "UnsupportedReturnType"
    assert len(handler._session._conn.arrays._arrays) == before
