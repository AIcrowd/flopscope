"""Server-side tests for round-2 bugfixes."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler, _decode_index_key
from flopscope_server._server import _normalize_arg
from flopscope_server._session import Session

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


# =========================================================================
# Fix 2: flop_multiplier server support removed (tests live in test_budget_integrity.py)
# =========================================================================

# =========================================================================
# Fix 3: __getitem__ server handling
# =========================================================================


class TestFix3ServerDecodeKey:
    def test_decode_int(self):
        assert _decode_index_key(3) == 3

    def test_decode_slice(self):
        result = _decode_index_key({"__slice__": [0, 5, 2]})
        assert result == slice(0, 5, 2)

    def test_decode_slice_with_none(self):
        result = _decode_index_key({"__slice__": [None, 3, None]})
        assert result == slice(None, 3, None)

    def test_decode_tuple_of_slices(self):
        raw = [{"__slice__": [1, 3, None]}, 0]
        result = _decode_index_key(raw)
        assert result == (slice(1, 3, None), 0)


class TestFix3ServerGetitem:
    def test_getitem_integer(self, handler, session):
        handle = session.store_array(np.array([10.0, 20.0, 30.0]))
        resp = handler.handle(
            {
                "op": "__getitem__",
                "args": [{"__handle__": handle}, 1],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        assert resp["result"]["value"] == 20.0

    def test_getitem_slice(self, handler, session):
        handle = session.store_array(np.arange(10, dtype=np.float64))
        resp = handler.handle(
            {
                "op": "__getitem__",
                "args": [{"__handle__": handle}, {"__slice__": [2, 5, None]}],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        assert "id" in resp["result"]
        assert resp["result"]["shape"] == [3]

    def test_getitem_2d_row(self, handler, session):
        handle = session.store_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        resp = handler.handle(
            {
                "op": "__getitem__",
                "args": [{"__handle__": handle}, 0],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        assert "id" in resp["result"]
        assert resp["result"]["shape"] == [2]


# =========================================================================
# Fix 8: _pack_result includes dtype for scalars
# =========================================================================


class TestFix8ScalarDtype:
    def test_numpy_bool_dtype(self, handler):
        result = handler._pack_result(np.bool_(True))
        assert result["result"]["dtype"] == "bool"
        assert result["result"]["value"] is True

    def test_numpy_float64_dtype(self, handler):
        result = handler._pack_result(np.float64(3.14))
        assert result["result"]["dtype"] == "float64"

    def test_numpy_int32_dtype(self, handler):
        result = handler._pack_result(np.int32(42))
        assert result["result"]["dtype"] == "int32"

    def test_python_bool_dtype(self, handler):
        result = handler._pack_result(True)
        assert result["result"]["dtype"] == "bool"

    def test_python_int_dtype(self, handler):
        result = handler._pack_result(42)
        assert result["result"]["dtype"] == "int64"

    def test_python_float_dtype(self, handler):
        result = handler._pack_result(3.14)
        assert result["result"]["dtype"] == "float64"

    def test_tuple_items_include_dtype(self, handler):
        result = handler._pack_result((np.bool_(True), np.float32(1.5)))
        items = result["result"]["multi"]
        assert items[0]["dtype"] == "bool"
        assert items[1]["dtype"] == "float32"


# =========================================================================
# Fix 9: _normalize_arg recursive for dict values
# =========================================================================


class TestFix9NormalizeArgRecursive:
    def test_dict_with_nested_list_preserves_binary(self):
        # Binary values stay bytes through recursion (keys decode for legacy bytes
        # keys; under raw=False keys already arrive as str).
        result = _normalize_arg({b"key": [b"12345678"]})
        assert result == {"key": [b"12345678"]}

    def test_dict_with_nested_dict_preserves_binary(self):
        result = _normalize_arg({b"outer": {b"inner": b"\x00\x01\x02"}})
        assert result == {"outer": {"inner": b"\x00\x01\x02"}}

    def test_dict_with_bytes_value_long(self):
        binary = bytes(range(256)) * 2
        result = _normalize_arg({b"data": binary})
        assert result["data"] is binary

    def test_dict_with_short_printable_bytes_value_preserved(self):
        # The old content heuristic decoded short all-printable byte VALUES to str;
        # they are now preserved as binary (the create_from_data fix).
        result = _normalize_arg({b"dtype": b"12345678"})
        assert result == {"dtype": b"12345678"}
