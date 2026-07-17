"""Server-side decoding of tagged list index keys.

The client encodes genuine Python list keys as ``{"__list__": [...]}`` so the
server can distinguish ``x[[0, 1]]`` (row selection) from ``x[0, 1]``
(element access, sent as a bare msgpack list). Both str and bytes dict keys
must decode, matching the existing ``__slice__``/``__ellipsis__`` handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler, _decode_index_key
from flopscope_server._session import Session


@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


class TestDecodeListKeyModuleLevel:
    def test_tagged_list_decodes_to_list(self):
        assert _decode_index_key({"__list__": [0, 1]}) == [0, 1]

    def test_tagged_list_bytes_key(self):
        assert _decode_index_key({b"__list__": [0, 1]}) == [0, 1]

    def test_nested_tagged_list(self):
        raw = {"__list__": [{"__list__": [0, 1]}, {"__list__": [1, 0]}]}
        assert _decode_index_key(raw) == [[0, 1], [1, 0]]

    def test_bare_list_still_decodes_to_tuple(self):
        # Tuple keys stay bare lists on the wire (backward compatibility).
        assert _decode_index_key([0, 1]) == (0, 1)

    def test_tuple_containing_tagged_list(self):
        raw = [{"__slice__": [None, None, None]}, {"__list__": [0, 1]}]
        assert _decode_index_key(raw) == (slice(None), [0, 1])


class TestDecodeListKeyInstanceMethod:
    def test_tagged_list_decodes_to_list(self, handler):
        assert handler._decode_index_key({"__list__": [0, 1]}) == [0, 1]

    def test_tagged_list_bytes_key(self, handler):
        assert handler._decode_index_key({b"__list__": [0, 1]}) == [0, 1]

    def test_getitem_with_tagged_list_selects_rows(self, handler, session):
        handle = session.store_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        resp = handler.handle(
            {
                "op": "__getitem__",
                "args": [{"__handle__": handle}, {"__list__": [0, 1]}],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        # x[[0, 1]] on a 2x2 array selects both rows -> shape (2, 2).
        # The buggy heuristic decoded it as x[0, 1] -> scalar 2.0.
        assert resp["result"]["shape"] == [2, 2]
        result = session.get_array(resp["result"]["id"])
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]
