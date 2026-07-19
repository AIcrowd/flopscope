"""List index keys must be distinguishable from tuple keys on the wire.

``x[[0, 1]]`` (fancy row selection) and ``x[0, 1]`` (element access) are
different numpy operations, but both used to encode as a bare msgpack list.
Genuine list keys now encode as ``{"__list__": [...]}``; tuples keep the
bare-list encoding for backward compatibility.
"""

from __future__ import annotations

from flopscope._remote_array import _encode_index_key


class TestEncodeListKey:
    def test_list_key_is_tagged(self):
        assert _encode_index_key([0, 1]) == {"__list__": [0, 1]}

    def test_single_element_list_is_tagged(self):
        assert _encode_index_key([1]) == {"__list__": [1]}

    def test_nested_list_key(self):
        assert _encode_index_key([[0, 1], [1, 0]]) == {
            "__list__": [{"__list__": [0, 1]}, {"__list__": [1, 0]}]
        }

    def test_tuple_key_stays_bare_list(self):
        assert _encode_index_key((0, 1)) == [0, 1]

    def test_tuple_containing_list(self):
        # x[(slice(None), [0, 1])] — per-axis fancy index inside a tuple
        encoded = _encode_index_key((slice(None), [0, 1]))
        assert encoded == [
            {"__slice__": [None, None, None]},
            {"__list__": [0, 1]},
        ]

    def test_int_passthrough(self):
        assert _encode_index_key(3) == 3
