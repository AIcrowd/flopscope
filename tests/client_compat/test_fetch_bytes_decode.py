"""A fetched array whose raw buffer happens to be short + all printable-ASCII
must still decode correctly through float()/int()/tolist().

Prod regression (rc4, 11 subs; e.g. ``estimator.py predict`` ->
``RemoteArray.__float__``): ``TypeError: a bytes-like object is required, not
'str'``. The client response decoder (``_protocol.decode_response``) unpacks
with ``raw=True`` and then a heuristic (``_normalize``) re-decodes any short
(<=32 byte), all-printable-ASCII ``bytes`` value to ``str``. That heuristic
cannot distinguish a string field from a raw array buffer: a size-1 array (the
float()/int() path) serializes to 4-8 bytes, so whenever the value's IEEE-754
bytes all land in ``[32, 128)`` the ``data`` buffer is turned into a ``str`` and
``_bytes_to_list``'s ``struct.unpack`` blows up.

The server packs ``data`` as a msgpack ``bin`` field (``use_bin_type=True``), so
the bin/str distinction is already on the wire -- the client just discards it.
"""

from __future__ import annotations

import struct

import flopscope as fnp

# A float64 whose little-endian IEEE-754 bytes are exactly b"12345678" -- all
# printable ASCII, length 8 (<= the 32-byte heuristic threshold).
_TRIGGER_F64 = struct.unpack("<d", b"12345678")[0]


def test_float_on_allprintable_buffer():
    """float() on a size-1 array with an all-printable buffer (the prod path)."""
    arr = fnp.array([_TRIGGER_F64])
    assert float(arr) == _TRIGGER_F64


def test_int_on_allprintable_buffer():
    """int() on a size-1 int array whose buffer is printable ASCII."""
    one = fnp.array([49], dtype="int8")  # buffer == b"1"
    assert int(one) == 49


def test_tolist_on_allprintable_buffer():
    """tolist() (the general fetch path) on a multi-element printable buffer."""
    arr = fnp.array([49, 50, 51, 52], dtype="int8")  # buffer == b"1234"
    assert arr.tolist() == [49, 50, 51, 52]


def test_normal_value_unaffected():
    """A value whose buffer has high/zero bytes already works -- proves the bug
    is the heuristic, not the fetch path as a whole."""
    arr = fnp.array([1.0])  # struct.pack('<d', 1.0) has 0x00 and 0xf0 bytes
    assert float(arr) == 1.0
