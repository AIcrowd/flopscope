"""Client file I/O — reads/writes locally via the stdlib codec, then moves only
inert numeric buffers over the existing free `create_from_data` ingress and
`_fetch_data` egress. The server never sees a path and never deserializes."""

from __future__ import annotations

import struct
from typing import Any

from flopscope import _codec
from flopscope._connection import get_connection
from flopscope._protocol import encode_create_from_data
from flopscope._remote_array import (
    _DTYPE_INFO,
    RemoteArray,
    _result_from_response,
)

_META_KEY = "__meta__"


def _ingest(dtype: str, shape: tuple, buffer: bytes) -> RemoteArray:
    conn = get_connection()
    resp = conn.send_recv(encode_create_from_data(buffer, list(shape), dtype))
    return _result_from_response(resp)


def _flatten_list(obj):
    if not isinstance(obj, (list, tuple)):
        return [obj], ()
    if len(obj) == 0:
        return [], (0,)
    first, inner = _flatten_list(obj[0])
    flat = list(first)
    for item in obj[1:]:
        f2, _s2 = _flatten_list(item)
        flat.extend(f2)
    return flat, (len(obj),) + inner


def _as_triple(val: Any) -> tuple[str, tuple, bytes]:
    """Return (dtype, shape, buffer) for a RemoteArray or a plain list/scalar."""
    if isinstance(val, RemoteArray):
        data, shape, dtype = val._fetch_data()
        return dtype, tuple(shape), data
    flat, shape = _flatten_list(val)
    dtype = "float64"
    fmt = _DTYPE_INFO[dtype][0]
    return dtype, shape, struct.pack(f"<{len(flat)}{fmt}", *[float(x) for x in flat])


def load(file: str) -> Any:
    """Load .npy/.npz. Returns a RemoteArray, or {name: RemoteArray, __meta__}."""
    with open(file, "rb") as fh:
        blob = fh.read()
    if file.endswith(".npz"):
        arrays, meta = _codec.read_npz(blob)
        out: dict[str, Any] = {
            k: _ingest(dt, sh, buf) for k, (dt, sh, buf) in arrays.items()
        }
        if meta is not None:
            out[_META_KEY] = meta
        return out
    dtype, shape, data = _codec.read_npy(blob)
    return _ingest(dtype, shape, data)


def save(file: str, arr: Any) -> None:
    dtype, shape, buf = _as_triple(arr)
    with open(file, "wb") as fh:
        fh.write(_codec.write_npy(dtype, shape, buf))


def _write_npz(file: str, arrays: dict, compressed: bool) -> None:
    meta = arrays.pop(_META_KEY, None)
    if meta is not None and not isinstance(meta, dict):
        raise ValueError(f"'{_META_KEY}' must be a JSON-serializable dict")
    triples = {}
    for key, val in arrays.items():
        if key == _META_KEY:
            raise ValueError(f"'{_META_KEY}' is a reserved array name")
        triples[key] = _as_triple(val)
    blob = _codec.write_npz(triples, meta=meta, compressed=compressed)
    with open(file, "wb") as fh:
        fh.write(blob)


def savez(file: str, **arrays: Any) -> None:
    _write_npz(file, arrays, compressed=False)


def savez_compressed(file: str, **arrays: Any) -> None:
    _write_npz(file, arrays, compressed=True)
