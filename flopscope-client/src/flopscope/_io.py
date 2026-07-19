"""Client file I/O — reads/writes locally via the stdlib codec, then moves only
inert numeric buffers over the existing free `create_from_data` ingress and
`_fetch_data` egress. The server never sees a path and never deserializes.

`save`/`savez`/`savez_compressed` additionally round-trip to the server before
writing (see `_bill_save_on_server`): no array data crosses the wire, only
handle references, but the SERVER -- the sole owner of the FLOP budget --
bills the same 4*numel data-egress cost the in-process reference charges.
Without this round-trip the client could write computed results to disk for
free (fetch + local write never touched the budget).

`savez`/`savez_compressed`'s optional `__meta__` dict is billed the same way:
`_write_npz` serializes it to the identical uint8 byte blob the local codec
writes to the archive, ingests that blob to get a server handle (mirroring
`_as_triple`'s treatment of a plain value), and includes it in the handles
billed via `_bill_save_on_server` -- so the server sums its byte length into
the egress cost exactly like a named array. Skipping this let a participant
round-trip unlimited data through `__meta__` for a flat, size-independent
cost."""

from __future__ import annotations

import json
import struct
from typing import Any

from flopscope import _codec
from flopscope._connection import get_connection
from flopscope._dispatch import timed_dispatch
from flopscope._protocol import encode_create_from_data, encode_request
from flopscope._remote_array import (
    _DTYPE_INFO,
    RemoteArray,
    _result_from_response,
)

_META_KEY = "__meta__"


@timed_dispatch
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


@timed_dispatch
def _bill_save_on_server(op: str, values: list[Any]) -> None:
    """Round-trip to the server so it deducts the 4*numel egress cost for
    save/savez/savez_compressed before any local write happens.

    Server-owned counting: only handle references cross the wire, never array
    data, and the client never mutates a budget itself. A ``RemoteArray``
    already has a server handle; a plain list/scalar value has none yet, so
    it is first ingested via the existing free `create_from_data` path
    (mirrors `_as_triple`) to get one, then billed exactly like every other
    save shape.
    """
    handles = []
    for v in values:
        if isinstance(v, RemoteArray):
            handles.append({"__handle__": v.handle_id})
        else:
            ingested = _ingest(*_as_triple(v))
            handles.append({"__handle__": ingested.handle_id})
    get_connection().send_recv(encode_request(op, args=handles))


@timed_dispatch
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


# Overhead attribution: absorb local codec encode + any _fetch_data egress +
# the server-side billing round-trip.
@timed_dispatch
def save(file: str, arr: Any) -> None:
    _bill_save_on_server("save", [arr])
    dtype, shape, buf = _as_triple(arr)
    with open(file, "wb") as fh:
        fh.write(_codec.write_npy(dtype, shape, buf))


def _write_npz(file: str, arrays: dict, compressed: bool, op_name: str) -> None:
    meta = arrays.pop(_META_KEY, None)
    if meta is not None and not isinstance(meta, dict):
        raise ValueError(f"'{_META_KEY}' must be a JSON-serializable dict")
    billed_values = list(arrays.values())
    if meta is not None:
        # Serialize to the identical uint8 blob `_codec.write_npz` will write
        # to the archive (see below) and ingest it to get a server handle, so
        # `_bill_save_on_server` sums its byte length into the egress cost --
        # mirrors the in-process `flopscope._io._prepare`, which does the same
        # json.dumps(...).encode("utf-8") before billing. Routing raw bytes
        # through the generic non-RemoteArray branch of `_bill_save_on_server`
        # would be wrong: `_as_triple` always encodes plain values as float64,
        # which would misprice a uint8 byte blob.
        meta_blob = json.dumps(meta).encode("utf-8")  # raises on non-JSON-safe values
        billed_values.append(_ingest("uint8", (len(meta_blob),), meta_blob))
    _bill_save_on_server(op_name, billed_values)
    triples = {}
    for key, val in arrays.items():
        if key == _META_KEY:
            raise ValueError(f"'{_META_KEY}' is a reserved array name")
        triples[key] = _as_triple(val)
    blob = _codec.write_npz(triples, meta=meta, compressed=compressed)
    with open(file, "wb") as fh:
        fh.write(blob)


@timed_dispatch
def savez(file: str, **arrays: Any) -> None:
    _write_npz(file, arrays, compressed=False, op_name="savez")


@timed_dispatch
def savez_compressed(file: str, **arrays: Any) -> None:
    _write_npz(file, arrays, compressed=True, op_name="savez_compressed")
