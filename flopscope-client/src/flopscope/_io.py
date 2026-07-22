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
cost.

`savez`/`savez_compressed`'s archive MEMBER NAMES (the kwargs keys, plus
"__meta__" when a meta block is present) are billed the same way too: those
strings are written into the .npz archive and read back verbatim by `load`,
exactly like array data, and a member name can be tens of thousands of bytes
long. `_write_npz` concatenates them into one blob, ingests it to get a
server handle, and includes it in the handles billed via
`_bill_save_on_server` -- mirroring the root package's `_savez_name_bytes`.
Skipping this let a participant smuggle data through many tiny arrays given
huge names instead of through the array values, at a near-zero,
size-independent cost.

`savez`/`savez_compressed` accept arrays positionally as well as by keyword,
matching `numpy.savez`: positional arrays are stored under auto-generated
member names `arr_0`, `arr_1`, ... in call order (see `_savez_merge_args`),
keyword arrays keep their given names, and a keyword name that collides with
a positional array's auto-generated name raises the same `ValueError` numpy
raises -- checked before any dispatch or billing happens, so a call numpy
would reject is never billed. A positional array bills identically to the
same array passed under its auto-generated keyword name."""

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


def _savez_merge_args(args: tuple[Any, ...], arrays: dict[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword arrays into a single name->value mapping,
    mirroring ``numpy.savez``'s own naming (``numpy.lib._npyio_impl._savez``):
    positional arrays are named ``arr_0``, ``arr_1``, ... in call order and
    appended after the keyword arrays, which keep their given names and
    order.

    Raises the same ``ValueError`` numpy raises when a keyword name collides
    with a positional array's auto-generated name -- e.g. a positional array
    together with ``arr_0=...``. This is checked here, against the ORIGINAL
    keyword names only, before any dispatch/billing work happens, so a call
    numpy would reject is never billed (positional auto-names can never
    collide with each other, only with a pre-existing keyword name, so a
    single membership check per positional index is equivalent to numpy's
    incremental insert-and-check loop).
    """
    for i in range(len(args)):
        key = f"arr_{i}"
        if key in arrays:
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
    merged = dict(arrays)
    for i, val in enumerate(args):
        merged[f"arr_{i}"] = val
    return merged


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
    # Archive MEMBER NAMES are written to disk and read back verbatim by
    # `load`, exactly like array data (see module docstring) -- concatenate
    # them into one blob and ingest it the same way as the __meta__ blob
    # above, so the server sums their byte length into the egress cost too.
    # Mirrors the root package's `flopscope._io._savez_name_bytes`.
    member_names = list(arrays.keys()) + ([_META_KEY] if meta is not None else [])
    names_blob = "".join(member_names).encode("utf-8")
    if names_blob:
        billed_values.append(_ingest("uint8", (len(names_blob),), names_blob))
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
def savez(file: str, *args: Any, **arrays: Any) -> None:
    """Arrays given positionally are stored under auto-generated names
    ``arr_0``, ``arr_1``, ...; arrays given as keywords keep their given
    names (matching ``numpy.savez``). A keyword name that collides with a
    positional array's auto-generated name raises the same ``ValueError``
    numpy raises -- see ``_savez_merge_args``."""
    _write_npz(file, _savez_merge_args(args, arrays), compressed=False, op_name="savez")


@timed_dispatch
def savez_compressed(file: str, *args: Any, **arrays: Any) -> None:
    """Arrays given positionally are stored under auto-generated names
    ``arr_0``, ``arr_1``, ...; arrays given as keywords keep their given
    names (matching ``numpy.savez_compressed``). A keyword name that
    collides with a positional array's auto-generated name raises the same
    ``ValueError`` numpy raises -- see ``_savez_merge_args``."""
    _write_npz(
        file,
        _savez_merge_args(args, arrays),
        compressed=True,
        op_name="savez_compressed",
    )
