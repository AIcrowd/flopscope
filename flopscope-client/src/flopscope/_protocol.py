"""Client-side message encoding/decoding using msgpack."""

from __future__ import annotations

import msgpack

AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY = "authoritative_budget_summary_v1"


def encode_request(op: str, args=None, kwargs=None) -> bytes:
    """Msgpack-encode ``{"op": op, "args": args, "kwargs": kwargs}``."""
    return msgpack.packb(
        {"op": op, "args": args, "kwargs": kwargs},
        use_bin_type=True,
    )


def encode_hello(client_version: str) -> bytes:
    """Encode a `hello` op carrying the client's leading X.Y.Z version."""
    return encode_request("hello", kwargs={"client_version": client_version})


def encode_create_from_data(data: bytes, shape: list, dtype: str) -> bytes:
    """Encode a create_from_data request."""
    return encode_request("create_from_data", args=[data, shape, dtype])


def encode_budget_open(flop_budget: int) -> bytes:
    """Encode a budget_open request."""
    return encode_request("budget_open", kwargs={"flop_budget": flop_budget})


def encode_budget_close() -> bytes:
    """Encode a budget_close request."""
    return encode_request("budget_close")


def encode_budget_status() -> bytes:
    """Encode a budget_status request."""
    return encode_request("budget_status")


def encode_fetch(handle_id: str) -> bytes:
    """Encode a fetch request."""
    return encode_request("fetch", kwargs={"handle_id": handle_id})


def encode_free(handles: list[str]) -> bytes:
    """Encode a free request."""
    return encode_request("free", kwargs={"handles": handles})


def decode_response(raw: bytes) -> dict:
    """Decode a msgpack response using msgpack's native bin/str distinction.

    The server packs every field with ``use_bin_type=True`` -- string fields
    (status, dtype, handle IDs, error messages) as msgpack ``str`` and raw array
    buffers as msgpack ``bin``.  Decoding with ``raw=False`` therefore yields
    ``str`` for strings and ``bytes`` for binary natively.  We must NOT guess
    types from content: a short, all-printable-ASCII array buffer (e.g.
    ``struct.pack('<d', x) == b'12345678'``) is indistinguishable from a string
    by content, and the old heuristic mis-decoded it to ``str`` -- breaking
    ``__float__``/``__int__``/``tolist`` (see test_fetch_bytes_decode).
    """
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)
