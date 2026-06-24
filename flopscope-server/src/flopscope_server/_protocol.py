"""Protocol layer for flopscope server — message encoding/decoding.

Message format: msgpack over ZMQ.

All messages are msgpack-encoded dicts. By default, msgpack returns bytes keys
when raw=False is not used, so decode_request handles normalisation carefully:
  - top-level string fields (op, dtype, request_id) are decoded to str
  - binary payload fields (data) are kept as bytes
"""

from __future__ import annotations

from typing import Any

import msgpack

from flopscope._registry import REGISTRY

# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------

#: Protocol-level ops not in the numpy REGISTRY.
_PROTOCOL_OPS: frozenset[str] = frozenset(
    {
        "hello",
        "budget_open",
        "budget_close",
        "budget_status",
        "fetch",
        "fetch_slice",
        "free",
        "create_from_data",
        "__getitem__",
        "astype",
        # Analytical FLOP cost estimators (flopscope.accounting). Pure functions
        # of shapes — the client proxies these here; the server computes the
        # WEIGHTED cost via native flopscope.accounting (authoritative weights +
        # the single-source einsum cost engine).
        "flops.einsum_cost",
        "flops.svd_cost",
    }
)

#: Full set of permitted op names.
WHITELIST: frozenset[str] = frozenset(REGISTRY.keys()) | _PROTOCOL_OPS

# ---------------------------------------------------------------------------
# String fields that should always be decoded from bytes -> str
# ---------------------------------------------------------------------------

_STRING_FIELDS: frozenset[str] = frozenset({"op", "dtype", "request_id"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidRequestError(Exception):
    """Raised when a client request cannot be decoded or is not permitted."""


# ---------------------------------------------------------------------------
# decode_request
# ---------------------------------------------------------------------------


def decode_request(raw: bytes) -> dict:
    """Decode a msgpack-encoded request from the client.

    Parameters
    ----------
    raw:
        Raw msgpack bytes received from the client.

    Returns
    -------
    dict
        Decoded request with top-level keys as strings.
        Binary payload fields (e.g. ``data``) are kept as :class:`bytes`.

    Raises
    ------
    InvalidRequestError
        If *raw* is empty, cannot be unpacked, or is missing the ``op`` field.
    """
    if not raw:
        raise InvalidRequestError("malformed request: empty bytes")

    try:
        # Use raw=True so that we get bytes keys and can selectively decode.
        msg_raw = msgpack.unpackb(raw, raw=True)
    except Exception as exc:
        raise InvalidRequestError(f"malformed request: {exc}") from exc

    if not isinstance(msg_raw, dict):
        raise InvalidRequestError("malformed request: top-level value must be a dict")

    # Normalise keys and selected string values.
    msg: dict[str, Any] = {}
    for k, v in msg_raw.items():
        # Decode key from bytes -> str
        key: str = k.decode("utf-8") if isinstance(k, bytes) else str(k)

        # Selectively decode known string-valued fields
        if key in _STRING_FIELDS and isinstance(v, bytes):
            value: Any = v.decode("utf-8")
        else:
            value = v

        msg[key] = value

    if "op" not in msg:
        raise InvalidRequestError("malformed request: missing 'op' field")

    return msg


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------


def validate_request(msg: dict) -> None:
    """Check that the op in *msg* is on the permitted whitelist.

    Parameters
    ----------
    msg:
        Decoded request dict (as returned by :func:`decode_request`).

    Raises
    ------
    InvalidRequestError
        If the op name is not in :data:`WHITELIST`.
    """
    op = msg.get("op", "")
    # Generator.<method> ops are dispatched to a server-side RNG handle and are
    # method-whitelisted in the request handler (_ALLOWED_GEN_METHODS).
    if op.startswith("Generator."):
        return
    if op not in WHITELIST:
        raise InvalidRequestError(f"unknown op: {op!r}")


# ---------------------------------------------------------------------------
# encode_response
# ---------------------------------------------------------------------------


def encode_response(result: Any, budget: int, comms_overhead_ns: int) -> bytes:
    """Encode a successful operation response.

    Parameters
    ----------
    result:
        The value returned by the operation (must be msgpack-serialisable).
    budget:
        Remaining budget after the operation.
    comms_overhead_ns:
        Round-trip communications overhead in nanoseconds.

    Returns
    -------
    bytes
        msgpack-encoded response dict with ``status="ok"``.
    """
    payload = {
        "status": "ok",
        "result": result,
        "budget": budget,
        "comms_overhead_ns": comms_overhead_ns,
    }
    return msgpack.packb(payload, use_bin_type=True)


# ---------------------------------------------------------------------------
# encode_error_response
# ---------------------------------------------------------------------------


def encode_error_response(error_type: str, message: str) -> bytes:
    """Encode an error response.

    Parameters
    ----------
    error_type:
        Name of the exception class (e.g. ``"InvalidRequestError"``).
    message:
        Human-readable error description.

    Returns
    -------
    bytes
        msgpack-encoded response dict with ``status="error"``.
    """
    payload = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    return msgpack.packb(payload, use_bin_type=True)


# ---------------------------------------------------------------------------
# encode_fetch_response
# ---------------------------------------------------------------------------


def encode_fetch_response(
    data: bytes,
    shape: tuple[int, ...],
    dtype: str,
    comms_overhead_ns: int,
) -> bytes:
    """Encode a fetch response carrying raw array bytes.

    Parameters
    ----------
    data:
        Raw array bytes (e.g. ``array.tobytes()``).
    shape:
        Array shape as a tuple/list of ints.
    dtype:
        NumPy dtype string (e.g. ``"float32"``).
    comms_overhead_ns:
        Round-trip communications overhead in nanoseconds.

    Returns
    -------
    bytes
        msgpack-encoded response dict with ``status="ok"`` and ``data`` as
        raw bytes.
    """
    payload = {
        "status": "ok",
        "data": data,
        "shape": list(shape),
        "dtype": dtype,
        "comms_overhead_ns": comms_overhead_ns,
    }
    return msgpack.packb(payload, use_bin_type=True)
