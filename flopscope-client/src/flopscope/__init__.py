"""flopscope — transparent proxy to a remote flopscope server.

This module exposes a numpy-like API where every operation is dispatched
to a remote server over ZMQ.  Participants use it as::

    import flopscope as flops
    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=1_000_000) as ctx:
        a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = fnp.zeros((2, 2))
        c = fnp.add(a, b)
"""

from __future__ import annotations

import builtins
import struct
from typing import Any

import msgpack

__version__ = "0.12.0"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
from flopscope._budget import (  # noqa: E402
    BudgetContext,
    BudgetSnapshot,
    OpRecord,
    budget,
    budget_reset,
    budget_summary_dict,
    current_budget,
)
from flopscope._config import configure  # noqa: E402,F401
from flopscope._dispatch import timed_dispatch  # noqa: E402
from flopscope._display import budget_live, budget_summary  # noqa: E402
from flopscope._math_compat import e, inf, nan, pi  # noqa: E402
from flopscope._perm_group import SymmetryGroup  # noqa: E402

# ---------------------------------------------------------------------------
# Remote types
# ---------------------------------------------------------------------------
from flopscope._remote_array import (  # noqa: E402
    _DTYPE_INFO,
    RemoteArray,
    RemoteScalar,
    _encode_arg,
    _result_from_response,
)
from flopscope.errors import (  # noqa: E402
    BudgetExhaustedError,
    FlopscopeError,
    FlopscopeServerError,
    FlopscopeWarning,
    NoBudgetContextError,
    RemoteCallbackError,
    RemoteSerializationError,
    SymmetryError,
    SymmetryLossWarning,
    TimeExhaustedError,
    UnauthorizedControlError,
    UnsupportedFunctionError,
    UnsupportedReturnType,
)

# Alias: ``fnp.ndarray`` refers to the RemoteArray class.
ndarray = RemoteArray

# ---------------------------------------------------------------------------
# Connection / protocol (private)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Submodules (imported so ``fnp.linalg``, ``fnp.random``, ``fnp.fft`` work)
# ---------------------------------------------------------------------------
from flopscope import (
    accounting,  # noqa: E402, F401
    fft,  # noqa: E402, F401
    flops,  # noqa: E402, F401
    linalg,  # noqa: E402, F401
    random,  # noqa: E402, F401
    stats,  # noqa: E402, F401
)
from flopscope._connection import get_connection  # noqa: E402
from flopscope._protocol import (  # noqa: E402
    encode_create_from_data,
    encode_request,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from flopscope._registry import (  # noqa: E402
    BLACKLISTED,
    FUNCTION_CATEGORIES,
    get_category,
    is_valid_op,
    iter_proxyable,
)
from flopscope._registry_data import FUNCTION_CATEGORIES as _FC  # noqa: E402
from flopscope._registry_data import LOCAL_CALLBACK_OPS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (no server round-trip needed)
# ---------------------------------------------------------------------------

pi: float = pi
e: float = e
inf: float = inf
nan: float = nan
newaxis = None

# ---------------------------------------------------------------------------
# Dtypes (numpy-free dual-purpose objects: callable constructor + dtype label)
# ---------------------------------------------------------------------------

from flopscope._dtypes import (  # noqa: E402
    _DTYPE_LABELS,
    _normalize_dtype,
    bool_,
    complex64,
    complex128,
    dtype,
    finfo,
    float16,
    float32,
    float64,
    iinfo,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

# ---------------------------------------------------------------------------
# Proxy factory
# ---------------------------------------------------------------------------


# Ordered by how often each type appears as a leaf, not alphabetically: the
# membership test below is a linear scan run once per leaf of every payload, so
# a bulk numeric argument hits `float`/`int` on the first or second compare.
# Order is a pure speed choice — every test is exact identity, so no ordering
# can change the result.
_MSGPACK_OK = (
    builtins.float,
    builtins.int,
    builtins.str,
    builtins.bool,
    builtins.type(None),
    builtins.bytes,
    builtins.bytearray,
    builtins.memoryview,
    msgpack.ExtType,
    msgpack.Timestamp,
)

_MSGPACK_SUBCLASS_BASES = (
    builtins.list,
    builtins.tuple,
    builtins.dict,
    builtins.str,
)


def _is_exact_msgpack_scalar(value: Any) -> bool:
    """Check msgpack scalar support without invoking participant hooks.

    Spelled as a loop rather than ``any(... for ...)``: this runs once per leaf
    of every dispatched payload, and the generator it used to build cost more
    than the comparisons themselves (4x, measured on a plain-float leaf).
    """
    value_type = builtins.type(value)
    for supported_type in _MSGPACK_OK:
        if value_type is supported_type:
            return True
    return False


def _has_msgpack_subclass_base(value: Any) -> bool:
    """Recognize supported subclasses without invoking their protocols."""
    return builtins.any(
        mro_type is supported_type
        for mro_type in builtins.type.__getattribute__(builtins.type(value), "__mro__")
        for supported_type in _MSGPACK_SUBCLASS_BASES
    )


def _static_type_name(value: Any) -> str:
    """Return a class name without consulting a participant metaclass."""
    name = builtins.type.__getattribute__(builtins.type(value), "__name__")
    return name if builtins.type(name) is builtins.str else "object"


def _describe_unserializable(args: Any, kwargs: Any) -> str:
    """Return a short descriptor of the first value msgpack cannot encode,
    e.g. ``"of type 'generator'"``; ``""`` if none can be pinpointed."""

    def walk(value: Any):
        value_type = builtins.type(value)
        if _is_exact_msgpack_scalar(value):
            return None
        if value_type is builtins.list or value_type is builtins.tuple:
            for item in value:
                bad = walk(item)
                if bad is not None:
                    return bad
            return None
        if value_type is builtins.dict:
            for key, item in builtins.dict.items(value):
                bad = walk(key)
                if bad is not None:
                    return bad
                bad = walk(item)
                if bad is not None:
                    return bad
            return None
        if _has_msgpack_subclass_base(value):
            return None
        return _static_type_name(value)

    bad = walk(args)
    if bad is not None:
        return f"of type {bad!r}"
    bad = walk(kwargs)
    if bad is not None:
        return f"of type {bad!r}"
    return ""


def _raise_serialization_error(op_name: str, bad: str = "") -> None:
    """Raise the client-side error for a value rejected before transmission."""
    if op_name in LOCAL_CALLBACK_OPS:
        raise RemoteCallbackError(
            f"{op_name}() requires a Python callback, which the "
            "client/server backend cannot execute remotely. Run it "
            "in the in-process flopscope backend, or precompute the "
            "result."
        )
    detail = f" {bad}" if bad else ""
    raise RemoteSerializationError(
        f"{op_name}() received an argument{detail} that cannot be sent "
        f"to the remote (client/server) backend. Pass a materialized "
        f"array or built-in (list / number / str) instead."
    )


def _make_proxy(op_name: str):
    """Create a proxy function that dispatches *op_name* to the server."""

    def proxy(*args: Any, **kwargs: Any):
        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        # This walk must stay AHEAD of the pack, not deferred into the handler
        # below. It is not only building an error message: it is the last gate
        # before msgpack touches the payload, and msgpack's packer consults
        # `__class__`, which participant code can define as a property. Letting
        # a hostile object reach the packer executes that property inside our
        # dispatch (see tests/test_remote_callback_error.py). The walk decides
        # everything from `type()`, so it refuses such an object without ever
        # running its hooks.
        bad = _describe_unserializable(encoded_args, encoded_kwargs)
        if bad:
            _raise_serialization_error(op_name, bad)
        try:
            request = encode_request(op_name, args=encoded_args, kwargs=encoded_kwargs)
        except (TypeError, ValueError) as exc:
            _raise_serialization_error(op_name, bad)
            raise AssertionError("unreachable") from exc
        resp = get_connection().send_recv(request)
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = op_name
    return timed_dispatch(proxy)


# ---------------------------------------------------------------------------
# Special-case: array()
# ---------------------------------------------------------------------------


def _flatten(obj):
    """Recursively flatten a nested list/tuple and return ``(flat, shape)``."""
    if not isinstance(obj, (list, tuple)):
        return [obj], ()
    if len(obj) == 0:
        return [], (0,)
    first_flat, inner_shape = _flatten(obj[0])
    flat = list(first_flat)
    for item in obj[1:]:
        item_flat, item_shape = _flatten(item)
        if item_shape != inner_shape:
            raise ValueError(
                f"Inhomogeneous shape: expected inner shape {inner_shape}, "
                f"got {item_shape}"
            )
        flat.extend(item_flat)
    return flat, (len(obj),) + inner_shape


def _infer_dtype(values):
    """Infer a dtype string from a list of Python scalars."""
    # Use builtins.any/all to avoid collision with the proxy functions
    # that shadow these names at module level.
    _any = builtins.any
    _all = builtins.all
    has_float = _any(isinstance(v, float) for v in values)
    has_complex = _any(isinstance(v, complex) for v in values)
    if has_complex:
        return "complex128"
    if has_float:
        return "float64"
    if _all(isinstance(v, bool) for v in values):
        return "bool"
    if _all(isinstance(v, int) for v in values):
        return "int64"
    return "float64"  # mixed or float values


# Buffer-protocol formats with a direct wire dtype. Anything else (big-endian
# '>d', structured 'T{...}', complex 'Zd', unicode '1w') has no wire dtype and
# falls through to the TypeError at the end of array().
_BUFFER_FORMAT_TO_WIRE = {
    "f": "float32",
    "d": "float64",
    "e": "float16",
    "b": "int8",
    "B": "uint8",
    "h": "int16",
    "H": "uint16",
    "i": "int32",
    "I": "uint32",
    "l": "int64",
    "L": "uint64",
    "q": "int64",
    "Q": "uint64",
    "?": "bool",
}


@timed_dispatch
def array(object, dtype=None, **kwargs):  # noqa: F811
    """Create a remote array from a Python list, tuple, buffer, or RemoteArray.

    Parameters
    ----------
    object:
        Data to create the array from.  May be a nested list/tuple of
        numbers, an object exposing the buffer protocol (``array.array``,
        ``memoryview``, ``numpy.ndarray``) at any rank, or an existing
        :class:`RemoteArray`.
    dtype:
        Optional dtype string (e.g. ``"float64"``).  Inferred from data
        if not given.

    Returns
    -------
    RemoteArray
        A new remote array on the server.
    """
    if isinstance(object, RemoteArray):
        if dtype is None:
            return object
        # dtype cast: dispatch to server
        conn = get_connection()
        resp = conn.send_recv(
            encode_request(
                "astype",
                args=[{"__handle__": object.handle_id}, _normalize_dtype(dtype)],
            )
        )
        return _result_from_response(resp)

    if isinstance(object, (list, tuple)):
        flat, shape = _flatten(object)
        if not flat:
            # Empty array
            dtype_str = "float64" if dtype is None else _normalize_dtype(dtype)
            conn = get_connection()
            resp = conn.send_recv(encode_create_from_data(b"", list(shape), dtype_str))
            return _result_from_response(resp)

        dtype_str = _infer_dtype(flat) if dtype is None else _normalize_dtype(dtype)
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info

        # Complex types: split each value into (real, imag) pairs
        if dtype_str in ("complex64", "complex128"):
            expanded = []
            for v in flat:
                c = complex(v)
                expanded.extend([c.real, c.imag])
            flat = expanded
            fmt_char = "f" if dtype_str == "complex64" else "d"
            data = struct.pack(f"<{len(flat)}{fmt_char}", *flat)
        else:
            data = struct.pack(f"<{len(flat)}{fmt_char}", *flat)

        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, list(shape), dtype_str))
        return _result_from_response(resp)

    if isinstance(object, (int, float, complex)):
        # Scalar -> 0-d array
        if isinstance(object, complex) and dtype is None:
            dtype_str = "complex128"
        else:
            dtype_str = "float64" if dtype is None else _normalize_dtype(dtype)
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info

        if dtype_str in ("complex64", "complex128"):
            c = complex(object)
            pack_fmt = "f" if dtype_str == "complex64" else "d"
            data = struct.pack(f"<2{pack_fmt}", c.real, c.imag)
        else:
            data = struct.pack(f"<1{fmt_char}", object)
        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, [], dtype_str))
        return _result_from_response(resp)

    # Buffer-protocol inputs (stdlib array.array, memoryview, numpy.ndarray,
    # bytes-backed buffers). Native numpy-backed flopscope accepts these; mirror
    # it. Send the raw bytes directly (C-speed) rather than materializing a
    # Python list, so the timed array() dispatch is not inflated.
    # bytes/bytearray/str expose a buffer but numpy treats them as string/bytes
    # dtypes (e.g. np.array(b"abc") -> |S3 scalar), which flopscope has no dtype
    # for. Reject cleanly rather than mis-reading them as a uint8 buffer.
    if isinstance(object, (bytes, bytearray, str)):
        raise TypeError(
            f"Cannot create array from {type(object).__name__}. "
            f"Expected list, tuple, int, float, RemoteArray, or a numeric buffer "
            f"(array.array / memoryview of a numeric type)."
        )
    try:
        mv = memoryview(object)
    except TypeError:
        mv = None
    if mv is not None:
        native_wire = _BUFFER_FORMAT_TO_WIRE.get(mv.format)
        if native_wire is not None:
            # tobytes() is C-order for F-order and strided buffers alike, and
            # mv.shape is the logical shape at any rank -- including () for a
            # 0-d buffer, which create_from_data already accepts (the scalar
            # branch above sends exactly that). The old path derived the length
            # from nbytes//itemsize and always sent [n], which is why anything
            # above rank 1 was refused and a 0-d buffer came back as rank 1.
            data = mv.tobytes()
            conn = get_connection()
            resp = conn.send_recv(
                encode_create_from_data(data, list(mv.shape), native_wire)
            )
            arr = _result_from_response(resp)
            if dtype is not None:
                want = _normalize_dtype(dtype)
                if want != native_wire:
                    return array(arr, dtype=want)  # server-side cast (astype)
            return arr

    raise TypeError(
        f"Cannot create array from {type(object).__name__}. "
        f"Expected list, tuple, int, float, RemoteArray, or a numeric buffer "
        f"(array.array / memoryview of a numeric type)."
    )


# ---------------------------------------------------------------------------
# Special-case: asarray()
# ---------------------------------------------------------------------------


def _needs_local_buffer_upload(value) -> bool:
    """Whether *value* is a numeric buffer the wire cannot carry as it stands.

    Deliberately narrow: it answers True only for values the backend refuses
    today. Everything the wire already represents -- a ``RemoteArray`` handle,
    a nested list, a number, a numpy scalar ``_encode_arg`` unwraps to a plain
    one -- is excluded, so every call that works today keeps its exact
    dispatch, and therefore its exact billing. The only behaviour that changes
    is a refusal turning into a working call.
    """
    # bytes-like values ARE carried by the wire (as binary), and array()
    # rejects them on purpose since numpy reads them as an S-dtype scalar.
    if isinstance(value, (bytes, bytearray, str)):
        return False
    try:
        mv = memoryview(value)
    except (TypeError, ValueError):
        # Not a buffer at all, or a dtype memoryview itself refuses (e.g.
        # datetime64): keep today's outcome rather than inventing a new one.
        # This is also the cheap early exit for lists, numbers and handles.
        return False
    if _BUFFER_FORMAT_TO_WIRE.get(mv.format) is None:
        return False
    # A buffer _encode_arg rewrites (a numpy scalar it unwraps to a Python
    # float, say) already has a wire form that works; only the ones it hands
    # back untouched would be msgpack'd as opaque binary and misread.
    return _encode_arg(value) is value


_asarray_remote = _make_proxy("asarray")


@timed_dispatch
def asarray(a, *args, **kwargs):
    """Convert *a* to a remote array, without copying when it already is one.

    Accepts everything the server accepts plus, unlike the auto-generated
    proxy this replaces, objects exposing the buffer protocol at any rank
    (``numpy.ndarray``, ``array.array``, ``memoryview``). Those are uploaded
    with :func:`array` first, because a raw buffer has no wire representation;
    every other input is dispatched to the server unchanged.
    """
    if _needs_local_buffer_upload(a):
        a = array(a)
    return _asarray_remote(a, *args, **kwargs)


# ---------------------------------------------------------------------------
# Special-case: einsum()
# ---------------------------------------------------------------------------


@timed_dispatch
def einsum(subscripts, *operands, **kwargs):
    """Einstein summation on remote arrays.

    Parameters
    ----------
    subscripts:
        Subscript string (e.g. ``"ij,jk->ik"``).
    *operands:
        Input :class:`RemoteArray` objects.
    **kwargs:
        Additional keyword arguments forwarded to the server.

    Returns
    -------
    RemoteArray
        Result of the einsum operation.
    """
    conn = get_connection()
    encoded_args = [subscripts] + [_encode_arg(op) for op in operands]
    encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
    resp = conn.send_recv(
        encode_request("einsum", args=encoded_args, kwargs=encoded_kwargs)
    )
    return _result_from_response(resp)


# ---------------------------------------------------------------------------
# Auto-generate proxy functions for all non-blacklisted top-level ops
# ---------------------------------------------------------------------------

from flopscope._io import (  # noqa: E402
    load,
    save,
    savez,
    savez_compressed,
)
from flopscope._module import Module  # noqa: E402

# Functions that are special-cased above and should not be overwritten.
_SPECIAL_CASED = frozenset(
    {"array", "asarray", "einsum", "load", "save", "savez", "savez_compressed"}
)

# Functions that belong to submodules (contain a dot) are handled by the
# submodule packages themselves.
_generated_proxies: list[str] = []
for _op_name in iter_proxyable():
    if "." in _op_name:
        continue  # submodule function
    if _op_name in _SPECIAL_CASED:
        continue
    globals()[_op_name] = _make_proxy(_op_name)
    _generated_proxies.append(_op_name)

del _op_name  # clean up loop variable


# ---------------------------------------------------------------------------
# Module-level __getattr__ for blacklisted / unknown names
# ---------------------------------------------------------------------------

# We import the factory but define the function inline so we can also
# check against names that are already defined in the module namespace.

from flopscope._getattr import make_module_getattr as _make_module_getattr  # noqa: E402

_module_getattr = _make_module_getattr("", "flopscope")


def __getattr__(name: str):
    return _module_getattr(name)


# ---------------------------------------------------------------------------
# Public surface (controls ``from flopscope import *`` and dir hygiene)
# ---------------------------------------------------------------------------

# Implementation details that must NOT leak into the public ``fnp`` namespace.
_INTERNAL_NAMES = frozenset(
    {
        "Any",
        "annotations",
        "builtins",
        "struct",
        "get_connection",
        "encode_request",
        "encode_create_from_data",
        "iter_proxyable",
        "is_valid_op",
        "get_category",
        "BLACKLISTED",
        "FUNCTION_CATEGORIES",
        "LOCAL_CALLBACK_OPS",
        "timed_dispatch",
        "Module",
    }
)

__all__ = sorted(
    name
    for name in list(globals())
    if not name.startswith("_") and name not in _INTERNAL_NAMES
)
