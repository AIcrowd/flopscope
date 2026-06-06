"""Transparent proxy classes for server-side arrays and scalar values.

``RemoteArray`` and ``RemoteScalar`` make the client-server split invisible
to participants: metadata is cached locally while data access and arithmetic
operations are dispatched to the server transparently.
"""

from __future__ import annotations

import struct
from typing import Any

from flopscope._dispatch import timed_dispatch
from flopscope._math_compat import prod as _prod

# ---------------------------------------------------------------------------
# dtype helpers  (NO numpy -- pure struct)
# ---------------------------------------------------------------------------

#: Maps dtype string to (struct format char, byte width).
#: Complex types use their float-component format char; _bytes_to_list
#: handles pairing them into Python complex numbers.
_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "float64": ("d", 8),
    "float32": ("f", 4),
    "float16": ("e", 2),
    "int64": ("q", 8),
    "int32": ("i", 4),
    "int16": ("h", 2),
    "int8": ("b", 1),
    "uint64": ("Q", 8),
    "uint32": ("I", 4),
    "uint16": ("H", 2),
    "uint8": ("B", 1),
    "bool": ("?", 1),
    "complex64": ("f", 8),  # two float32 components
    "complex128": ("d", 16),  # two float64 components
}

#: dtypes that are stored as pairs of real components.
_COMPLEX_DTYPES = frozenset({"complex64", "complex128"})


def _bytes_to_list(data: bytes, shape: tuple[int, ...], dtype: str) -> Any:
    """Convert raw *data* bytes into a (possibly nested) Python list.

    Uses :mod:`struct` for unpacking -- no numpy dependency.

    Parameters
    ----------
    data:
        Raw little-endian bytes produced by the server.
    shape:
        Array shape.  An empty tuple means scalar.
    dtype:
        Element data-type string, e.g. ``"float64"``.

    Returns
    -------
    Scalar value, flat list, or nested list of lists depending on *shape*.
    """
    fmt_char, item_size = _DTYPE_INFO[dtype]
    total = _prod(shape) if shape else 1

    # Empty array — no data to unpack
    if total == 0:
        return _reshape([], shape) if len(shape) > 1 else []

    if dtype in _COMPLEX_DTYPES:
        # Unpack as pairs of floats and construct complex numbers
        flat_reals = list(struct.unpack(f"<{total * 2}{fmt_char}", data))
        flat = [
            complex(flat_reals[i], flat_reals[i + 1])
            for i in range(0, len(flat_reals), 2)
        ]
    else:
        flat = list(struct.unpack(f"<{total}{fmt_char}", data))

    # Scalar
    if not shape:
        return flat[0]

    # 1-D
    if len(shape) == 1:
        return flat

    # N-D: reshape into nested lists
    return _reshape(flat, shape)


def _reshape(flat: list, shape: tuple[int, ...]) -> Any:
    """Reshape a flat list into nested lists matching *shape*."""
    if len(shape) == 1:
        return flat

    stride = _prod(shape[1:])
    return [
        _reshape(flat[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


# ---------------------------------------------------------------------------
# RemoteScalar
# ---------------------------------------------------------------------------


class RemoteScalar:
    """Wraps a scalar value returned from the server.

    Behaves like a Python number for comparisons, arithmetic (via
    ``float()``), and hashing.  Also passes ``isinstance(s, RemoteArray)``
    checks (see :meth:`RemoteArray.__instancecheck__`).

    Parameters
    ----------
    value:
        The scalar numeric value.
    dtype:
        Data-type string (e.g. ``"float64"``).
    """

    __slots__ = ("_value", "_dtype")

    def __init__(self, value: int | float, dtype: str) -> None:
        self._value = value
        self._dtype = dtype

    # -- array-like metadata ------------------------------------------------

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def handle_id(self) -> None:
        return None

    # -- conversions --------------------------------------------------------

    def tolist(self) -> int | float:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RemoteScalar({self._value!r}, dtype={self._dtype!r})"

    def __str__(self) -> str:
        return str(self._value)

    # -- comparisons --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value == other._value
        return self._value == other

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value < other._value
        return self._value < other  # type: ignore[operator]

    def __le__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value <= other._value
        return self._value <= other  # type: ignore[operator]

    def __gt__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value > other._value
        return self._value > other  # type: ignore[operator]

    def __ge__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value >= other._value
        return self._value >= other  # type: ignore[operator]

    def __hash__(self) -> int:
        return hash(self._value)

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value + other_val, self._dtype)

    def __radd__(self, other):
        return RemoteScalar(other + self._value, self._dtype)

    def __sub__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value - other_val, self._dtype)

    def __rsub__(self, other):
        return RemoteScalar(other - self._value, self._dtype)

    def __mul__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value * other_val, self._dtype)

    def __rmul__(self, other):
        return RemoteScalar(other * self._value, self._dtype)

    def __truediv__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value / other_val, self._dtype)

    def __rtruediv__(self, other):
        return RemoteScalar(other / self._value, self._dtype)

    def __floordiv__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value // other_val, self._dtype)

    def __rfloordiv__(self, other):
        return RemoteScalar(other // self._value, self._dtype)

    def __mod__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value % other_val, self._dtype)

    def __rmod__(self, other):
        return RemoteScalar(other % self._value, self._dtype)

    def __pow__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return RemoteScalar(self._value**other_val, self._dtype)

    def __rpow__(self, other):
        return RemoteScalar(other**self._value, self._dtype)

    def __neg__(self):
        return RemoteScalar(-self._value, self._dtype)

    def __abs__(self):
        return RemoteScalar(abs(self._value), self._dtype)


# ---------------------------------------------------------------------------
# RemoteArray
# ---------------------------------------------------------------------------


def _encode_index_key(key):
    """Encode an index key for transmission to the server via msgpack.

    Slices become ``{"__slice__": [start, stop, step]}``.
    Tuples become lists of encoded items.
    RemoteArray -> ``{"__handle__": handle_id}`` (fancy indexing).
    RemoteScalar -> its raw value.
    Integers pass through as-is.
    """
    # Check RemoteScalar before RemoteArray (metaclass makes scalar pass isinstance check)
    if type(key) is RemoteScalar:
        return key._value
    if isinstance(key, RemoteArray):
        return {"__handle__": key.handle_id}
    if isinstance(key, slice):
        return {"__slice__": [key.start, key.stop, key.step]}
    if isinstance(key, tuple):
        return [_encode_index_key(k) for k in key]
    if isinstance(key, list):
        return [_encode_index_key(k) for k in key]
    return key


class _RemoteArrayMeta(type):
    """Metaclass so that ``isinstance(RemoteScalar(...), RemoteArray)`` is True."""

    def __instancecheck__(cls, instance):
        if type.__instancecheck__(cls, instance):
            return True
        # RemoteScalar should also be considered an ndarray-like object.
        return isinstance(instance, RemoteScalar)


class RemoteArray(metaclass=_RemoteArrayMeta):
    """Transparent proxy for a server-side numpy array.

    The constructor only stores metadata -- no data is transferred until
    explicitly requested (via :meth:`tolist`, :meth:`__repr__`, etc.).

    Parameters
    ----------
    handle_id:
        Opaque server handle for this array.
    shape:
        Array shape tuple.
    dtype:
        Element data-type string (e.g. ``"float64"``).
    """

    __slots__ = ("_handle_id", "_shape", "_dtype", "_symmetry")

    def __init__(self, handle_id: str, shape: tuple, dtype: str, symmetry=None) -> None:
        self._handle_id = handle_id
        self._shape = tuple(shape)
        self._dtype = dtype
        self._symmetry = symmetry

    # -- cached metadata (no round-trip) ------------------------------------

    @property
    def handle_id(self) -> str:
        return self._handle_id

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def symmetry(self):
        """SymmetryGroup metadata, or None if not a symmetric tensor."""
        return self._symmetry

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _prod(self._shape) if self._shape else 1

    @property
    def nbytes(self) -> int:
        _, item_size = _DTYPE_INFO[self._dtype]
        return self.size * item_size

    @property
    def T(self):
        """Transpose of the array (dispatched to server)."""
        return self._dispatch_op("transpose", self)

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of unsized object (0-d array)")
        return self._shape[0]

    # -- data access (auto-fetch from server) -------------------------------

    @timed_dispatch
    def _fetch_data(self) -> tuple[bytes, tuple, str]:
        """Fetch the raw data from the server.

        Returns ``(raw_bytes, shape, dtype)``.
        """
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_fetch

        resp = get_connection().send_recv(encode_fetch(self._handle_id))
        # Fetch responses may have data at top level or inside "result"
        if "data" in resp:
            return resp["data"], tuple(resp["shape"]), resp["dtype"]
        result = resp.get("result", {})
        return result["data"], tuple(result["shape"]), result["dtype"]

    @timed_dispatch
    def tolist(self) -> Any:
        """Fetch data and convert to a nested Python list."""
        data, shape, dtype = self._fetch_data()
        return _bytes_to_list(data, shape, dtype)

    @timed_dispatch
    def __repr__(self) -> str:
        try:
            values = self.tolist()
            return f"array({values!r})"
        except Exception:
            return (
                f"RemoteArray(handle_id={self._handle_id!r}, "
                f"shape={self._shape}, dtype={self._dtype!r})"
            )

    @timed_dispatch
    def __str__(self) -> str:
        return self.__repr__()

    @timed_dispatch
    def __float__(self) -> float:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        # Unwrap single-element lists (e.g., shape (1,) returns [42.0])
        while isinstance(result, list):
            result = result[0]
        return float(result)

    @timed_dispatch
    def __int__(self) -> int:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        while isinstance(result, list):
            result = result[0]
        return int(result)

    @timed_dispatch
    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
            )
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        while isinstance(result, list):
            result = result[0]
        return bool(result)

    def __iter__(self):
        if not self._shape:
            raise TypeError("iteration over a 0-d array")
        for i in range(self._shape[0]):
            yield self[i]

    @timed_dispatch
    def __getitem__(self, key):
        """Index into the array, dispatching to the server.

        For integer keys on 1-D arrays, returns the scalar value.
        For slices or indexing on 2D+ arrays, returns a RemoteArray.
        """
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        # Encode the key for transmission
        encoded_key = _encode_index_key(key)
        encoded_handle = {"__handle__": self._handle_id}
        resp = get_connection().send_recv(
            encode_request("__getitem__", args=[encoded_handle, encoded_key])
        )
        return _result_from_response(resp)

    def __setitem__(self, key, value):
        raise TypeError(
            "flopscope arrays are immutable. Cannot assign to array elements."
        )

    # -- operator overloads (dispatch to server) ----------------------------

    @timed_dispatch
    def _dispatch_op(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Encode and send an operation to the server, return the result."""
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = get_connection().send_recv(
            encode_request(op_name, args=encoded_args, kwargs=encoded_kwargs)
        )
        return _result_from_response(resp)

    # Arithmetic
    def __add__(self, other):
        return self._dispatch_op("add", self, other)

    def __radd__(self, other):
        return self._dispatch_op("add", other, self)

    def __sub__(self, other):
        return self._dispatch_op("subtract", self, other)

    def __rsub__(self, other):
        return self._dispatch_op("subtract", other, self)

    def __mul__(self, other):
        return self._dispatch_op("multiply", self, other)

    def __rmul__(self, other):
        return self._dispatch_op("multiply", other, self)

    def __truediv__(self, other):
        return self._dispatch_op("true_divide", self, other)

    def __rtruediv__(self, other):
        return self._dispatch_op("true_divide", other, self)

    def __floordiv__(self, other):
        return self._dispatch_op("floor_divide", self, other)

    def __rfloordiv__(self, other):
        return self._dispatch_op("floor_divide", other, self)

    def __mod__(self, other):
        return self._dispatch_op("remainder", self, other)

    def __rmod__(self, other):
        return self._dispatch_op("remainder", other, self)

    def __pow__(self, other):
        return self._dispatch_op("power", self, other)

    def __rpow__(self, other):
        return self._dispatch_op("power", other, self)

    def __matmul__(self, other):
        return self._dispatch_op("matmul", self, other)

    def __rmatmul__(self, other):
        return self._dispatch_op("matmul", other, self)

    def __neg__(self):
        return self._dispatch_op("negative", self)

    def __abs__(self):
        return self._dispatch_op("abs", self)

    # Comparisons (dispatch to server -- element-wise, returning RemoteArray)
    def __eq__(self, other):
        # Only dispatch to server for array/RemoteArray comparisons
        if isinstance(other, (RemoteArray, RemoteScalar)):
            return self._dispatch_op("equal", self, other)
        # For plain scalars, also dispatch
        return self._dispatch_op("equal", self, other)

    def __ne__(self, other):
        return self._dispatch_op("not_equal", self, other)

    def __lt__(self, other):
        return self._dispatch_op("less", self, other)

    def __le__(self, other):
        return self._dispatch_op("less_equal", self, other)

    def __gt__(self, other):
        return self._dispatch_op("greater", self, other)

    def __ge__(self, other):
        return self._dispatch_op("greater_equal", self, other)

    # -- convenience methods (delegate to server-side ops) ------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return self._dispatch_op("reshape", self, list(shape))

    def astype(self, dtype):
        return self._dispatch_op("astype", self, dtype)

    def sum(self, axis=None, **kwargs):
        if axis is not None:
            return self._dispatch_op("sum", self, axis=axis, **kwargs)
        return self._dispatch_op("sum", self, **kwargs)

    def mean(self, axis=None, **kwargs):
        if axis is not None:
            return self._dispatch_op("mean", self, axis=axis, **kwargs)
        return self._dispatch_op("mean", self, **kwargs)

    def max(self, axis=None, **kwargs):
        if axis is not None:
            return self._dispatch_op("max", self, axis=axis, **kwargs)
        return self._dispatch_op("max", self, **kwargs)

    def min(self, axis=None, **kwargs):
        if axis is not None:
            return self._dispatch_op("min", self, axis=axis, **kwargs)
        return self._dispatch_op("min", self, **kwargs)

    def flatten(self):
        return self._dispatch_op("ravel", self)

    def ravel(self):
        return self._dispatch_op("ravel", self)

    def transpose(self, *axes):
        if axes:
            return self._dispatch_op("transpose", self, list(axes))
        return self._dispatch_op("transpose", self)

    def dot(self, other):
        return self._dispatch_op("dot", self, other)

    def copy(self):
        return self._dispatch_op("copy", self)

    # RemoteArray is not hashable (same as numpy arrays)
    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# RemoteGenerator
# ---------------------------------------------------------------------------


class RemoteGenerator:
    """Transparent proxy for a server-side numpy ``Generator``.

    ``fnp.random.default_rng(seed)`` returns one of these. Sampling methods
    dispatch to the server, where the RNG state lives and advances — so the
    stream is deterministic per seed and FLOP-counted server-side, exactly
    like the in-process counted Generator.
    """

    __slots__ = ("_handle_id",)

    def __init__(self, handle_id: str) -> None:
        self._handle_id = handle_id

    @property
    def handle_id(self) -> str:
        return self._handle_id

    def __repr__(self) -> str:
        return f"RemoteGenerator(handle_id={self._handle_id!r})"

    @timed_dispatch
    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        encoded_args = [_encode_arg(self)] + [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = get_connection().send_recv(
            encode_request(
                f"Generator.{method}", args=encoded_args, kwargs=encoded_kwargs
            )
        )
        return _result_from_response(resp)

    def uniform(self, *args, **kwargs):
        return self._call("uniform", *args, **kwargs)

    def standard_normal(self, *args, **kwargs):
        return self._call("standard_normal", *args, **kwargs)

    def normal(self, *args, **kwargs):
        return self._call("normal", *args, **kwargs)

    def integers(self, *args, **kwargs):
        return self._call("integers", *args, **kwargs)

    def random(self, *args, **kwargs):
        return self._call("random", *args, **kwargs)

    def standard_exponential(self, *args, **kwargs):
        return self._call("standard_exponential", *args, **kwargs)

    def exponential(self, *args, **kwargs):
        return self._call("exponential", *args, **kwargs)

    def poisson(self, *args, **kwargs):
        return self._call("poisson", *args, **kwargs)

    def binomial(self, *args, **kwargs):
        return self._call("binomial", *args, **kwargs)

    def beta(self, *args, **kwargs):
        return self._call("beta", *args, **kwargs)

    def gamma(self, *args, **kwargs):
        return self._call("gamma", *args, **kwargs)

    def choice(self, *args, **kwargs):
        return self._call("choice", *args, **kwargs)

    def permutation(self, *args, **kwargs):
        return self._call("permutation", *args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _result_from_response(resp: dict) -> RemoteArray | RemoteScalar | tuple | dict:
    """Convert a server response dict into the appropriate proxy object.

    Examines the ``"result"`` key:

    * ``"value"`` present  -> :class:`RemoteScalar`
    * ``"multi"`` present  -> ``tuple`` of :class:`RemoteArray`
    * ``"id"``   present  -> single :class:`RemoteArray`
    * otherwise           -> raw dict
    """
    result = resp.get("result", {})

    if "gen_id" in result:
        return RemoteGenerator(result["gen_id"])

    if "value" in result:
        return RemoteScalar(value=result["value"], dtype=result.get("dtype", "float64"))

    if "multi" in result:
        items = []
        for item in result["multi"]:
            if "id" in item:
                symmetry = None
                if "symmetry" in item:
                    from flopscope._perm_group import SymmetryGroup

                    symmetry = SymmetryGroup.from_payload(item["symmetry"])
                items.append(
                    RemoteArray(
                        handle_id=item["id"],
                        shape=tuple(item["shape"]),
                        dtype=item["dtype"],
                        symmetry=symmetry,
                    )
                )
            elif "value" in item:
                items.append(
                    RemoteScalar(
                        value=item["value"],
                        dtype=item.get("dtype", "float64"),
                    )
                )
            else:
                items.append(item)
        return tuple(items)

    if "id" in result:
        symmetry = None
        if "symmetry" in result:
            from flopscope._perm_group import SymmetryGroup

            symmetry = SymmetryGroup.from_payload(result["symmetry"])
        return RemoteArray(
            handle_id=result["id"],
            shape=tuple(result["shape"]),
            dtype=result["dtype"],
            symmetry=symmetry,
        )

    return result


# ---------------------------------------------------------------------------
# Argument encoding helper (used by _dispatch_op and proxy factories)
# ---------------------------------------------------------------------------


def _encode_arg(arg):
    """Recursively encode RemoteArray/RemoteScalar objects for wire transmission.

    - RemoteScalar -> its raw ``_value``
    - RemoteArray  -> ``{"__handle__": handle_id}``
    - list/tuple   -> recursively encoded list (msgpack can't distinguish tuple/list)
    - everything else passes through unchanged

    Note: RemoteScalar must be checked *before* RemoteArray because the
    metaclass makes ``isinstance(RemoteScalar(...), RemoteArray)`` True.
    """
    # Check RemoteScalar first (it passes isinstance RemoteArray due to metaclass)
    if type(arg) is RemoteScalar:
        return arg._value
    if isinstance(arg, RemoteArray):
        return {"__handle__": arg.handle_id}
    if isinstance(arg, RemoteGenerator):
        return {"__gen__": arg.handle_id}
    from flopscope._perm_group import SymmetryGroup

    if isinstance(arg, SymmetryGroup):
        return {"__symmetry_group__": arg.to_payload()}
    if isinstance(arg, (list, tuple)):
        return [_encode_arg(item) for item in arg]
    return arg
