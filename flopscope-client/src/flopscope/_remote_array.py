"""Transparent proxy classes for server-side arrays and scalar values.

``RemoteArray`` and ``RemoteScalar`` make the client-server split invisible
to participants: metadata is cached locally while data access and arithmetic
operations are dispatched to the server transparently.
"""

from __future__ import annotations

import struct
import weakref
from typing import Any

from flopscope._dispatch import timed_dispatch
from flopscope._math_compat import prod as _prod

# ---------------------------------------------------------------------------
# dtype helpers  (NO numpy -- pure struct)
# ---------------------------------------------------------------------------

#: Maps dtype string to (struct format char, byte width).
#: Complex types use their float-component format char; _bytes_to_list
#: handles pairing them into Python complex numbers.
_RAW_BUFFER_MSG = (
    "'{attr}' exposes a pointer to the array's local memory buffer, which does "
    "not exist for a flopscope RemoteArray — the data lives on the remote grading "
    "server, not in this process. If you genuinely need the bytes locally, "
    "materialize first with arr.tolist() or numpy.asarray(arr)."
)

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

# Accepted dtype string spellings -> canonical wire name (keys of _DTYPE_INFO
# plus the "bool_" alias). Kept here (not flopscope._dtypes) so _encode_arg can
# resolve dtype-like args without a circular import.
_DTYPE_ALIASES: dict[str, str] = {name: name for name in _DTYPE_INFO}
_DTYPE_ALIASES["bool_"] = "bool"

# Python builtin types -> wire dtype (numpy's defaults).
_PY_TYPE_TO_WIRE: dict[type, str] = {
    float: "float64",
    int: "int64",
    bool: "bool",
    complex: "complex128",
}


def _flatten_to_list(nested):
    """Flatten arbitrarily-nested lists (RemoteArray.tolist() output) to a flat list."""
    out: list = []

    def _rec(x):
        if isinstance(x, list):
            for e in x:
                _rec(e)
        else:
            out.append(x)

    _rec(nested)
    return out


def _resolve_dtype_wire_name(spec: Any) -> str | None:
    """Return the canonical wire dtype name for a dtype-like *spec*, else None.

    Numpy-free + duck-typed, so it recognizes — without importing numpy — every
    dtype spelling a participant (or numpy's own code) may pass:

    * flopscope dtype labels/objects (``_flopscope_dtype_name``);
    * dtype string spellings (``"float64"``, ``"bool_"``);
    * Python builtin types (``float``/``int``/``bool``/``complex``);
    * numpy scalar TYPE objects (``np.float64`` -> ``__name__`` == "float64");
    * numpy dtype objects / numpy 2.x new-style DType instances
      (``np.dtype("float64")`` / a ``Float64DType`` -> ``.name`` == "float64").

    Exotic/unsupported dtypes (e.g. ``longdouble``, structured) return ``None``;
    the caller decides whether to reject or fall through.
    """
    name = getattr(spec, "_flopscope_dtype_name", None)
    if isinstance(name, str):
        return name
    if isinstance(spec, str):
        return _DTYPE_ALIASES.get(spec)
    if isinstance(spec, type):
        if spec in _PY_TYPE_TO_WIRE:
            return _PY_TYPE_TO_WIRE[spec]
        return _DTYPE_ALIASES.get(getattr(spec, "__name__", ""))
    nm = getattr(spec, "name", None)
    if isinstance(nm, str):
        return _DTYPE_ALIASES.get(nm)
    return None


def _dtype_itemsize(dtype_name: str) -> int:
    return _DTYPE_INFO[dtype_name][1]


def _c_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


class _RemoteFlags:
    """Read-only numpy-flags-like view over a remote array's layout."""

    def __init__(self, shape, strides, itemsize):
        c_contig = strides == _c_strides(shape, itemsize)
        self._d = {
            "C_CONTIGUOUS": c_contig,
            "F_CONTIGUOUS": c_contig and len(shape) <= 1,
            "OWNDATA": False,
            "WRITEABLE": False,  # client RemoteArray is immutable
            "ALIGNED": True,
        }

    def __getitem__(self, key):
        return self._d[key]

    def __getattr__(self, name):
        upper = name.upper()
        if upper in self._d:
            return self._d[upper]
        raise AttributeError(name)


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

    def __index__(self) -> int:
        # Only integer/bool scalars can index (numpy parity: float/complex
        # scalars have no __index__, even for whole values like 2.0). The "int"
        # substring matches BOTH signed and unsigned widths -- "int64" and
        # "uint64" both contain "int" -- so unsigned scalars index fine too.
        if "int" not in self._dtype and self._dtype != "bool":
            raise TypeError(
                f"only integer scalars can be used as indices; this RemoteScalar "
                f"has dtype {self._dtype!r}"
            )
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
    #
    # A scalar combined with an *array* broadcasts to an array, exactly like
    # numpy (``np.float64(2) * np.array([1, 2])`` -> ``array([2, 4])``). When
    # ``other`` is a RemoteArray the ``self._value <op> other`` expression is
    # dispatched through ``RemoteArray.__r<op>__`` and already yields a
    # RemoteArray; we must return that array UNWRAPPED. Wrapping it back in a
    # ``RemoteScalar`` would create a malformed "scalar" whose ``_value`` is a
    # RemoteArray, which later serializes as a raw RemoteArray and crashes the
    # wire encoder with ``can not serialize 'RemoteArray' object``. The
    # ``_scalar_result`` helper mirrors numpy: scalar results stay scalars,
    # array results pass through.

    def __add__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value + other_val, self._dtype)

    def __radd__(self, other):
        return _scalar_result(other + self._value, self._dtype)

    def __sub__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value - other_val, self._dtype)

    def __rsub__(self, other):
        return _scalar_result(other - self._value, self._dtype)

    def __mul__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value * other_val, self._dtype)

    def __rmul__(self, other):
        return _scalar_result(other * self._value, self._dtype)

    def __truediv__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value / other_val, self._dtype)

    def __rtruediv__(self, other):
        return _scalar_result(other / self._value, self._dtype)

    def __floordiv__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value // other_val, self._dtype)

    def __rfloordiv__(self, other):
        return _scalar_result(other // self._value, self._dtype)

    def __mod__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value % other_val, self._dtype)

    def __rmod__(self, other):
        return _scalar_result(other % self._value, self._dtype)

    def __pow__(self, other):
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return _scalar_result(self._value**other_val, self._dtype)

    def __rpow__(self, other):
        return _scalar_result(other**self._value, self._dtype)

    def __neg__(self):
        return _scalar_result(-self._value, self._dtype)

    def __abs__(self):
        return _scalar_result(abs(self._value), self._dtype)


def _scalar_result(value, dtype):
    """Wrap a scalar arithmetic result, passing array/proxy results through.

    ``RemoteScalar`` arithmetic with a plain number yields a number, which we
    re-wrap as a ``RemoteScalar`` (numpy-scalar analog). But ``RemoteScalar``
    combined with a ``RemoteArray`` broadcasts to a ``RemoteArray`` (the op runs
    server-side via the array's reflected dunder). That array result must be
    returned as-is: re-wrapping it would yield a ``RemoteScalar`` holding an
    array, which the wire encoder later unwraps to a raw ``RemoteArray`` and
    fails on (``can not serialize 'RemoteArray' object``). ``RemoteScalar``
    itself passes ``isinstance(_, RemoteArray)`` via the metaclass, so an
    already-proxy result is also returned unchanged.
    """
    if isinstance(value, RemoteArray):
        return value
    return RemoteScalar(value, dtype)


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
    if key is Ellipsis:
        return {"__ellipsis__": True}
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

    # __weakref__ makes instances weak-referenceable (required for the
    # weakref.finalize below that releases the server handle on GC).
    __slots__ = ("_handle_id", "_shape", "_dtype", "_symmetry", "__weakref__")

    def __init__(self, handle_id: str, shape: tuple, dtype: str, symmetry=None) -> None:
        self._handle_id = handle_id
        self._shape = tuple(shape)
        self._dtype = dtype
        self._symmetry = symmetry
        if handle_id is not None:
            # Release the server handle when this proxy is GC'd. The callback
            # takes handle_id (NOT self), so it never resurrects the instance.
            # The import is deferred AND guarded: cross-package parity/integration
            # tests exec this module by file path with `flopscope` resolving to
            # the full package (which has no `_handles`). There the GC-free is not
            # needed, so we skip the finalizer rather than fail construction. In
            # the real client venv `_handles` always resolves, so the leak fix is
            # active. (ModuleNotFoundError only — a broken `_handles` still raises.)
            try:
                from flopscope._handles import enqueue_free
            except ModuleNotFoundError:
                pass
            else:
                weakref.finalize(self, enqueue_free, handle_id)

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
    def is_symmetric(self) -> bool:
        """True if this array carries symmetry metadata (mirrors native
        SymmetricTensor.is_symmetric)."""
        return self._symmetry is not None

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
    def itemsize(self) -> int:
        return _dtype_itemsize(self._dtype)

    @property
    def strides(self) -> tuple:
        return _c_strides(self._shape, self.itemsize)

    @property
    def flags(self):
        return _RemoteFlags(self._shape, self.strides, self.itemsize)

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

    # --- conversion / protocol dunders (rc3 parity) ---
    def __contains__(self, item):
        return item in _flatten_to_list(self.tolist())

    @timed_dispatch
    def __complex__(self):
        # Fetch the scalar and call complex() directly: routing through
        # float(self) would raise for a genuinely complex size-1 array
        # (e.g. fnp.array([1 + 2j])), unlike NumPy's complex(arr).
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        while isinstance(result, list):
            result = result[0]
        return complex(result)

    def __index__(self):
        return int(self)

    def __divmod__(self, other):
        return (self // other, self % other)

    def __rdivmod__(self, other):
        return (other // self, other % self)

    def __copy__(self):
        return self  # immutable proxy: a copy is the same handle

    def __deepcopy__(self, memo):
        return self

    def __array__(self, dtype=None):
        import numpy as _np

        arr = _np.asarray(self.tolist())
        return arr if dtype is None else arr.astype(dtype)

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
            "flopscope arrays are immutable, so item assignment (arr[i] = ...) "
            "and indexed in-place updates (arr[i] += ...) are not supported. "
            "Build the result functionally instead: collect the pieces in a "
            "list and combine them with fnp.stack(...) / fnp.concatenate(...), "
            "or use a whole-array update (arr = arr + x). See "
            "https://aicrowd.github.io/flopscope/docs/getting-started/competition/#immutable-arrays"
        )

    # In-place operators raise, mirroring native FlopscopeArray. Without these,
    # `a += b` would fall back to __add__ and silently rebind `a` on the client
    # while raising locally — breaking local==eval immutability parity.
    def _raise_inplace(self, verb: str, sym: str, func: str):
        raise TypeError(
            f"in-place {verb} (arr {sym} x) is not supported; flopscope arrays "
            f"are immutable. Use arr = fnp.{func}(arr, x) instead."
        )

    def __iadd__(self, other):
        self._raise_inplace("add", "+=", "add")

    def __isub__(self, other):
        self._raise_inplace("subtract", "-=", "subtract")

    def __imul__(self, other):
        self._raise_inplace("multiply", "*=", "multiply")

    def __itruediv__(self, other):
        self._raise_inplace("divide", "/=", "true_divide")

    def __ifloordiv__(self, other):
        self._raise_inplace("floor divide", "//=", "floor_divide")

    def __imod__(self, other):
        self._raise_inplace("mod", "%=", "mod")

    def __ipow__(self, other):
        self._raise_inplace("power", "**=", "power")

    def __imatmul__(self, other):
        self._raise_inplace("matmul", "@=", "matmul")

    def __iand__(self, other):
        self._raise_inplace("bitwise-and", "&=", "bitwise_and")

    def __ior__(self, other):
        self._raise_inplace("bitwise-or", "|=", "bitwise_or")

    def __ixor__(self, other):
        self._raise_inplace("bitwise-xor", "^=", "bitwise_xor")

    def __ilshift__(self, other):
        self._raise_inplace("left shift", "<<=", "left_shift")

    def __irshift__(self, other):
        self._raise_inplace("right shift", ">>=", "right_shift")

    # -- operator overloads (dispatch to server) ----------------------------

    @timed_dispatch
    def _dispatch_op(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Encode and send an operation to the server, return the result."""
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        try:
            request = encode_request(op_name, args=encoded_args, kwargs=encoded_kwargs)
        except (TypeError, ValueError) as exc:
            # Mirror the function-dispatch proxy (_make_proxy in __init__): an
            # unserializable arg leaks an opaque "can not serialize 'X' object"
            # from msgpack (this is how the RemoteScalar bug surfaced). Surface
            # a clear error naming the offending type instead. Imported lazily
            # (error path only) to avoid a circular import: __init__ imports us.
            from flopscope import _describe_unserializable
            from flopscope.errors import RemoteSerializationError

            bad = _describe_unserializable(encoded_args, encoded_kwargs)
            detail = f" {bad}" if bad else ""
            raise RemoteSerializationError(
                f"{op_name}() received an argument{detail} that cannot be sent "
                f"to the remote (client/server) backend. Pass a materialized "
                f"array or built-in (list / number / str) instead."
            ) from exc
        resp = get_connection().send_recv(request)
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

    def __pos__(self):
        return self._dispatch_op("positive", self)

    def __abs__(self):
        return self._dispatch_op("abs", self)

    # --- bitwise / shift operators (rc3 parity) ---
    def __and__(self, other):
        return self._dispatch_op("bitwise_and", self, other)

    def __rand__(self, other):
        return self._dispatch_op("bitwise_and", other, self)

    def __or__(self, other):
        return self._dispatch_op("bitwise_or", self, other)

    def __ror__(self, other):
        return self._dispatch_op("bitwise_or", other, self)

    def __xor__(self, other):
        return self._dispatch_op("bitwise_xor", self, other)

    def __rxor__(self, other):
        return self._dispatch_op("bitwise_xor", other, self)

    def __invert__(self):
        return self._dispatch_op("invert", self)

    def __lshift__(self, other):
        return self._dispatch_op("left_shift", self, other)

    def __rlshift__(self, other):
        return self._dispatch_op("left_shift", other, self)

    def __rshift__(self, other):
        return self._dispatch_op("right_shift", self, other)

    def __rrshift__(self, other):
        return self._dispatch_op("right_shift", other, self)

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

    # --- read-only ndarray methods bridged to server ops (rc3 parity) ---
    def all(self, *args, **kwargs):
        return self._dispatch_op("all", self, *args, **kwargs)

    def any(self, *args, **kwargs):
        return self._dispatch_op("any", self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        return self._dispatch_op("argmax", self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        return self._dispatch_op("argmin", self, *args, **kwargs)

    def argpartition(self, *args, **kwargs):
        return self._dispatch_op("argpartition", self, *args, **kwargs)

    def argsort(self, *args, **kwargs):
        return self._dispatch_op("argsort", self, *args, **kwargs)

    def choose(self, *args, **kwargs):
        return self._dispatch_op("choose", self, *args, **kwargs)

    def clip(self, *args, **kwargs):
        return self._dispatch_op("clip", self, *args, **kwargs)

    def compress(self, condition, *args, **kwargs):
        # ndarray method: a.compress(condition, axis=...) == np.compress(condition, a, axis=...)
        return self._dispatch_op("compress", condition, self, *args, **kwargs)

    def conj(self, *args, **kwargs):
        return self._dispatch_op("conj", self, *args, **kwargs)

    def conjugate(self, *args, **kwargs):
        return self._dispatch_op("conjugate", self, *args, **kwargs)

    def cumprod(self, *args, **kwargs):
        return self._dispatch_op("cumprod", self, *args, **kwargs)

    def cumsum(self, *args, **kwargs):
        return self._dispatch_op("cumsum", self, *args, **kwargs)

    def diagonal(self, *args, **kwargs):
        return self._dispatch_op("diagonal", self, *args, **kwargs)

    def nonzero(self, *args, **kwargs):
        return self._dispatch_op("nonzero", self, *args, **kwargs)

    def prod(self, *args, **kwargs):
        return self._dispatch_op("prod", self, *args, **kwargs)

    def repeat(self, *args, **kwargs):
        return self._dispatch_op("repeat", self, *args, **kwargs)

    def round(self, *args, **kwargs):
        return self._dispatch_op("round", self, *args, **kwargs)

    def searchsorted(self, *args, **kwargs):
        return self._dispatch_op("searchsorted", self, *args, **kwargs)

    def squeeze(self, *args, **kwargs):
        return self._dispatch_op("squeeze", self, *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._dispatch_op("std", self, *args, **kwargs)

    def swapaxes(self, *args, **kwargs):
        return self._dispatch_op("swapaxes", self, *args, **kwargs)

    def take(self, *args, **kwargs):
        return self._dispatch_op("take", self, *args, **kwargs)

    def trace(self, *args, **kwargs):
        return self._dispatch_op("trace", self, *args, **kwargs)

    def var(self, *args, **kwargs):
        return self._dispatch_op("var", self, *args, **kwargs)

    def item(self, *args):
        # No server op: materialize and index (numpy .item() common cases).
        flat = _flatten_to_list(self.tolist())
        if not args:
            if len(flat) != 1:
                raise ValueError(
                    "can only convert an array of size 1 to a Python scalar"
                )
            return flat[0]
        if len(args) == 1:
            return flat[args[0]]
        raise TypeError("RemoteArray.item() supports item() or item(flat_index)")

    # --- raw-buffer pointer properties: raise a clear error (rc3 parity) ---
    @property
    def data(self):
        raise AttributeError(_RAW_BUFFER_MSG.format(attr="data"))

    @property
    def ctypes(self):
        raise AttributeError(_RAW_BUFFER_MSG.format(attr="ctypes"))

    @property
    def __array_interface__(self):
        raise AttributeError(_RAW_BUFFER_MSG.format(attr="__array_interface__"))

    @property
    def __array_struct__(self):
        raise AttributeError(_RAW_BUFFER_MSG.format(attr="__array_struct__"))

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

    def permuted(self, *args, **kwargs):
        return self._call("permuted", *args, **kwargs)

    def chisquare(self, *args, **kwargs):
        return self._call("chisquare", *args, **kwargs)


# ---------------------------------------------------------------------------
# RemoteRandomState
# ---------------------------------------------------------------------------


class RemoteRandomState:
    """Transparent proxy for a server-side counted ``RandomState``.

    ``fnp.random.RandomState(seed)`` constructs one (dispatched; the legacy RNG
    state lives + advances server-side and is FLOP-counted). Sampler methods
    dispatch as ``RandomState.<method>``.
    """

    __slots__ = ("_handle_id",)

    def __init__(self, seed=None):
        from flopscope._connection import get_connection
        from flopscope._dispatch import dispatch_span
        from flopscope._protocol import encode_request

        with dispatch_span():
            resp = get_connection().send_recv(
                encode_request("random.RandomState", args=[_encode_arg(seed)])
            )
        self._handle_id = resp["result"]["rs_id"]

    @property
    def handle_id(self) -> str:
        return self._handle_id

    def __repr__(self) -> str:
        return f"RemoteRandomState(handle_id={self._handle_id!r})"

    @timed_dispatch
    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        encoded_args = [_encode_arg(self)] + [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = get_connection().send_recv(
            encode_request(
                f"RandomState.{method}", args=encoded_args, kwargs=encoded_kwargs
            )
        )
        return _result_from_response(resp)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            return self._call(name, *args, **kwargs)

        method.__name__ = name
        return method


# ---------------------------------------------------------------------------
# RemoteSeedSequence
# ---------------------------------------------------------------------------


class RemoteSeedSequence:
    """Proxy for a server-side ``numpy.random.SeedSequence`` (dispatched
    construction). Usable as a seed argument to ``fnp.random.default_rng``.
    ``generate_state`` is supported; ``spawn`` is not yet implemented."""

    __slots__ = ("_handle_id",)

    def __init__(self, entropy=None):
        from flopscope._connection import get_connection
        from flopscope._dispatch import dispatch_span
        from flopscope._protocol import encode_request

        with dispatch_span():
            resp = get_connection().send_recv(
                encode_request("random.SeedSequence", args=[_encode_arg(entropy)])
            )
        self._handle_id = resp["result"]["seq_id"]

    @property
    def handle_id(self) -> str:
        return self._handle_id

    def __repr__(self) -> str:
        return f"RemoteSeedSequence(handle_id={self._handle_id!r})"

    @timed_dispatch
    def generate_state(self, n_words, dtype="uint32"):
        from flopscope._connection import get_connection
        from flopscope._protocol import encode_request

        resp = get_connection().send_recv(
            encode_request(
                "SeedSequence.generate_state",
                args=[_encode_arg(self), n_words],
                kwargs={"dtype": dtype},
            )
        )
        return _result_from_response(resp)

    def spawn(self, n_children):
        raise NotImplementedError(
            "RemoteSeedSequence.spawn is not yet supported in the flopscope "
            "client; construct SeedSequences directly or use default_rng(seed)."
        )


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

    if "rs_id" in result:
        rs = RemoteRandomState.__new__(RemoteRandomState)
        rs._handle_id = result["rs_id"]
        return rs

    if "seq_id" in result:
        seq = RemoteSeedSequence.__new__(RemoteSeedSequence)
        seq._handle_id = result["seq_id"]
        return seq

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
    if isinstance(arg, RemoteRandomState):
        return {"__rs__": arg.handle_id}
    if isinstance(arg, RemoteSeedSequence):
        return {"__seq__": arg.handle_id}
    from flopscope._perm_group import SymmetryGroup

    if isinstance(arg, SymmetryGroup):
        return {"__symmetry_group__": arg.to_payload()}
    # Dtype-like args serialize to their canonical wire-name string: flopscope
    # dtype labels/objects, Python builtin types (``float``), numpy scalar types
    # (``np.float64``), and numpy dtype / new-style DType objects. Strings pass
    # through unchanged below (the server accepts dtype strings directly).
    if not isinstance(arg, str):
        _wire = _resolve_dtype_wire_name(arg)
        if _wire is not None:
            return _wire
    if isinstance(arg, (list, tuple)):
        return [_encode_arg(item) for item in arg]
    return arg
