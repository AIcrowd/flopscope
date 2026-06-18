"""Numpy-free dtype objects for the flopscope client.

Each dtype name is dual-purpose, mirroring numpy:

* callable like a scalar constructor — ``fnp.float32(0.5)`` (dispatches to the
  server via the existing ``array(value, dtype=...)`` create-from-data path and
  returns a 0-d ``RemoteArray`` of that dtype, 0 FLOPs);
* usable as a ``dtype=`` / ``astype()`` label — normalized to its wire string.

A ``_DtypeLabel`` is the scalar *type* (``fnp.float32 != "float32"``, like
``np.float32``); ``dtype("float32")`` returns a ``_DType`` that *does* equal the
string (like ``np.dtype``). The client ships without numpy, so nothing here
imports numpy.
"""

from __future__ import annotations

from typing import Any

from flopscope._remote_array import _DTYPE_INFO

# Public attribute name -> canonical wire name (keys of _DTYPE_INFO).
_PUBLIC_TO_WIRE = {
    "float16": "float16", "float32": "float32", "float64": "float64",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "bool_": "bool",
    "complex64": "complex64", "complex128": "complex128",
}

# Accepted string spellings -> canonical wire name.
_STRING_ALIASES: dict[str, str] = {name: name for name in _DTYPE_INFO}
_STRING_ALIASES["bool_"] = "bool"


class _DtypeLabel:
    """A numpy-free dtype scalar type: callable + usable as a dtype label."""

    __slots__ = ("name",)

    def __init__(self, wire_name: str) -> None:
        self.name = wire_name

    @property
    def _flopscope_dtype_name(self) -> str:
        return self.name

    def __call__(self, value: Any):
        # Lazy import: ``array`` is defined in flopscope/__init__.py.
        from flopscope import array

        return array(value, dtype=self.name)

    def __repr__(self) -> str:
        return f"flopscope.numpy.{self.name}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DtypeLabel) and other.name == self.name

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(("_DtypeLabel", self.name))


class _DType:
    """numpy-free analog of ``np.dtype``: EQUALS its string name."""

    __slots__ = ("name",)

    def __init__(self, wire_name: str) -> None:
        self.name = wire_name

    @property
    def _flopscope_dtype_name(self) -> str:
        return self.name

    @property
    def itemsize(self) -> int:
        return _DTYPE_INFO[self.name][1]

    def __repr__(self) -> str:
        return f"dtype('{self.name}')"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (_DType, _DtypeLabel)):
            return other.name == self.name
        if isinstance(other, str):
            return _STRING_ALIASES.get(other, other) == self.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


def _normalize_dtype(spec: Any) -> str:
    """Return the canonical wire dtype string for *spec*.

    Accepts a ``_DtypeLabel``, a ``_DType``, or a (possibly aliased) string.
    Raises ``TypeError`` for anything else.
    """
    name = getattr(spec, "_flopscope_dtype_name", None)
    if isinstance(name, str):
        return name
    if isinstance(spec, str):
        wire = _STRING_ALIASES.get(spec)
        if wire is None:
            raise TypeError(f"Unknown dtype: {spec!r}")
        return wire
    raise TypeError(f"Cannot interpret {spec!r} as a flopscope dtype")


def dtype(spec: Any) -> _DType:
    """numpy-free ``np.dtype`` analog."""
    return _DType(_normalize_dtype(spec))


# --- The 14 dtype label instances ---
float16 = _DtypeLabel("float16")
float32 = _DtypeLabel("float32")
float64 = _DtypeLabel("float64")
int8 = _DtypeLabel("int8")
int16 = _DtypeLabel("int16")
int32 = _DtypeLabel("int32")
int64 = _DtypeLabel("int64")
uint8 = _DtypeLabel("uint8")
uint16 = _DtypeLabel("uint16")
uint32 = _DtypeLabel("uint32")
uint64 = _DtypeLabel("uint64")
bool_ = _DtypeLabel("bool")
complex64 = _DtypeLabel("complex64")
complex128 = _DtypeLabel("complex128")

#: Public attribute name -> label instance (used by __init__ and __all__).
_DTYPE_LABELS: dict[str, _DtypeLabel] = {
    public: globals()[public] for public in _PUBLIC_TO_WIRE
}
