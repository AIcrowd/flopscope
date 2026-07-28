"""Record what a backend actually did. Pure functions, no backend imports.

THE RULE: these functions RECORD, they do not NORMALISE. Materialising an array
via ``tolist()`` is fine *because* dtype, shape and the Python type are recorded
separately alongside it. Replacing those fields with just the materialised
value would make every dtype divergence between backends invisible, no matter
how many tests were layered on top of ``tests/client_compat`` afterwards.
"""

from __future__ import annotations

import builtins
import struct
from typing import Any

_BUILTIN_EXCEPTIONS = {
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
}


def fingerprint(value: Any) -> str:
    """Return a canonical, exact string fingerprint of *value*.

    Floats become their IEEE-754 bit pattern, so ``nan``, ``-0.0`` and last-ULP
    differences are all distinguishable. Every scalar carries a type tag, so
    ``1``, ``1.0`` and ``True`` never collide. String and bytes encodings carry
    an explicit length prefix so structural separators inside the content
    (``,``, ``[``, ``]``) can never be mistaken for container structure.

    The ``?:`` branch at the end is a last-resort, non-exact fallback for
    values of a type this module does not model (it falls back to ``repr``,
    which is not guaranteed to be injective or stable). Callers that need
    exactness — this harness's whole purpose — must materialise unmodelled
    types (e.g. numpy scalars) down to a modelled Python type before calling
    ``fingerprint``, the way ``observe_result`` does via ``_materialize``.
    """
    if isinstance(value, bool):
        return f"b:{value}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return "f:" + struct.pack(">d", value).hex()
    if isinstance(value, complex):
        return (
            "c:"
            + struct.pack(">d", value.real).hex()
            + ":"
            + struct.pack(">d", value.imag).hex()
        )
    if isinstance(value, str):
        return f"s:{len(value)}:{value}"
    if isinstance(value, (bytes, bytearray)):
        hexdigits = bytes(value).hex()
        return f"y:{len(hexdigits)}:{hexdigits}"
    if value is None:
        return "n:"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(fingerprint(item) for item in value) + "]"
    return f"?:{type(value).__name__}:{value!r}"


def _container_of(value: Any) -> str:
    if isinstance(value, tuple):
        fields = getattr(value, "_fields", None)
        if fields is not None:
            return f"namedtuple:{type(value).__name__}({','.join(fields)})"
        return "tuple"
    if isinstance(value, list):
        return "list"
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return "array"
    return "scalar"


def _materialize(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            # A `tolist` attribute that promises the ndarray protocol but
            # blows up when actually called (e.g. `value` is a class, so
            # `tolist` is an unbound function expecting `self`) must not
            # take the whole recording down with it. Fall back to the raw
            # value, same as if `tolist` had never been callable.
            return value
    if isinstance(value, (list, tuple)):
        return [_materialize(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        # Zero-argument `.item()` is how numpy scalars (np.float32, np.int64,
        # ...) unwrap to a plain Python scalar; they aren't subclasses of the
        # builtin types `fingerprint` models, so without this they'd fall
        # through to its inexact `?:` fallback.
        return item()
    return value


#: The two backends' concrete array-wrapper classes. Two implementations of
#: the same array type are necessarily two different classes, so comparing
#: their raw names would report a "pytype" divergence on essentially every
#: array-returning case, drowning any real signal. Collapsed here to one
#: shared token. Scalar wrapper classes (e.g. the client's ``RemoteScalar``)
#: are deliberately excluded from this collapse: a plain Python ``bool`` or
#: ``int`` on one backend against a wrapper object on the other means a value
#: failed to unwrap the way it should have, which is a real defect that
#: "pytype" exists to catch.
_ARRAY_WRAPPER_CLASS_NAMES = frozenset({"FlopscopeArray", "RemoteArray"})
_ARRAY_WRAPPER_TOKEN = "<array-wrapper>"


def _pytype_of(value: Any) -> str:
    name = type(value).__name__
    return _ARRAY_WRAPPER_TOKEN if name in _ARRAY_WRAPPER_CLASS_NAMES else name


#: A real dtype's ``str()`` is always a short label (``"float32"``,
#: ``"dtype('complex128')"``, ...). A misattributed descriptor's default
#: repr (e.g. ``"<attribute 'dtype' of 'X' objects>"`` for a class's
#: getset_descriptor, ``"<property object at 0x...>"`` for a class's
#: property) is not, which is how this tells the two apart without
#: hard-coding any backend's dtype type: both are the generic
#: ``<... at 0x...>`` / ``<... of ... objects>`` shape Python's default
#: ``__repr__``/``__str__`` produces for objects with no custom string form,
#: i.e. angle-bracket-wrapped end to end. A genuine dtype string is never
#: wrapped like that (numpy's own byte-order-prefixed forms, e.g.
#: ``"<U10"``, start with ``<`` but do not end with ``>``).
_MAX_PLAUSIBLE_DTYPE_LEN = 64


def _shape_of(value: Any) -> list | None:
    """Return ``value.shape`` as a list, or ``None`` if it is not genuinely one.

    ``value`` may be a class rather than an instance (a real defect this
    harness has observed from the registry under test), in which case a
    ``.shape`` attribute implemented as a C-level getset descriptor resolves
    to the descriptor object itself rather than a tuple. ``list()`` on that
    raises, so it must be validated before use, not just null-checked.
    """
    shape = getattr(value, "shape", None)
    if not isinstance(shape, (tuple, list)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return None
    return list(shape)


def _dtype_of(value: Any) -> str | None:
    """Return ``str(value.dtype)``, or ``None`` if that is not a plausible dtype.

    Same hazard as ``_shape_of``: on a misbehaving ``value`` the ``dtype``
    attribute can be a descriptor object rather than a real dtype. ``str()``
    on it will not raise, so this also rejects the implausible-looking
    result (long, multi-line, or otherwise not dtype-shaped) rather than
    recording it as if it were faithful.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    try:
        text = str(dtype)
    except Exception:
        return None
    if not isinstance(text, str) or not text:
        return None
    if len(text) > _MAX_PLAUSIBLE_DTYPE_LEN or "\n" in text:
        return None
    if text.startswith("<") and text.endswith(">"):
        return None
    return text


def observe_result(value: Any, flops: int) -> dict:
    """Record a successful return."""
    return {
        "outcome": "returned",
        "pytype": _pytype_of(value),
        "dtype": _dtype_of(value),
        "shape": _shape_of(value),
        "container": _container_of(value),
        "value": fingerprint(_materialize(value)),
        "flops": flops,
    }


def observe_exception(exc: BaseException, flops: int) -> dict:
    """Record a raised exception by CLASS, not by message.

    Messages legitimately differ between backends and would be pure noise; the
    class is the contract. The message is still carried for triage.
    """
    bases = [
        cls.__name__
        for cls in type(exc).__mro__[1:]
        if cls.__name__ in _BUILTIN_EXCEPTIONS
    ]
    return {
        "outcome": "raised",
        "exc_type": type(exc).__name__,
        "exc_bases": bases,
        "exc_msg": str(exc),
        "flops": flops,
    }


def observe_record_failure(exc: BaseException, flops: int) -> dict:
    """Record that the harness itself failed to describe a result.

    A last-resort outcome for when ``observe_result`` itself raises despite
    its own defenses (an object hostile enough to defeat ``_shape_of``,
    ``_dtype_of`` and ``_materialize`` alike). The case is still recorded,
    with the FLOP delta preserved, rather than taking the worker down.
    """
    return {
        "outcome": "record_failed",
        "exc_type": type(exc).__name__,
        "flops": flops,
    }


def observe_timeout(flops: int) -> dict:
    return {"outcome": "timeout", "flops": flops}


def observe_worker_died() -> dict:
    return {"outcome": "worker_died", "flops": 0}
