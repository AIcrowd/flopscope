"""Record what a backend actually did. Pure functions, no backend imports.

THE RULE: these functions RECORD, they do not NORMALISE. Materialising an array
via ``tolist()`` is fine *because* dtype, shape and the Python type are recorded
separately alongside it. Replacing those with the materialised value is what
blinded ``tests/client_compat`` to every dtype divergence in the 2026-07-25
audit.
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
    ``1``, ``1.0`` and ``True`` never collide.
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
        return f"s:{value}"
    if isinstance(value, (bytes, bytearray)):
        return "y:" + bytes(value).hex()
    if value is None:
        return "n:"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(fingerprint(item) for item in value) + "]"
    return f"?:{type(value).__name__}:{value!r}"


def _container_of(value: Any) -> str:
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        fields = ",".join(value._fields)
        return f"namedtuple:{type(value).__name__}({fields})"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return "array"
    return "scalar"


def _materialize(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, (list, tuple)):
        return [_materialize(item) for item in value]
    return value


def observe_result(value: Any, flops: int) -> dict:
    """Record a successful return."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return {
        "outcome": "returned",
        "pytype": type(value).__name__,
        "dtype": None if dtype is None else str(dtype),
        "shape": None if shape is None else list(shape),
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


def observe_timeout(flops: int) -> dict:
    return {"outcome": "timeout", "flops": flops}


def observe_worker_died() -> dict:
    return {"outcome": "worker_died", "flops": 0}
