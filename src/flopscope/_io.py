"""File I/O for the in-process flopscope package — numpy-backed, pickle-free.

`load`/`save`/`savez`/`savez_compressed` move only inert numeric arrays plus an
optional inert JSON `__meta__` block. `allow_pickle` is always False and object
dtype is rejected, so loading a file can never execute code. These ops cost
0 FLOPs (data ingress is free), matching the competition client.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as _np

from flopscope._ndarray import FlopscopeArray, _to_base_ndarray

_WHITELIST = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "complex64",
        "complex128",
    }
)
_META_KEY = "__meta__"


def _check_dtype(arr: _np.ndarray, *, where: str) -> None:
    if arr.dtype.kind == "O" or str(arr.dtype) not in _WHITELIST:
        raise ValueError(
            f"{where}: dtype {arr.dtype!r} is not supported. Only numeric arrays "
            f"may be saved/loaded (object dtype would require pickle)."
        )


def _wrap(arr: _np.ndarray) -> FlopscopeArray:
    return arr.view(FlopscopeArray)


def _decode_meta(arr: _np.ndarray) -> dict:
    return json.loads(bytes(arr.tobytes()).decode("utf-8"))


def load(file: str) -> Any:
    """Load arrays from a .npy/.npz file. Cost: 0 FLOPs. Never unpickles."""
    obj = _np.load(file, allow_pickle=False)
    if isinstance(obj, _np.lib.npyio.NpzFile):
        out: dict[str, Any] = {}
        try:
            for key in obj.files:
                arr = obj[key]
                if key == _META_KEY:
                    out[_META_KEY] = _decode_meta(arr)
                    continue
                _check_dtype(arr, where=f"load[{key}]")
                out[key] = _wrap(arr)
        finally:
            obj.close()
        return out
    _check_dtype(obj, where="load")
    return _wrap(obj)


def save(file: str, arr: Any) -> None:
    """Save a single array to .npy. Cost: 0 FLOPs."""
    base = _np.asarray(_to_base_ndarray(arr))
    _check_dtype(base, where="save")
    _np.save(file, base, allow_pickle=False)


def _prepare(arrays: dict[str, Any]) -> dict[str, _np.ndarray]:
    meta = arrays.pop(_META_KEY, None)
    converted: dict[str, _np.ndarray] = {}
    for key, val in arrays.items():
        base = _np.asarray(_to_base_ndarray(val))
        _check_dtype(base, where=f"savez[{key}]")
        converted[key] = base
    if meta is not None:
        if not isinstance(meta, dict):
            raise ValueError(
                f"'{_META_KEY}' must be a JSON-serializable dict, got {type(meta).__name__!r}. "
                f"'{_META_KEY}' is reserved — pass a plain dict for metadata"
            )
        blob = json.dumps(meta).encode("utf-8")  # raises on non-JSON-safe values
        converted[_META_KEY] = _np.frombuffer(blob, dtype=_np.uint8).copy()
    return converted


def savez(file: str, **arrays: Any) -> None:
    """Save multiple named arrays (+ optional __meta__ dict) to .npz. 0 FLOPs."""
    _np.savez(file, **_prepare(arrays))  # type: ignore[arg-type]


def savez_compressed(file: str, **arrays: Any) -> None:
    """Save multiple named arrays (+ optional __meta__ dict) to compressed .npz."""
    _np.savez_compressed(file, **_prepare(arrays))  # type: ignore[arg-type]
