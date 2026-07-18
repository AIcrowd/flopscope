"""File I/O for the in-process flopscope package — numpy-backed, pickle-free.

`load`/`save`/`savez`/`savez_compressed` move only inert numeric arrays plus an
optional inert JSON `__meta__` block. `allow_pickle` is always False and object
dtype is rejected, so loading a file can never execute code. `load` costs
0 FLOPs (data ingress is free), matching the competition client; `save`/
`savez`/`savez_compressed` bill 4*size (data egress: the elements the call
writes to disk).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as _np

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._validation import require_budget

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


@_counted_wrapper
def save(file: str, arr: Any) -> None:
    """Save a single array to .npy. Cost: 4*numel(arr)."""
    budget = require_budget()
    base = _np.asarray(_to_base_ndarray(arr))
    _check_dtype(base, where="save")
    with budget.deduct(
        "save",
        flop_cost=4 * max(int(base.size), 1),
        subscripts=None,
        shapes=(base.shape,),
        dtypes=(base.dtype,),
    ):
        _call_numpy(_np.save, file, base, allow_pickle=False)


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


def _savez_billed_arrays(converted: dict[str, _np.ndarray]) -> list[_np.ndarray]:
    """The saved arrays that count toward billing (everything but __meta__)."""
    return [v for k, v in converted.items() if k != _META_KEY]


@_counted_wrapper
def savez(file: str, **arrays: Any) -> None:
    """Save multiple named arrays (+ optional __meta__ dict) to .npz. Cost: 4*sum(numel)."""
    budget = require_budget()
    converted = _prepare(arrays)
    billed = _savez_billed_arrays(converted)
    with budget.deduct(
        "savez",
        flop_cost=4 * max(sum(int(v.size) for v in billed), 1),
        subscripts=None,
        shapes=(),
        dtypes=tuple(v.dtype for v in billed),
    ):
        _call_numpy(_np.savez, file, **converted)  # type: ignore[arg-type]


@_counted_wrapper
def savez_compressed(file: str, **arrays: Any) -> None:
    """Save multiple named arrays (+ optional __meta__ dict) to compressed .npz.

    Cost: 4*sum(numel).
    """
    budget = require_budget()
    converted = _prepare(arrays)
    billed = _savez_billed_arrays(converted)
    with budget.deduct(
        "savez_compressed",
        flop_cost=4 * max(sum(int(v.size) for v in billed), 1),
        subscripts=None,
        shapes=(),
        dtypes=tuple(v.dtype for v in billed),
    ):
        _call_numpy(_np.savez_compressed, file, **converted)  # type: ignore[arg-type]
