"""File I/O for the in-process flopscope package — numpy-backed, pickle-free.

`load`/`save`/`savez`/`savez_compressed` move only inert numeric arrays plus an
optional inert JSON `__meta__` block. `allow_pickle` is always False and object
dtype is rejected, so loading a file can never execute code. `load` costs
0 FLOPs (data ingress is free), matching the competition client; `save`/
`savez`/`savez_compressed` bill 4*size (data egress: the elements the call
writes to disk, INCLUDING the serialized `__meta__` blob when present -- it
is written to the archive as a uint8 array like any other member (see
`_prepare`), so it bills the same per-byte egress cost. Excluding it would
let a participant round-trip unlimited data through `__meta__` for a flat,
size-independent cost).

`savez`/`savez_compressed` accept arrays positionally as well as by keyword,
matching `numpy.savez`: positional arrays are stored under auto-generated
member names `arr_0`, `arr_1`, ... in call order (see `_savez_merge_args`),
keyword arrays keep their given names, and a keyword name that collides with
a positional array's auto-generated name raises the same `ValueError` numpy
raises -- checked before any dtype conversion or billing happens, so a call
numpy would reject is never billed. A positional array bills identically to
the same array passed under its auto-generated keyword name.

`savez`/`savez_compressed` ALSO bill the archive's MEMBER NAMES -- the keyword
argument names themselves (or the auto-generated `arr_0`, `arr_1`, ... names
for positional arrays), plus the literal `"__meta__"` name when a meta
block is present. Those strings are written into the .npz archive and read
back verbatim by `load` at 0 FLOPs, exactly like array data, but a `.npz`
member name can be tens of thousands of bytes long -- so without billing them,
a participant could smuggle megabytes of data through many tiny arrays given
huge names instead of through the array values, at a near-zero, size-independent
cost. Folded into the same 4*size formula via `extra_egress_bytes` (see
`_bill_save_egress`).

`_bill_save_egress` is the single-sourced egress-billing formula: the
in-process wrappers below call it directly, and the flopscope-server request
handler (``flopscope_server._request_handler._handle_save``) imports and
calls the same function to bill the identical cost when a *remote*
flopscope-client dispatches a save/savez/savez_compressed round-trip (the
client writes the file locally; only the billing crosses the wire, as a
handle-only request with no array data).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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


def _bill_save_egress(
    op_name: str,
    arrays: Sequence[_np.ndarray],
    *,
    extra_egress_bytes: int = 0,
) -> None:
    """Deduct the data-egress cost of writing *arrays* to disk:
    4*(sum(numel) + sum(ndim*8)).

    Single-sourced so the server-side billing round-trip
    (``flopscope_server._request_handler._handle_save``) charges exactly the
    same formula as the in-process ``save``/``savez``/``savez_compressed``
    wrappers below -- an empty ``arrays`` sequence still floors at 1.

    Each array's ``.npy``/``.npz`` header also encodes its shape as one
    8-byte int per dimension -- a channel a participant fully controls (e.g.
    ``zeros((0, K))`` has 0 elements but an arbitrary ``K``), so it is billed
    alongside the element data (``ndim*8`` bytes per array) instead of riding
    along for free.

    ``extra_egress_bytes`` folds in egress that has no backing ndarray --
    ``savez``/``savez_compressed`` use it for the archive's member-NAME bytes
    (see module docstring): those strings are written to disk and read back
    by ``load`` just like array data, so they must count toward the same
    size-proportional bill instead of riding along for free.
    """
    shape_header_bytes = sum(a.ndim * 8 for a in arrays)
    require_budget().deduct(
        op_name,
        flop_cost=4
        * max(
            sum(int(a.size) for a in arrays) + shape_header_bytes + extra_egress_bytes,
            1,
        ),
        subscripts=None,
        shapes=tuple(a.shape for a in arrays),
        dtypes=tuple(a.dtype for a in arrays),
    )


@_counted_wrapper
def save(file: str, arr: Any) -> None:
    """Save a single array to .npy. Cost: 4*(numel + ndim*8)."""
    base = _np.asarray(_to_base_ndarray(arr))
    _check_dtype(base, where="save")
    _bill_save_egress("save", [base])
    _call_numpy(_np.save, file, base, allow_pickle=False)


def _savez_merge_args(args: tuple[Any, ...], arrays: dict[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword arrays into a single name->value mapping,
    mirroring ``numpy.savez``'s own naming (``numpy.lib._npyio_impl._savez``):
    positional arrays are named ``arr_0``, ``arr_1``, ... in call order and
    appended after the keyword arrays, which keep their given names and
    order.

    Raises the same ``ValueError`` numpy raises when a keyword name collides
    with a positional array's auto-generated name -- e.g. a positional array
    together with ``arr_0=...``. This is checked here, against the ORIGINAL
    keyword names only, before any dtype conversion or billing happens, so a
    call numpy would reject is never billed (positional auto-names can never
    collide with each other, only with a pre-existing keyword name, so a
    single membership check per positional index is equivalent to numpy's
    incremental insert-and-check loop).
    """
    for i in range(len(args)):
        key = f"arr_{i}"
        if key in arrays:
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
    merged = dict(arrays)
    for i, val in enumerate(args):
        merged[f"arr_{i}"] = val
    return merged


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
    """The arrays that count toward billing: every array in *converted*,
    including the serialized __meta__ blob when present.

    The __meta__ dict (if any) was already turned into a uint8 byte array by
    `_prepare` and is written to the archive exactly like any other named
    array, so it must be billed the same way -- it is real data written to
    disk. (Previously excluded here, which let a participant round-trip
    unlimited data through __meta__ for a flat, size-independent cost.)
    """
    return list(converted.values())


def _savez_name_bytes(converted: dict[str, _np.ndarray]) -> int:
    """Total UTF-8 byte length of every archive member NAME in *converted*
    (the savez kwargs keys, plus "__meta__" when a meta blob is present).

    Member names are written into the .npz archive and read back verbatim by
    `load` at 0 FLOPs, exactly like array data -- but a `.npz` member name can
    be tens of thousands of bytes long, so without billing them a participant
    could smuggle data through many tiny arrays given huge names instead of
    through the array values, at a near-zero, size-independent cost.
    """
    return sum(len(key.encode("utf-8")) for key in converted)


@_counted_wrapper
def savez(file: str, *args: Any, **arrays: Any) -> None:
    """Save multiple arrays (+ optional __meta__ dict) to .npz.

    Arrays given positionally are stored under auto-generated names
    ``arr_0``, ``arr_1``, ...; arrays given as keywords keep their given
    names (matching ``numpy.savez``). A keyword name that collides with a
    positional array's auto-generated name raises the same ``ValueError``
    numpy raises.

    Cost: 4*(sum(numel) + sum(len(member name bytes))), including any
    __meta__ blob's serialized byte length and the "__meta__" member name
    itself when present. A positional array bills identically to the same
    array passed under its auto-generated keyword name.
    """
    converted = _prepare(_savez_merge_args(args, arrays))
    billed = _savez_billed_arrays(converted)
    name_bytes = _savez_name_bytes(converted)
    # The server ingests the concatenated member-names as one 1-D uint8 blob;
    # its 8-byte shape header (ndim*8) must be billed in-process too so the two
    # paths stay byte-identical. Present whenever there is at least one name.
    names_shape_header = 8 if name_bytes > 0 else 0
    _bill_save_egress(
        "savez", billed, extra_egress_bytes=name_bytes + names_shape_header
    )
    _call_numpy(_np.savez, file, **converted)  # type: ignore[arg-type]


@_counted_wrapper
def savez_compressed(file: str, *args: Any, **arrays: Any) -> None:
    """Save multiple arrays (+ optional __meta__ dict) to compressed .npz.

    Arrays given positionally are stored under auto-generated names
    ``arr_0``, ``arr_1``, ...; arrays given as keywords keep their given
    names (matching ``numpy.savez_compressed``). A keyword name that
    collides with a positional array's auto-generated name raises the same
    ``ValueError`` numpy raises.

    Cost: 4*(sum(numel) + sum(len(member name bytes))). A positional array
    bills identically to the same array passed under its auto-generated
    keyword name.
    """
    converted = _prepare(_savez_merge_args(args, arrays))
    billed = _savez_billed_arrays(converted)
    name_bytes = _savez_name_bytes(converted)
    # The server ingests the concatenated member-names as one 1-D uint8 blob;
    # its 8-byte shape header (ndim*8) must be billed in-process too so the two
    # paths stay byte-identical. Present whenever there is at least one name.
    names_shape_header = 8 if name_bytes > 0 else 0
    _bill_save_egress(
        "savez_compressed",
        billed,
        extra_egress_bytes=name_bytes + names_shape_header,
    )
    _call_numpy(_np.savez_compressed, file, **converted)  # type: ignore[arg-type]
