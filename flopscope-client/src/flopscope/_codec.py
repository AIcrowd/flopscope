"""Pure-stdlib .npy/.npz codec — no numpy, no pickle.

Converts between file bytes and (dtype_str, shape, raw_buffer) triples. Only
numeric dtypes are accepted; object dtype is rejected, so there is no code path
that can deserialize a Python object. The `.npy` v1.0 format is parsed by hand.
"""

from __future__ import annotations

import ast
import io
import json
import struct
import zipfile

# flopscope dtype string  <->  numpy descr (byte-order normalized on read)
_DTYPE_TO_DESCR = {
    "float64": "<f8",
    "float32": "<f4",
    "float16": "<f2",
    "int64": "<i8",
    "int32": "<i4",
    "int16": "<i2",
    "int8": "|i1",
    "uint64": "<u8",
    "uint32": "<u4",
    "uint16": "<u2",
    "uint8": "|u1",
    "bool": "|b1",
    "complex64": "<c8",
    "complex128": "<c16",
}
_ITEMSIZE = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint64": 8,
    "uint32": 4,
    "uint16": 2,
    "uint8": 1,
    "bool": 1,
    "complex64": 8,
    "complex128": 16,
}
_MAGIC = b"\x93NUMPY"
_META_KEY = "__meta__"
_DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # mirrors server MAX_ARRAY_BYTES


def _descr_to_dtype(descr: object) -> str:
    if not isinstance(descr, str):
        raise ValueError(f"invalid dtype descriptor {descr!r}: must be a string")
    # Normalize byte order: accept '<', '|', '=' (native LE); reject '>'.
    if descr and descr[0] == ">":
        raise ValueError(f"big-endian dtype {descr!r} is not supported")
    if descr and descr[0] in "<=|":
        body = descr[1:]
        for name, canon in _DTYPE_TO_DESCR.items():
            if canon[1:] == body:
                return name
    for name, canon in _DTYPE_TO_DESCR.items():
        if canon == descr:
            return name
    raise ValueError(f"dtype {descr!r} is not supported (numeric arrays only)")


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def read_npy(blob: bytes) -> tuple[str, tuple, bytes]:
    """Parse .npy bytes -> (dtype_str, shape, raw_buffer). Rejects object/fortran."""
    if blob[:6] != _MAGIC:
        raise ValueError("not a .npy file (bad magic)")
    if len(blob) < 10:
        raise ValueError("truncated .npy file (header too short)")
    major = blob[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", blob[8:10])
        start = 10
    elif major == 2:
        if len(blob) < 12:
            raise ValueError("truncated .npy file (header too short)")
        (hlen,) = struct.unpack("<I", blob[8:12])
        start = 12
    else:
        raise ValueError(f"unsupported .npy version {major}; only 1 and 2 supported")
    if len(blob) < start + hlen:
        raise ValueError("truncated .npy header")
    header = blob[start : start + hlen].decode("latin1")
    try:
        meta = ast.literal_eval(header)  # safe: literals only, never executes code
    except Exception as exc:  # SyntaxError, ValueError, etc. → uniform ValueError
        raise ValueError(f"malformed .npy header: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("malformed .npy header: not a dict")
    if meta.get("fortran_order"):
        raise ValueError("fortran-order arrays unsupported; re-save in C order")
    descr = meta.get("descr")
    raw_shape = meta.get("shape")
    if descr is None or raw_shape is None:
        raise ValueError("malformed .npy header: missing 'descr' or 'shape'")
    dtype = _descr_to_dtype(descr)
    shape = tuple(raw_shape)
    if not all(isinstance(d, int) and d >= 0 for d in shape):
        raise ValueError(f"invalid shape {shape!r}: dims must be non-negative ints")
    data = blob[start + hlen :]
    expected = _prod(shape) * _ITEMSIZE[dtype]
    if len(data) < expected:
        raise ValueError("truncated .npy buffer")
    return dtype, shape, data[:expected]


def write_npy(dtype: str, shape, buffer: bytes) -> bytes:
    if dtype not in _DTYPE_TO_DESCR:
        raise ValueError(f"dtype {dtype!r} is not supported")
    if shape:
        shape_repr = "(" + ", ".join(str(d) for d in shape)
        shape_repr += ",)" if len(shape) == 1 else ")"
    else:
        shape_repr = "()"
    header = f"{{'descr': '{_DTYPE_TO_DESCR[dtype]}', 'fortran_order': False, 'shape': {shape_repr}, }}"
    hb = header.encode("latin1")
    total = 10 + len(hb) + 1
    hb = hb + b" " * ((64 - total % 64) % 64) + b"\n"
    return _MAGIC + b"\x01\x00" + struct.pack("<H", len(hb)) + hb + buffer


def read_npz(blob: bytes, *, max_bytes: int = _DEFAULT_MAX_BYTES):
    """Parse .npz bytes -> (arrays{name:(dtype,shape,buf)}, meta_dict_or_None)."""
    arrays: dict[str, tuple] = {}
    meta = None
    total = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid .npz (zip) file: {exc}") from exc
    with zf:
        for info in zf.infolist():
            name = info.filename
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            total += info.file_size
            if total > max_bytes:
                raise ValueError(
                    f"archive too large: {total} bytes exceeds {max_bytes} limit"
                )
            try:
                member = zf.read(name)  # decompresses into memory
            except zipfile.BadZipFile as exc:
                raise ValueError(f"corrupt .npz member {name!r}: {exc}") from exc
            dtype, shape, data = read_npy(member)
            if key == _META_KEY:
                try:
                    meta = json.loads(bytes(data).decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"malformed __meta__ JSON: {exc}") from exc
                continue
            arrays[key] = (dtype, shape, data)
    return arrays, meta


def write_npz(arrays: dict[str, tuple], *, meta, compressed: bool) -> bytes:
    """arrays: {name:(dtype,shape,buffer)} -> .npz bytes. meta: dict|None."""
    out = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(out, "w", compression=mode) as zf:
        for key, (dtype, shape, buffer) in arrays.items():
            if key == _META_KEY:
                raise ValueError(f"'{_META_KEY}' is a reserved array name")
            zf.writestr(key + ".npy", write_npy(dtype, shape, buffer))
        if meta is not None:
            blob = json.dumps(meta).encode("utf-8")
            zf.writestr(_META_KEY + ".npy", write_npy("uint8", (len(blob),), blob))
    return out.getvalue()
