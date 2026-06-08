"""Unit tests for the stdlib .npy/.npz codec (no server, no numpy required)."""

import struct

import pytest

from flopscope import _codec


def _make_npy(descr, shape, buffer, fortran=False):
    header = f"{{'descr': {descr!r}, 'fortran_order': {fortran}, 'shape': {tuple(shape)!r}, }}"
    hb = header.encode("latin1")
    total = 10 + len(hb) + 1
    hb = hb + b" " * ((64 - total % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hb)) + hb + buffer


def test_read_npy_float64():
    buf = struct.pack("<3d", 1.0, 2.0, 3.0)
    dtype, shape, data = _codec.read_npy(_make_npy("<f8", (3,), buf))
    assert dtype == "float64"
    assert shape == (3,)
    assert data == buf


def test_read_npy_rejects_object_dtype():
    with pytest.raises(ValueError, match="not supported"):
        _codec.read_npy(_make_npy("|O", (1,), b"\x00" * 8))


def test_read_npy_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        _codec.read_npy(b"NOTNUMPY" + b"\x00" * 40)


def test_read_npy_rejects_fortran_order():
    buf = struct.pack("<4d", 1, 2, 3, 4)
    with pytest.raises(ValueError, match="C order|fortran"):
        _codec.read_npy(_make_npy("<f8", (2, 2), buf, fortran=True))


def test_write_then_read_npy_roundtrip():
    buf = struct.pack("<2i", 7, 9)
    blob = _codec.write_npy("int32", (2,), buf)
    dtype, shape, data = _codec.read_npy(blob)
    assert (dtype, shape, data) == ("int32", (2,), buf)


def test_npz_roundtrip_with_meta():
    buf = struct.pack("<2d", 1.0, 2.0)
    blob = _codec.write_npz(
        {"W": ("float64", (2,), buf)}, meta={"sizes": [2]}, compressed=False
    )
    arrays, meta = _codec.read_npz(blob)
    assert meta == {"sizes": [2]}
    assert arrays["W"] == ("float64", (2,), buf)


def test_npz_decompressed_size_guard():
    big = struct.pack("<d", 0.0) * 1
    blob = _codec.write_npz({"W": ("float64", (1,), big)}, meta=None, compressed=True)
    with pytest.raises(ValueError, match="too large"):
        _codec.read_npz(blob, max_bytes=4)


def test_interop_with_real_numpy(tmp_path):
    np = pytest.importorskip("numpy")
    p = tmp_path / "x.npy"
    np.save(str(p), np.arange(6, dtype=np.float32).reshape(2, 3))
    dtype, shape, data = _codec.read_npy(p.read_bytes())
    assert dtype == "float32" and shape == (2, 3)
    blob = _codec.write_npy("float64", (2,), struct.pack("<2d", 4.0, 5.0))
    q = tmp_path / "y.npy"
    q.write_bytes(blob)
    np.testing.assert_array_equal(np.load(str(q)), np.array([4.0, 5.0]))


def _npy_with_header(header: str, buffer: bytes = b"") -> bytes:
    hb = header.encode("latin1")
    total = 10 + len(hb) + 1
    hb = hb + b" " * ((64 - total % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hb)) + hb + buffer


def test_read_npy_rejects_negative_shape():
    with pytest.raises(ValueError, match="shape"):
        _codec.read_npy(_make_npy("<f8", (-1,), b"\x00" * 8))


def test_read_npy_rejects_bad_version():
    blob = b"\x93NUMPY\x03\x00" + struct.pack("<H", 64) + b" " * 62 + b"\n"
    with pytest.raises(ValueError, match="version"):
        _codec.read_npy(blob)


def test_read_npy_rejects_truncated_blob():
    with pytest.raises(ValueError, match="truncated|magic"):
        _codec.read_npy(b"\x93NUMPY\x01")


def test_read_npy_rejects_non_dict_header():
    with pytest.raises(ValueError, match="malformed|dict"):
        _codec.read_npy(_npy_with_header("[1, 2, 3]"))


def test_read_npy_normalizes_malformed_header_to_valueerror():
    # Broken literal -> ast.literal_eval raises SyntaxError; the codec must
    # normalize it to ValueError, never let it escape (DoS hardening).
    with pytest.raises(ValueError):
        _codec.read_npy(_npy_with_header("{'descr': '<f8', 'shape': (1,}"))


def test_read_npz_rejects_non_zip():
    # A non-zip blob must normalize to ValueError, not leak zipfile.BadZipFile.
    with pytest.raises(ValueError, match="zip"):
        _codec.read_npz(b"this is definitely not a zip archive")
