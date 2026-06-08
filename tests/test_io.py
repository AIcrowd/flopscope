"""Tests for flopscope.numpy file I/O (load/save/savez/savez_compressed)."""

import struct

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def test_savez_load_roundtrip(tmp_path):
    p = tmp_path / "w.npz"
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.array([1, 2, 3], dtype=np.int32)
    fnp.savez(str(p), a=a, b=b)
    out = fnp.load(str(p))
    np.testing.assert_array_equal(np.asarray(out["a"]), a)
    np.testing.assert_array_equal(np.asarray(out["b"]), b)


def test_save_load_single_npy(tmp_path):
    p = tmp_path / "x.npy"
    a = np.array([[1.5, 2.5]], dtype=np.float32)
    fnp.save(str(p), a)
    np.testing.assert_array_equal(np.asarray(fnp.load(str(p))), a)


def test_meta_roundtrip(tmp_path):
    p = tmp_path / "m.npz"
    fnp.savez(str(p), W=np.zeros((2, 2)), __meta__={"sizes": [2, 2], "name": "x"})
    out = fnp.load(str(p))
    assert out["__meta__"] == {"sizes": [2, 2], "name": "x"}


def test_numpy_can_read_our_npz(tmp_path):
    p = tmp_path / "interop.npz"
    fnp.savez(str(p), W=np.ones((3,), dtype=np.float64))
    with np.load(str(p)) as z:
        np.testing.assert_array_equal(z["W"], np.ones((3,)))


def test_we_can_read_numpy_npz(tmp_path):
    p = tmp_path / "fromnumpy.npz"
    np.savez(str(p), W=np.full((2,), 7.0))
    out = fnp.load(str(p))
    np.testing.assert_array_equal(np.asarray(out["W"]), np.full((2,), 7.0))


def test_load_is_free(tmp_path):
    p = tmp_path / "free.npz"
    fnp.savez(str(p), W=np.zeros((100, 100)))
    with flops.BudgetContext(flop_budget=1_000_000) as budget:
        fnp.load(str(p))
        assert budget.flops_used == 0


def test_load_rejects_object_array_pickle(tmp_path):
    # Build a malicious .npy with an object dtype header + a pickle-ish payload.
    # Assert load REFUSES (object dtype) and never executes/loads it.
    sentinel = tmp_path / "PWNED"
    payload = b"\x80\x04\x95\x2a\x00\x00\x00\x00\x00\x00\x00\x8c\x02os"
    descr = "|O"
    header = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': (1,), }}"
    header_b = header.encode("latin1")
    total = 10 + len(header_b) + 1
    pad = (64 - total % 64) % 64
    header_b = header_b + b" " * pad + b"\n"
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_b)) + header_b + payload
    p = tmp_path / "evil.npy"
    p.write_bytes(npy)
    with pytest.raises((ValueError, OSError)):
        fnp.load(str(p))
    assert not sentinel.exists()


def test_savez_rejects_reserved_meta_array(tmp_path):
    p = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="reserved"):
        fnp.savez(str(p), __meta__=np.zeros((2,)))
