"""fnp.array() must accept array.array / buffer-protocol inputs (numpy parity).

Prod regression (subs 310414, 311239, 311356): participants build a stdlib
array.array('f', ...) C buffer for speed and pass it to fnp.array(); native
numpy-backed flopscope accepts the buffer protocol, the client rejected it with
"Cannot create array from array".

The ndarray cases below are #194: ``fnp.array(ndarray)`` was gated at rank 1
and ``fnp.asarray(ndarray)`` had no encoding at all. They live here rather than
in ``flopscope-client/tests/`` because the client package is numpy-free, so its
own venv cannot build an ndarray, a Fortran-order view, or a string dtype.

``np.array`` / ``np.arange`` / ``np.asfortranarray`` stay native in this
harness (see ``_patch_client._SKIP`` and ``_coerce``), so they really do build
numpy arrays here.
"""

from __future__ import annotations

import array as _array

import numpy as np
import pytest

import flopscope as fnp


def test_array_from_array_array_float32_with_dtype():
    buf = _array.array("f", [1.0, 2.0, 3.0, 4.0])
    out = fnp.array(buf, dtype="float32")
    assert type(out).__name__ == "RemoteArray"
    assert out.shape == (4,)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_array_from_array_array_infers_dtype_from_typecode():
    buf = _array.array("d", [1.5, 2.5, 3.5])  # 'd' -> float64
    out = fnp.array(buf)
    assert out.tolist() == [1.5, 2.5, 3.5]
    assert out.dtype == "float64"


def test_array_from_array_array_casts_when_dtype_differs():
    buf = _array.array("d", [1.0, 2.0, 3.0])  # double buffer
    out = fnp.array(buf, dtype="float32")  # cast to float32 (numpy parity)
    assert out.dtype == "float32"
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_array_from_memoryview():
    buf = _array.array("i", [10, 20, 30])
    out = fnp.array(memoryview(buf))
    assert out.tolist() == [10, 20, 30]


def test_array_rejects_raw_bytes_cleanly():
    # numpy makes a |S3 string scalar; flopscope has no string dtype, so reject
    # cleanly (not a cryptic downstream error, not a wrong uint8 array).
    for bad in (b"abc", bytearray(b"abc")):
        with pytest.raises(TypeError, match="Cannot create array from"):
            fnp.array(bad)


# ---------------------------------------------------------------------------
# ndarray input (#194)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_array_accepts_ndarray_at_any_rank(shape):
    src = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape)
    out = fnp.array(src)
    assert out.shape == shape
    assert out.tolist() == src.tolist()


@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 4)])
def test_asarray_accepts_ndarray_at_any_rank(shape):
    src = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape)
    out = fnp.asarray(src)
    assert out.shape == shape
    assert out.tolist() == src.tolist()


@pytest.mark.parametrize("ctor", ["array", "asarray"])
def test_fortran_order_ndarray_keeps_its_logical_values(ctor):
    """tobytes() is C-order for an F-order buffer, so the values must not
    transpose. This is the case the memoryview-only path most easily gets
    wrong."""
    src = np.asfortranarray(np.arange(6, dtype="float64").reshape(2, 3))
    assert not src.flags["C_CONTIGUOUS"]
    out = getattr(fnp, ctor)(src)
    assert out.shape == (2, 3)
    assert out.tolist() == src.tolist()


@pytest.mark.parametrize("ctor", ["array", "asarray"])
def test_non_contiguous_ndarray_keeps_its_logical_values(ctor):
    src = np.arange(32, dtype="float64").reshape(4, 8)[:, ::2]
    assert not src.flags["C_CONTIGUOUS"]
    out = getattr(fnp, ctor)(src)
    assert out.shape == (4, 4)
    assert out.tolist() == src.tolist()


def test_array_of_zero_d_ndarray_matches_numpy_rank():
    src = np.array(3.0)
    assert src.shape == ()
    assert fnp.array(src).shape == ()


def test_array_accepts_ndarray_with_dtype_cast():
    # Built with np.arange().reshape() rather than np.ones(): only the names in
    # _patch_client._SKIP stay native here, and `ones` is not one of them, so
    # np.ones() would hand back a RemoteArray and this would exercise the
    # pre-existing astype branch instead of the ndarray buffer path.
    src = np.arange(6, dtype="float64").reshape(2, 3)
    assert type(src).__name__ == "ndarray"
    out = fnp.array(src, dtype="float32")
    assert out.shape == (2, 3)
    assert out.dtype == "float32"
    assert out.tolist() == src.tolist()


@pytest.mark.parametrize(
    "bad",
    [
        np.array(["a", "b"]),  # unicode: buffer format '1w'
        np.ones(4, dtype=">f8"),  # byte-swapped: buffer format '>d'
        np.zeros(2, dtype=[("a", "f8"), ("b", "i4")]),  # structured: 'T{...}'
    ],
    ids=["unicode", "big_endian", "structured"],
)
def test_array_still_rejects_buffers_with_no_wire_dtype(bad):
    with pytest.raises(TypeError, match="Cannot create array from"):
        fnp.array(bad)


def test_asarray_still_rejects_a_string_ndarray():
    """A unicode ndarray has no wire dtype, so asarray refuses it as before.

    Pinned to the concrete class and message: a bare ``raises(Exception)``
    would also pass if the refusal degraded into an unrelated local crash.
    """
    from flopscope.errors import RemoteSerializationError

    with pytest.raises(RemoteSerializationError, match="cannot be sent to the remote"):
        fnp.asarray(np.array(["a", "b"]))
