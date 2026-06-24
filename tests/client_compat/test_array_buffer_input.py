"""fnp.array() must accept array.array / buffer-protocol inputs (numpy parity).

Prod regression (subs 310414, 311239, 311356): participants build a stdlib
array.array('f', ...) C buffer for speed and pass it to fnp.array(); native
numpy-backed flopscope accepts the buffer protocol, the client rejected it with
"Cannot create array from array".
"""

from __future__ import annotations

import array as _array

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
