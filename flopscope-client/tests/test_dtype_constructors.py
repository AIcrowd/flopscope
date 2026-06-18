"""Tests for the numpy-free dtype object model (flopscope._dtypes)."""

from __future__ import annotations

import pytest

from flopscope import _dtypes
from flopscope._dtypes import (
    _DTYPE_LABELS,
    _DtypeLabel,
    _normalize_dtype,
    dtype,
)


class TestDtypeLabel:
    def test_label_is_callable(self):
        assert callable(_dtypes.float32)

    def test_label_repr_and_str(self):
        assert str(_dtypes.float32) == "float32"
        assert "float32" in repr(_dtypes.float32)

    def test_label_not_equal_to_string(self):
        # Matches numpy/full flopscope: the scalar TYPE != its string name.
        assert (_dtypes.float32 == "float32") is False

    def test_label_equal_to_sibling(self):
        assert _dtypes.float32 == _DTYPE_LABELS["float32"]
        assert _dtypes.float32 != _dtypes.float64

    def test_bool_label_wire_name_is_bool(self):
        assert _dtypes.bool_.name == "bool"

    def test_all_fourteen_labels_present(self):
        for name in [
            "float16", "float32", "float64",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "bool_", "complex64", "complex128",
        ]:
            label = getattr(_dtypes, name)
            assert isinstance(label, _DtypeLabel)


class TestNormalizeDtype:
    def test_normalize_label(self):
        assert _normalize_dtype(_dtypes.float32) == "float32"

    def test_normalize_string(self):
        assert _normalize_dtype("int64") == "int64"

    def test_normalize_bool_alias(self):
        assert _normalize_dtype("bool_") == "bool"

    def test_normalize_dtype_object(self):
        assert _normalize_dtype(dtype("float32")) == "float32"

    def test_normalize_rejects_unknown_string(self):
        with pytest.raises(TypeError):
            _normalize_dtype("float128")

    def test_normalize_rejects_nonsense(self):
        with pytest.raises(TypeError):
            _normalize_dtype(object())


class TestDtypeObject:
    def test_dtype_equals_string(self):
        assert dtype("float32") == "float32"

    def test_dtype_itemsize(self):
        assert dtype("float32").itemsize == 4
        assert dtype("complex128").itemsize == 16

    def test_dtype_accepts_label(self):
        assert dtype(_dtypes.float32) == "float32"
