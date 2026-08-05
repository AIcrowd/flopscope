"""Compatibility coverage for dtype normalization in the NumPy-free client."""

from __future__ import annotations

import sys

import numpy as np
from flopscope._remote_array import _resolve_dtype_wire_name

from flopscope import _dtypes


def test_numpy_scalar_types_remain_supported():
    assert _resolve_dtype_wire_name(np.float32) == "float32"
    assert _resolve_dtype_wire_name(np.int64) == "int64"


def test_lookalike_scalar_type_is_not_trusted():
    float32 = type("float32", (), {})
    assert _resolve_dtype_wire_name(float32) is None


def test_numpy_dtype_instances_remain_supported():
    assert _resolve_dtype_wire_name(np.dtype("float64")) == "float64"
    assert _resolve_dtype_wire_name(np.dtypes.Float32DType()) == "float32"


def test_client_dtypes_resolve_without_importing_numpy(monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert _resolve_dtype_wire_name(_dtypes.float32) == "float32"
    assert _resolve_dtype_wire_name(float) == "float64"
    assert "numpy" not in sys.modules
