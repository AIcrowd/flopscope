"""Compatibility coverage for dtype normalization in the NumPy-free client."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from flopscope._remote_array import _resolve_dtype_wire_name

import flopscope
from flopscope import _dtypes
from flopscope.errors import RemoteCallbackError


def test_numpy_scalar_types_remain_supported():
    assert _resolve_dtype_wire_name(np.float32) == "float32"
    assert _resolve_dtype_wire_name(np.int64) == "int64"


def test_lookalike_scalar_type_is_not_trusted():
    float32 = type("float32", (), {})
    assert _resolve_dtype_wire_name(float32) is None


def test_numpy_dtype_instances_remain_supported():
    assert _resolve_dtype_wire_name(np.dtype("float64")) == "float64"
    assert _resolve_dtype_wire_name(np.dtypes.Float32DType()) == "float32"


def test_hostile_class_property_is_not_executed_during_dtype_resolution(
    monkeypatch,
):
    class HostileDtype:
        def __init__(self):
            self.class_calls = 0
            self.name_calls = 0

        @property
        def __class__(self):
            self.class_calls += 1
            raise AssertionError("participant __class__ property executed")

        @property
        def name(self):
            self.name_calls += 1
            raise AssertionError("participant name property executed")

    spec = HostileDtype()
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteCallbackError, match="dtype.*descriptor"):
        flopscope.array([1.0], dtype=spec)
    assert spec.class_calls == 0
    assert spec.name_calls == 0
    assert network_calls == 0


def test_client_dtypes_resolve_without_importing_numpy(monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert _resolve_dtype_wire_name(_dtypes.float32) == "float32"
    assert _resolve_dtype_wire_name(float) == "float64"
    assert "numpy" not in sys.modules
