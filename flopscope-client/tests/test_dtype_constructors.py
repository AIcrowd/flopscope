"""Tests for the numpy-free dtype object model (flopscope._dtypes)."""

from __future__ import annotations

from unittest.mock import patch

import flopscope._remote_array as ra_mod
import pytest

import flopscope as fnp
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

    def test_label_dtype_equality_is_symmetric(self):
        # Both directions of label == dtype("...") must be True.
        assert _dtypes.float32 == dtype("float32")
        assert dtype("float32") == _dtypes.float32
        # label != string (NotImplemented falls back to identity, so False).
        assert (_dtypes.float32 == "float32") is False
        # label != different label.
        assert _dtypes.float32 != _dtypes.float64

    def test_bool_label_wire_name_is_bool(self):
        assert _dtypes.bool_.name == "bool"

    def test_all_fourteen_labels_present(self):
        for name in [
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
            "bool_",
            "complex64",
            "complex128",
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


class _FakeConn:
    """Captures requests and returns a canned create-from-data response."""

    def __init__(self, response):
        self.response = response
        self.requests: list[bytes] = []

    def send_recv(self, raw: bytes):
        self.requests.append(raw)
        return self.response


class TestEncodeArgDtype:
    def test_encode_arg_serializes_label_to_wire_name(self):
        assert ra_mod._encode_arg(fnp.float32) == "float32"
        assert ra_mod._encode_arg(fnp.bool_) == "bool"

    def test_encode_arg_serializes_dtype_object(self):
        assert ra_mod._encode_arg(dtype("int32")) == "int32"

    def test_encode_arg_passes_through_plain_values(self):
        assert ra_mod._encode_arg(0.5) == 0.5
        assert ra_mod._encode_arg("hello") == "hello"


class TestConstructorDispatch:
    def test_float32_constructor_returns_remote_array(self):
        fake = _FakeConn({"result": {"id": "h1", "shape": [], "dtype": "float32"}})
        with patch("flopscope.get_connection", return_value=fake):
            out = fnp.float32(0.5)
        assert out.dtype == "float32"
        assert out.shape == ()

    def test_original_failing_idiom_does_not_raise_typeerror(self):
        # fnp.multiply(arr, fnp.float32(0.5 * beta)) — the eval failure.
        create = {"result": {"id": "h1", "shape": [], "dtype": "float32"}}
        mul = {"result": {"id": "h2", "shape": [3], "dtype": "float32"}}

        class _Seq:
            def __init__(self):
                self.calls = 0

            def send_recv(self, raw):
                self.calls += 1
                return create if self.calls == 1 else mul

        seq = _Seq()
        arr = ra_mod.RemoteArray(handle_id="a", shape=(3,), dtype="float32")
        with patch("flopscope.get_connection", return_value=seq):
            scaled = fnp.float32(0.5)
            result = fnp.multiply(arr, scaled)
        assert result.dtype == "float32"


class TestAbstractScalarErrors:
    @pytest.mark.parametrize("name", ["floating", "integer", "number"])
    def test_clear_error(self, name):
        with pytest.raises(AttributeError) as exc:
            getattr(fnp, name)
        assert "abstract scalar type" in str(exc.value)

    def test_error_is_not_opaque(self, name="floating"):
        with pytest.raises(AttributeError) as exc:
            getattr(fnp, name)
        assert "has no attribute" not in str(exc.value)


class TestFinfoIinfo:
    def test_finfo_float32(self):
        fi = fnp.finfo(fnp.float32)
        assert fi.eps == 1.1920928955078125e-07
        assert fi.bits == 32
        assert fi.max == 3.4028234663852886e38

    def test_finfo_float64_eps(self):
        assert fnp.finfo(fnp.float64).eps == 2.220446049250313e-16

    def test_finfo_accepts_string(self):
        assert fnp.finfo("float32").bits == 32

    def test_finfo_rejects_int(self):
        with pytest.raises(ValueError):
            fnp.finfo(fnp.int32)

    def test_iinfo_int32(self):
        ii = fnp.iinfo(fnp.int32)
        assert ii.min == -2147483648
        assert ii.max == 2147483647
        assert ii.bits == 32

    def test_iinfo_uint8(self):
        ii = fnp.iinfo(fnp.uint8)
        assert ii.min == 0
        assert ii.max == 255

    def test_iinfo_rejects_float(self):
        with pytest.raises(ValueError):
            fnp.iinfo(fnp.float32)


class TestNamespaceHygiene:
    _LEAKS = [
        "builtins",
        "struct",
        "get_connection",
        "encode_request",
        "iter_proxyable",
        "timed_dispatch",
        "BLACKLISTED",
    ]
    _PUBLIC = [
        "array",
        "zeros",
        "float32",
        "dtype",
        "linalg",
        "random",
        "BudgetContext",
    ]

    def test_internal_names_not_in_all(self):
        for n in self._LEAKS:
            assert n not in fnp.__all__, n

    def test_public_names_in_all(self):
        for n in self._PUBLIC:
            assert n in fnp.__all__, n

    def test_star_import_is_clean(self):
        ns: dict = {}
        exec("from flopscope import *", ns)
        for n in self._LEAKS:
            assert n not in ns, n
        assert "float32" in ns

    def test_numpy_module_all_matches(self):
        import flopscope.numpy as _np_mod

        assert set(_np_mod.__all__) == set(fnp.__all__)
