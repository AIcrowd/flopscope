"""Dtype-rate table: loading, unit mode, fail-closed lookup."""

import pytest

import flopscope._weights as weights_module
from flopscope._weights import get_dtype_rate, load_weights, reset_weights
from flopscope.errors import UnsupportedDtypeError

EXPECTED_PACKAGED_RATES = {
    "bool": 1.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int16": 1.0,
    "uint16": 1.0,
    "float16": 1.0,
    "int32": 1.0,
    "uint32": 1.0,
    "float32": 1.0,
    "int64": 2.0,
    "uint64": 2.0,
    "float64": 2.0,
    "complex64": 1.0,  # component width: float32
    "complex128": 2.0,  # component width: float64
}


def test_packaged_rates_cover_exactly_the_wire_whitelist():
    load_weights()
    assert weights_module._ACTIVE_DTYPE_RATES == EXPECTED_PACKAGED_RATES


def test_unit_mode_returns_one_for_everything():
    reset_weights()  # conftest default state: unit weights AND unit rates
    assert get_dtype_rate("float64") == 1.0
    assert get_dtype_rate("complex128") == 1.0
    assert get_dtype_rate("float128") == 1.0  # unit mode is permissive


def test_production_mode_fails_closed_on_unknown_dtype():
    load_weights()
    with pytest.raises(UnsupportedDtypeError):
        get_dtype_rate("float128")


def test_disable_env_gives_unit_rates(monkeypatch):
    monkeypatch.setenv("FLOPSCOPE_DISABLE_WEIGHTS", "1")
    load_weights()
    assert get_dtype_rate("float64") == 1.0


def test_custom_weights_file_without_rates_warns_and_uses_unit(tmp_path, monkeypatch):
    p = tmp_path / "w.json"
    p.write_text('{"weights": {"add": 1.0}}')
    monkeypatch.setenv("FLOPSCOPE_WEIGHTS_FILE", str(p))
    with pytest.warns(RuntimeWarning):
        load_weights()
    assert get_dtype_rate("float64") == 1.0
