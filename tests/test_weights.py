"""Tests for FLOP weight loading and default import-time autoload behavior."""

import importlib
import json
from importlib import resources

import pytest

import flopscope._weights as weights_module
from flopscope._weights import get_weight, load_weights, reset_weights


@pytest.fixture(autouse=True)
def _reset_weights():
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()
    yield
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()


def _write_weights(tmp_path, weights):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"weights": weights}), encoding="utf-8")
    return str(path)


def _packaged_weight(op_name):
    resource = resources.files("flopscope").joinpath("data/default_weights.json")
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)["weights"][op_name]


def test_get_weight_default_is_one():
    assert get_weight("exp") == 1.0
    assert get_weight("nonexistent_op") == 1.0


def test_load_weights_from_file(tmp_path):
    path = _write_weights(tmp_path, {"exp": 18.3, "log": 12.7, "add": 1.0})
    load_weights(path)
    assert get_weight("exp") == 18.3
    assert get_weight("log") == 12.7
    assert get_weight("add") == 1.0
    assert get_weight("sin") == 1.0


def test_load_weights_from_env_var(monkeypatch, tmp_path):
    path = _write_weights(tmp_path, {"sqrt": 3.8})
    monkeypatch.setenv("FLOPSCOPE_WEIGHTS_FILE", path)
    load_weights()
    assert get_weight("sqrt") == 3.8


def test_load_weights_use_packaged_default_explicitly():
    load_weights(use_packaged_default=True)
    assert get_weight("exp") == _packaged_weight("exp")
    assert get_weight("linalg.cholesky") == _packaged_weight("linalg.cholesky")


def test_packaged_array_ops_have_zero_weight():
    load_weights(use_packaged_default=True)
    assert get_weight("empty") == 0.0
    # reshape is weight 1.0 as of Task 4 (billed numel(input)); transpose is
    # the still-free view-op witness now.
    assert get_weight("transpose") == 0.0


def test_load_weights_missing_file_warns_and_falls_back_to_packaged_default_when_enabled():
    with pytest.warns(RuntimeWarning, match="could not load custom weights"):
        load_weights("/nonexistent/path.json", use_packaged_default=True)
    assert get_weight("exp") == _packaged_weight("exp")


def test_load_weights_invalid_json_warns_and_falls_back_to_packaged_default_when_enabled(
    tmp_path,
):
    path = tmp_path / "invalid.json"
    path.write_text("not valid json", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="could not load custom weights"):
        load_weights(str(path), use_packaged_default=True)
    assert get_weight("exp") == _packaged_weight("exp")


def test_load_weights_invalid_override_can_still_leave_unit_weights():
    with pytest.warns(RuntimeWarning, match="falling back to unit weights"):
        load_weights("/nonexistent/path.json", use_packaged_default=False)
    assert get_weight("exp") == 1.0


def test_extract_weights_rejects_non_string_op_names():
    with pytest.raises(ValueError, match="non-string operation name"):
        weights_module._extract_weights({"weights": {1: 2.0}}, source="Test weights")


def test_invalid_weight_values_warn_and_fall_back_to_unit_weights(tmp_path):
    for bad_weights in (
        {"exp": "fast"},
        {"exp": True},
        {"exp": float("inf")},
        {"exp": -1.0},
    ):
        weights_module._WARNED_MESSAGES.clear()
        path = _write_weights(tmp_path, bad_weights)
        with pytest.warns(RuntimeWarning, match="falling back to unit weights"):
            load_weights(path, use_packaged_default=False)
        assert get_weight("exp") == 1.0


def test_invalid_weight_values_warn_and_fall_back_to_packaged_default(tmp_path):
    for bad_weights in (
        {"exp": "fast"},
        {"exp": True},
        {"exp": float("inf")},
        {"exp": -1.0},
    ):
        weights_module._WARNED_MESSAGES.clear()
        path = _write_weights(tmp_path, bad_weights)
        with pytest.warns(
            RuntimeWarning, match="falling back to packaged official weights"
        ):
            load_weights(path, use_packaged_default=True)
        assert get_weight("exp") == _packaged_weight("exp")


def test_disable_weights_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("FLOPSCOPE_DISABLE_WEIGHTS", "1")
    load_weights(use_packaged_default=True)
    assert get_weight("exp") == 1.0


def test_import_time_autoload_uses_packaged_default(monkeypatch):
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    monkeypatch.delenv("FLOPSCOPE_DISABLE_WEIGHTS", raising=False)
    importlib.reload(weights_module)
    assert weights_module.get_weight("exp") == _packaged_weight("exp")
    assert weights_module.get_weight("linalg.cholesky") == _packaged_weight(
        "linalg.cholesky"
    )


def test_import_time_autoload_reads_env_override(monkeypatch, tmp_path):
    path = _write_weights(tmp_path, {"sqrt": 3.8})
    monkeypatch.setenv("FLOPSCOPE_WEIGHTS_FILE", path)
    monkeypatch.delenv("FLOPSCOPE_DISABLE_WEIGHTS", raising=False)
    importlib.reload(weights_module)
    assert weights_module.get_weight("sqrt") == 3.8
    assert weights_module.get_weight("exp") == 1.0


def test_import_time_autoload_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    monkeypatch.setenv("FLOPSCOPE_DISABLE_WEIGHTS", "1")
    importlib.reload(weights_module)
    assert weights_module.get_weight("exp") == 1.0


def test_invalid_override_warning_is_emitted_once():
    with pytest.warns(RuntimeWarning) as record:
        load_weights("/nonexistent/path.json")
        load_weights("/nonexistent/path.json")
    assert len(record) == 1
    assert "falling back to packaged official weights" in str(record[0].message)


def test_reset_weights_clears():
    load_weights(use_packaged_default=True)
    assert get_weight("exp") == _packaged_weight("exp")
    reset_weights()
    assert get_weight("exp") == 1.0
