"""``fnp.astype`` must work on every supported numpy, including 2.0.

``np.astype`` only grew its ``device=`` keyword in numpy 2.1, while flopscope
supports ``numpy>=2.0.0,<2.5.0`` (pyproject / ``_registry``). Forwarding
``device=`` unconditionally made every ``fnp.astype`` call raise ``TypeError``
under a real numpy 2.0 -- while flopscope still reported itself as
``0.9.1+np2.0.x``. These tests pin the contract that behavior AND billing are
identical across the supported range: the numpy>=2.1 device semantics
(``None``/``"cpu"`` accepted; anything else ``ValueError`` -- raised after the
charge, since ``deduct`` bills on entry) hold even where numpy itself cannot
accept the keyword.

The ``_np20_astype`` shim replays numpy 2.0.2's exact ``np.astype`` (signature
and body) so the 2.0 code path is exercised under every numpy in the CI
matrix, not only in the 2.0 cells.
"""

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope import _array_ops


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        result = fn()
        return result, b.flops_used


def _np20_astype(x, dtype, /, *, copy=True):
    """``np.astype`` exactly as numpy 2.0.x defines it: no ``device=``."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input should be a NumPy array. It is a {type(x)} instead.")
    return x.astype(dtype, copy=copy)


@pytest.fixture
def numpy20_astype(monkeypatch):
    """Make the running numpy look like 2.0 to the astype wrapper."""
    monkeypatch.setattr(np, "astype", _np20_astype)
    # raising=False: on unfixed code the flag does not exist yet; the shimmed
    # np.astype alone is then what makes the wrapper fail (the RED state).
    monkeypatch.setattr(_array_ops, "_NP_ASTYPE_HAS_DEVICE", False, raising=False)


@pytest.mark.usefixtures("numpy20_astype")
def test_astype_works_when_numpy_lacks_device_kwarg():
    result, cost = _billed(lambda: fnp.astype(np.ones(4, np.float32), np.float64))
    assert np.asarray(result).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(result), np.ones(4))
    assert cost > 0  # exact formula pinned by test_data_movement_free_tier


@pytest.mark.usefixtures("numpy20_astype")
def test_astype_copy_false_noop_stays_free_without_device_kwarg():
    x = np.ones(4, np.float32)
    _, cost = _billed(lambda: fnp.astype(x, np.float32, copy=False))
    assert cost == 0  # the one free case must survive the 2.0 path


def test_astype_bills_identically_with_and_without_device_kwarg():
    x = np.arange(100, dtype=np.float32)
    _, native = _billed(lambda: fnp.astype(x, np.float64))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, "astype", _np20_astype)
        mp.setattr(_array_ops, "_NP_ASTYPE_HAS_DEVICE", False, raising=False)
        _, shimmed = _billed(lambda: fnp.astype(x, np.float64))
    assert shimmed == native


@pytest.mark.usefixtures("numpy20_astype")
def test_astype_device_cpu_accepted_on_numpy20_path():
    result, _ = _billed(
        lambda: fnp.astype(np.ones(4, np.float32), np.float64, device="cpu")
    )
    assert np.asarray(result).dtype == np.float64


def test_astype_device_cpu_accepted_native():
    """Whatever numpy is installed (2.0 included), device="cpu" must work."""
    result, _ = _billed(
        lambda: fnp.astype(np.ones(4, np.float32), np.float64, device="cpu")
    )
    assert np.asarray(result).dtype == np.float64


@pytest.mark.usefixtures("numpy20_astype")
def test_astype_bad_device_raises_valueerror_on_numpy20_path():
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        with pytest.raises(ValueError, match='Only "cpu" is allowed'):
            fnp.astype(np.ones(4, np.float32), np.float64, device="cuda")


def test_astype_bad_device_raises_valueerror_native():
    """numpy>=2.1 raises ValueError itself; the 2.0 path must match, so this
    holds under every installed numpy in the matrix."""
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        with pytest.raises(ValueError, match='Only "cpu" is allowed'):
            fnp.astype(np.ones(4, np.float32), np.float64, device="cuda")


def test_astype_bad_device_bills_identically_with_and_without_device_kwarg():
    """deduct charges on entry, so numpy>=2.1 bills then raises; 2.0 path too."""
    x = np.arange(50, dtype=np.float32)

    def billed_failure():
        with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
            with pytest.raises(ValueError):
                fnp.astype(x, np.float64, device="mps")
            return b.flops_used

    native = billed_failure()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, "astype", _np20_astype)
        mp.setattr(_array_ops, "_NP_ASTYPE_HAS_DEVICE", False, raising=False)
        shimmed = billed_failure()
    assert shimmed == native
