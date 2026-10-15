"""Singleton transforms still move their complex working buffers."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights


@pytest.mark.parametrize("name", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_singleton_batch_matches_numpy_and_pays_movement(name, dtype):
    x = np.arange(12, dtype=dtype).reshape(12, 1)
    with flops.BudgetContext(flop_budget=10**7) as budget:
        result = getattr(fnp.fft, name)(x, n=1)
    expected = getattr(np.fft, name)(x, n=1)
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert result.dtype == expected.dtype
    assert budget.flops_used == 24


@pytest.mark.parametrize("name", ["fftn", "ifftn", "rfftn", "irfftn"])
def test_nd_singleton_stages_equal_explicit_calls(name):
    x = np.arange(8, dtype=np.float64).reshape(8, 1, 1)
    with flops.BudgetContext(flop_budget=10**7) as budget:
        result = getattr(fnp.fft, name)(x, s=(1, 1), axes=(1, 2))
    np.testing.assert_array_equal(
        np.asarray(result), getattr(np.fft, name)(x, s=(1, 1), axes=(1, 2))
    )
    assert budget.flops_used == 2 * 8 * 2


@pytest.mark.parametrize(
    "dtype,expected",
    [(np.float32, 24), (np.float64, 48), (np.complex64, 24), (np.complex128, 48)],
)
def test_production_precision_rates(dtype, expected):
    load_weights()
    x = np.arange(12, dtype=dtype).reshape(12, 1)
    with flops.BudgetContext(flop_budget=10**7) as budget:
        fnp.fft.fft(x)
    assert budget.flops_used == expected


def test_mixed_axes_match_staged_execution():
    x = np.arange(4, dtype=np.float32).reshape(4, 1)
    with flops.BudgetContext(flop_budget=10**7) as fused:
        result = fnp.fft.fftn(x)
    with flops.BudgetContext(flop_budget=10**7) as staged:
        explicit = fnp.fft.fft(fnp.fft.fft(x, axis=1), axis=0)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(explicit))
    assert fused.flops_used == staged.flops_used == 48


def test_wider_out_retains_destination_rate():
    load_weights()
    x = np.arange(12, dtype=np.float32).reshape(12, 1)
    dest = fnp.empty((12, 1), dtype=np.complex128)
    with flops.BudgetContext(flop_budget=10**7) as budget:
        result = fnp.fft.fft(x, out=dest)
    assert result is dest
    assert budget.flops_used == 48
    np.testing.assert_array_equal(np.asarray(result), np.fft.fft(x))


def test_empty_batches_do_not_materialize_values():
    x = np.empty((0, 1))
    with flops.BudgetContext(flop_budget=10**7) as budget:
        result = fnp.fft.fft(x)
    assert result.shape == (0, 1)
    assert budget.flops_used == 0


@pytest.mark.parametrize("shape", [(1, 8), (8, 1), (1, 1, 8)])
@pytest.mark.parametrize("real", [False, True])
def test_mixed_singleton_axes_match_explicit_cascade(shape, real):
    x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    with flops.BudgetContext(flop_budget=10**7) as fused:
        result = (fnp.fft.rfftn if real else fnp.fft.fftn)(x)
    with flops.BudgetContext(flop_budget=10**7) as staged:
        explicit = x
        for axis in reversed(range(len(shape))):
            transform = fnp.fft.rfft if real and axis == len(shape) - 1 else fnp.fft.fft
            explicit = transform(explicit, axis=axis)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(explicit))
    assert fused.flops_used == staged.flops_used


def test_invalid_destination_does_not_charge():
    x = np.ones((8, 1))
    dest = np.empty((8, 2), dtype=np.complex128)
    with flops.BudgetContext(flop_budget=10**7) as budget:
        with pytest.raises(ValueError):
            fnp.fft.fft(x, out=dest)
    assert budget.flops_used == 0
