"""Quantile dimensions prepend, rather than replace, surviving symmetry axes."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


@pytest.mark.parametrize(
    "name", ["quantile", "percentile", "nanquantile", "nanpercentile"]
)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("q", [0.5, [0.25, 0.75]])
def test_quantile_shifts_surviving_group(name, keepdims, q):
    # Unequal q/batch extents expose incorrect tag placement immediately.
    values = np.random.default_rng(2).normal(size=(4, 3, 3))
    values = values + values.swapaxes(1, 2)
    percent = "percentile" in name
    request = np.asarray(q) * (100 if percent else 1)
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(values), symmetry=(1, 2))
        result = getattr(fnp, name)(tagged, request, axis=0, keepdims=keepdims)
        expected = getattr(np, name)(values, request, axis=0, keepdims=keepdims)
        np.testing.assert_allclose(np.asarray(result), expected)
        offset = int(np.ndim(q) > 0) + int(keepdims)
        assert result.symmetry.axes == (offset, offset + 1)
        # A subsequent pointwise op validates the actual tagged axes.
        np.testing.assert_allclose(np.asarray(fnp.exp(result)), np.exp(expected))


@pytest.mark.parametrize(
    "name", ["quantile", "percentile", "nanquantile", "nanpercentile"]
)
def test_vector_q_out_and_scalar_reduction(name):
    values = np.arange(27.0).reshape(3, 3, 3)
    values = values + values.swapaxes(1, 2)
    request = [25, 75] if "percentile" in name else [0.25, 0.75]
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(values), symmetry=(1, 2))
        out = fnp.zeros((2, 3, 3))
        result = getattr(fnp, name)(tagged, request, axis=0, out=out)
        assert result is out
        assert result.symmetry.axes == (1, 2)
        flat = getattr(fnp, name)(tagged, request, axis=None)
        assert getattr(flat, "symmetry", None) is None


@pytest.mark.parametrize("axis,shape", [(-1, (3, 3, 4)), ((2, 3), (3, 3, 4, 2))])
def test_quantile_negative_and_tuple_reduction_axes(axis, shape):
    values = np.random.default_rng(8).normal(size=shape)
    values = values + values.swapaxes(0, 1)
    q = np.array([[0.25, 0.5], [0.75, 1.0]])
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(values), symmetry=(0, 1))
        result = fnp.quantile(tagged, q, axis=axis)
        np.testing.assert_allclose(
            np.asarray(result), np.quantile(values, q, axis=axis)
        )
        assert result.symmetry.axes == (2, 3)
        fnp.exp(result)
