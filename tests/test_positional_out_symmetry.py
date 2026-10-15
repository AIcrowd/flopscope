"""Positional destinations invalidate every tag observing their buffer (B2)."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import _call_numpy, _positional_out_index
from flopscope._write_epoch import epoch_of


@pytest.mark.parametrize("ufunc", [np.add, np.exp, np.modf])
def test_ufunc_destinations_remain_under_staged_tracking(ufunc):
    assert _positional_out_index(ufunc) is None


def test_choose_positional_destination_matches_keyword():
    raw = np.arange(16.0).reshape(4, 4)
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        positional = fnp.zeros((4, 4))
        keyword = fnp.zeros((4, 4))
        fnp.choose(np.zeros((4, 4), dtype=int), [raw], positional)
        fnp.choose(np.zeros((4, 4), dtype=int), [raw], out=keyword)
        assert getattr(positional, "symmetry", None) is None
        assert getattr(keyword, "symmetry", None) is None
        np.testing.assert_array_equal(np.asarray(positional), raw)
        np.testing.assert_array_equal(np.asarray(keyword), raw)


@pytest.mark.parametrize(
    "write",
    [
        lambda out: _call_numpy(
            np.choose, np.zeros((4, 4), dtype=int), [np.arange(16.0).reshape(4, 4)], out
        ),
        lambda out: _call_numpy(
            np.take, np.arange(16.0).reshape(4, 4), np.arange(4), 0, out
        ),
        lambda out: _call_numpy(
            np.compress, [True] * 4, np.arange(16.0).reshape(4, 4), 0, out
        ),
        lambda out: _call_numpy(np.stack, [np.arange(4.0)] * 4, 0, out),
        lambda out: _call_numpy(
            np.concatenate, [np.arange(8.0).reshape(2, 4)] * 2, 0, out
        ),
        lambda out: _call_numpy(np.dot, np.arange(16.0).reshape(4, 4), np.eye(4), out),
    ],
)
def test_backend_positional_destination_invalidates_alias(write):
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = fnp.zeros((4, 4))
        alias = np.asarray(tagged)
        before = epoch_of(tagged)
        write(alias)
        assert epoch_of(tagged) > before
        assert getattr(tagged, "symmetry", None) is None


def test_choose_cannot_discount_general_matmul_with_five_axis_tag():
    rng = np.random.default_rng(12)
    left = rng.normal(size=(3, 32))
    right = rng.normal(size=(32, 7))
    tensor = right.reshape((2,) * 5 + (7,))
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        destination = fnp.zeros(tensor.shape)
        lhs = fnp.zeros((3,) + (2,) * 5)
        fnp.choose(np.zeros(lhs.shape, dtype=int), [left.reshape(lhs.shape)], lhs)
        fnp.choose(np.zeros(tensor.shape, dtype=int), [tensor], destination)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as tagged_bill:
        result = fnp.einsum("zabcde,abcdef->zf", lhs, destination)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as dense_bill:
        fnp.einsum("zabcde,abcdef->zf", left.reshape((3,) + (2,) * 5), tensor)
    np.testing.assert_allclose(np.asarray(result), left @ right)
    assert tagged_bill.flops_used == dense_bill.flops_used


@pytest.mark.parametrize("positional", [False, True])
def test_failed_backend_write_invalidates_equally(positional):
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        dest = fnp.zeros((4, 4))
        args = (np.zeros((4, 4), dtype=int), [np.ones((2, 2))])
        with pytest.raises(ValueError):
            if positional:
                _call_numpy(np.choose, *args, np.asarray(dest))
            else:
                _call_numpy(np.choose, *args, out=np.asarray(dest))
        assert getattr(dest, "symmetry", None) is None
