"""A Python scalar must not widen the billing dtype (NEP 50 weak promotion).

cost-model.md commits to NEP 50: ``f32_array * 2.0`` stays float32. The billing
dtype for ``where``/``insert`` must therefore match the numpy-scalar spelling of
the same call, not a 64-bit promotion of it.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights


def _billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.mark.parametrize(
    "dtype, py_scalar, np_scalar",
    [
        (np.float32, 0.0, np.float32(0.0)),
        (np.float32, 0, np.float32(0)),
        (np.float16, 0.0, np.float16(0.0)),
        (np.int32, 0, np.int32(0)),
        (np.complex64, 0j, np.complex64(0j)),
    ],
)
def test_where_python_scalar_bills_like_the_numpy_scalar(dtype, py_scalar, np_scalar):
    """A weak Python scalar must bill exactly like its same-dtype numpy scalar."""
    # tests/conftest.py's autouse fixture resets to unit dtype rates (all
    # 1.0) before every test, which would make this assertion pass trivially
    # regardless of correctness -- load production rates so a wrong
    # resolution actually shows up as a billing difference.
    load_weights()
    mask = fnp.array(np.ones(1000, dtype=bool))
    arr = fnp.array(np.ones(1000, dtype=dtype))
    assert _billed(lambda: fnp.where(mask, arr, py_scalar)) == _billed(
        lambda: fnp.where(mask, arr, np_scalar)
    )


@pytest.mark.parametrize(
    "dtype, py_scalar, np_scalar",
    [
        (np.float32, 0.0, np.float32(0.0)),
        (np.int32, 0, np.int32(0)),
        (np.complex64, 0j, np.complex64(0j)),
    ],
)
def test_insert_python_scalar_bills_like_the_numpy_scalar(dtype, py_scalar, np_scalar):
    """Same rule for ``insert``'s ``values`` argument."""
    load_weights()
    arr = fnp.array(np.ones(1000, dtype=dtype))
    assert _billed(lambda: fnp.insert(arr, 0, py_scalar)) == _billed(
        lambda: fnp.insert(arr, 0, np_scalar)
    )


def test_where_array_operands_are_unchanged():
    """Array-vs-array billing must not move: only the scalar case changes."""
    load_weights()
    mask = fnp.array(np.ones(1000, dtype=bool))
    a32 = fnp.array(np.ones(1000, dtype=np.float32))
    a64 = fnp.array(np.ones(1000, dtype=np.float64))
    # A genuine float64 operand still widens the bill above the all-float32 form.
    assert _billed(lambda: fnp.where(mask, a32, a64)) > _billed(
        lambda: fnp.where(mask, a32, a32)
    )


def test_where_python_scalar_does_not_widen_a_narrow_array():
    """The defect: a Python float made a float32 array bill at the float64 rate."""
    load_weights()
    mask = fnp.array(np.ones(1000, dtype=bool))
    a32 = fnp.array(np.ones(1000, dtype=np.float32))
    a64 = fnp.array(np.ones(1000, dtype=np.float64))
    assert _billed(lambda: fnp.where(mask, a32, 0.0)) < _billed(
        lambda: fnp.where(mask, a64, 0.0)
    )
