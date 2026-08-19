"""``dot``/``inner`` must accept a 0-d operand, as numpy does.

numpy treats a 0-d operand as a scalar multiply. flopscope refused the call with
a ZeroDivisionError raised inside its own cost helper.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.mark.parametrize("op_name", ["dot", "inner"])
@pytest.mark.parametrize("scalar_first", [True, False])
def test_zero_dim_operand_is_accepted(op_name, scalar_first):
    """The call must succeed and return numpy's own answer."""
    s_raw = np.array(3.0)
    v_raw = np.ones(8)
    s = fnp.array(s_raw)
    v = fnp.array(v_raw)
    op = getattr(fnp, op_name)
    np_op = getattr(np, op_name)
    with flops.budget(10**15, quiet=True):
        got = np.asarray(op(s, v) if scalar_first else op(v, s))
    want = np_op(s_raw, v_raw) if scalar_first else np_op(v_raw, s_raw)
    assert np.allclose(got, want)


@pytest.mark.parametrize("op_name", ["dot", "inner"])
def test_zero_dim_operand_is_billed(op_name):
    """A scalar multiply over n elements is real work and must be charged."""
    s = fnp.array(np.array(3.0))
    v = fnp.array(np.ones(1000))
    op = getattr(fnp, op_name)
    assert _billed(lambda: op(s, v)) > 0


@pytest.mark.parametrize("op_name", ["dot", "inner"])
def test_zero_dim_cost_scales_with_the_array(op_name):
    """Cost tracks the number of elements actually multiplied."""
    s = fnp.array(np.array(3.0))
    v1 = fnp.array(np.ones(1000))
    v2 = fnp.array(np.ones(2000))
    op = getattr(fnp, op_name)
    assert _billed(lambda: op(s, v2)) > _billed(lambda: op(s, v1))


@pytest.mark.parametrize("op_name", ["dot", "inner"])
def test_both_operands_zero_dim(op_name):
    """Scalar-by-scalar is also legal in numpy."""
    a = fnp.array(np.array(2.0))
    b = fnp.array(np.array(3.0))
    op = getattr(fnp, op_name)
    np_op = getattr(np, op_name)
    with flops.budget(10**15, quiet=True):
        got = np.asarray(op(a, b))
    assert np.allclose(got, np_op(np.array(2.0), np.array(3.0)))


@pytest.mark.parametrize("op_name", ["dot", "inner"])
def test_normal_operands_are_unchanged(op_name):
    """The 0-d handling must not disturb the ordinary paths."""
    a = fnp.array(np.ones(64))
    b = fnp.array(np.ones(64))
    op = getattr(fnp, op_name)
    assert _billed(lambda: op(a, b)) > 0
