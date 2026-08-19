"""``dot``/``inner``/two-array ``linalg.multi_dot`` must accept a 0-d operand.

numpy treats a 0-d operand as a scalar multiply. ``dot``/``inner`` refused the
call with a ZeroDivisionError raised inside flopscope's own cost helper.
``linalg.multi_dot`` shared the same class of bug for exactly two arrays (an
IndexError from its own chain-cost helper) -- numpy's own two-array
``multi_dot`` delegates straight to ``dot``, so the same 0-d-is-a-scalar-
multiply rule applies there. A 0-d operand inside a three-or-more-array
``multi_dot`` chain is different: real numpy itself refuses that case
(``LinAlgError``), so flopscope refusing it too is correct and untouched --
only the exact exception type differs, which this library does not guarantee.
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


# --- linalg.multi_dot: the same 0-d carve-out, but two-array calls only ----


@pytest.mark.parametrize("scalar_first", [True, False])
def test_multi_dot_zero_dim_operand_is_accepted(scalar_first):
    """A two-array call must succeed and return numpy's own answer."""
    s_raw = np.array(3.0)
    v_raw = np.ones(8)
    s = fnp.array(s_raw)
    v = fnp.array(v_raw)
    with flops.budget(10**15, quiet=True):
        args = [s, v] if scalar_first else [v, s]
        got = np.asarray(fnp.linalg.multi_dot(args))
    want_args = [s_raw, v_raw] if scalar_first else [v_raw, s_raw]
    want = np.linalg.multi_dot(want_args)
    assert np.allclose(got, want)


def test_multi_dot_zero_dim_operand_is_billed():
    """A scalar multiply over n elements is real work and must be charged."""
    s = fnp.array(np.array(3.0))
    v = fnp.array(np.ones(1000))
    assert _billed(lambda: fnp.linalg.multi_dot([s, v])) > 0


def test_multi_dot_zero_dim_cost_scales_with_the_array():
    """Cost tracks the number of elements actually multiplied."""
    s = fnp.array(np.array(3.0))
    v1 = fnp.array(np.ones(1000))
    v2 = fnp.array(np.ones(2000))
    assert _billed(lambda: fnp.linalg.multi_dot([s, v2])) > _billed(
        lambda: fnp.linalg.multi_dot([s, v1])
    )


def test_multi_dot_both_operands_zero_dim():
    """Scalar-by-scalar is also legal in numpy's two-array multi_dot."""
    a = fnp.array(np.array(2.0))
    b = fnp.array(np.array(3.0))
    with flops.budget(10**15, quiet=True):
        got = np.asarray(fnp.linalg.multi_dot([a, b]))
    assert np.allclose(got, np.linalg.multi_dot([np.array(2.0), np.array(3.0)]))


def test_multi_dot_normal_operands_are_unchanged():
    """The 0-d handling must not disturb the ordinary chain-cost path."""
    a = fnp.array(np.ones((4, 4)))
    b = fnp.array(np.ones((4, 4)))
    assert _billed(lambda: fnp.linalg.multi_dot([a, b])) > 0


def test_multi_dot_three_or_more_arrays_with_zero_dim_still_raises():
    """numpy itself refuses 3+ arrays with a 0-d operand -- this stays refused.

    ``np.linalg.multi_dot`` raises ``LinAlgError`` for a chain of three or
    more arrays containing a 0-d operand (ground truth asserted directly
    below, against real numpy). Exception-*type* parity is not guaranteed by
    this library, but the refusal itself must hold: this pins that the
    two-array 0-d fix does not silently overreach into accepting -- and
    billing -- a call numpy itself rejects.
    """
    mat = np.ones((3, 3))
    scalar = np.array(2.0)
    with pytest.raises(np.linalg.LinAlgError):  # ground truth: real numpy
        np.linalg.multi_dot([mat, scalar, mat])

    fmat = fnp.array(mat)
    fscalar = fnp.array(scalar)
    with flops.budget(10**15, quiet=True) as b:
        before = b.flops_used
        # Not a claim about which exception type flopscope raises here --
        # only that the call is refused (unchanged from before this fix) and
        # refused for free. See the module docstring: exception-type parity
        # for this case is explicitly not guaranteed.
        with pytest.raises(IndexError):
            fnp.linalg.multi_dot([fmat, fscalar, fmat])
        assert b.flops_used == before  # refused before any billing
