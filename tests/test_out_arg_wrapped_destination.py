"""A wrapped ``out=`` destination must be refused, not silently mis-written.

``out=[dest]`` — the destination wrapped in a container — used to be accepted by
the einsum path: ``_np.asarray([dest])`` builds a NEW array from the container,
so the copy landed in that temporary, ``dest`` kept its old contents, and the
caller received the untouched wrapper back having been billed the full
contraction. A plausible-looking array of the wrong values, at full price, is
the worst failure class in a metering system.

numpy rejects a wrapped destination for ufuncs on its own, so this only ever bit
the paths that copy into ``out`` themselves. The guard sits before billing, so a
refused call costs nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as fl
import flopscope.numpy as fnp


@pytest.fixture()
def budget():
    with fl.BudgetContext(flop_budget=10**12, quiet=True) as ctx:
        yield ctx


def test_einsum_wrapped_out_is_refused_and_costs_nothing(budget):
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = fnp.array([[5.0, 6.0], [7.0, 8.0]])
    dest = fnp.array([[0.0, 0.0], [0.0, 0.0]])
    before = budget.flops_used

    with pytest.raises(TypeError, match="out= must be an array"):
        fnp.einsum("ij,jk->ik", a, b, out=[dest])

    assert budget.flops_used == before, "a refused call must not be billed"
    assert np.array_equal(np.asarray(dest), np.zeros((2, 2)))


def test_einsum_with_a_real_out_still_works(budget):
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = fnp.array([[5.0, 6.0], [7.0, 8.0]])
    dest = fnp.array([[0.0, 0.0], [0.0, 0.0]])

    result = fnp.einsum("ij,jk->ik", a, b, out=dest)

    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert np.asarray(result).tolist() == expected
    assert np.asarray(dest).tolist() == expected, "out= must receive the result"
    assert result is dest, "out= must return the destination itself"


@pytest.mark.parametrize("wrapper", [list, tuple])
def test_pointwise_wrapped_out_is_refused(budget, wrapper):
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    dest = fnp.array([0.0, 0.0])

    with pytest.raises(TypeError):
        fnp.multiply(a, b, out=wrapper([dest]))

    assert np.array_equal(np.asarray(dest), np.zeros(2))


def test_out_still_works_on_the_routed_binaries(budget):
    """vecdot/matvec/vecmat expose out= publicly; they must keep working."""
    eye = fnp.array([[1.0, 0.0], [0.0, 1.0]])
    v = fnp.array([3.0, 4.0])

    dest = fnp.array([0.0, 0.0])
    assert np.asarray(fnp.matvec(eye, v, out=dest)).tolist() == [3.0, 4.0]

    scalar_dest = fnp.array(0.0)
    assert float(np.asarray(fnp.vecdot(v, v, out=scalar_dest))) == 25.0


def test_out_none_and_absent_are_unaffected(budget):
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    assert np.asarray(fnp.multiply(a, b)).tolist() == [3.0, 8.0]
    assert np.asarray(fnp.multiply(a, b, out=None)).tolist() == [3.0, 8.0]
