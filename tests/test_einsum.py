"""Tests for flopscope einsum with FLOP counting and symmetry."""

import numpy
import pytest

from flopscope import SymmetryGroup
from flopscope._budget import BudgetContext
from flopscope._einsum import einsum
from flopscope.errors import BudgetExhaustedError, SymmetryError


def test_matmul_result():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        C = einsum("ij,jk->ik", A, B)
        assert numpy.allclose(C, numpy.einsum("ij,jk->ik", A, B))


def test_matmul_flop_cost():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("ij,jk->ik", A, B)
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 60 + 60 - 15 = 105 (textbook 2n^3 - n^2 form).
        # First cell of each output orbit is a free copy.
        assert budget.flops_used == 105


def test_trace():
    A = numpy.eye(10)
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("ii->", A)
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 0 + 10 - 1 = 9 (scalar output: first cell is a free copy).
        assert budget.flops_used == 9
    assert float(result) == 10.0


def test_outer_product():
    a = numpy.ones((3,))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("i,j->ij", a, b)
        assert result.shape == (3, 4)
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits) = 12.
        # Outer product has no summed axes; alpha == num_output_orbits so the
        # accumulation term collapses, leaving only the multiplications.
        assert budget.flops_used == 12


def test_batch_matmul():
    A = numpy.ones((2, 3, 4))
    B = numpy.ones((2, 4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("bij,bjk->bik", A, B)
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 120 + 120 - 30 = 210. First cell of each output orbit is a free copy.
        assert budget.flops_used == 210


def test_symmetry_valid():
    x = numpy.ones((3, 10))
    y = numpy.ones((3, 10))
    A = numpy.eye(3)
    with BudgetContext(flop_budget=10**8) as budget:
        result = einsum(
            "ai,bj,ab->ij",
            x,
            y,
            A,
            symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
        )
        assert budget.flops_used > 0  # cost comes from opt_einsum now


def test_symmetry_invalid():
    x = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    y = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    with BudgetContext(flop_budget=10**8):
        with pytest.raises(SymmetryError):
            einsum(
                "ij,jk->ik",
                x,
                y,
                symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
            )


def test_outside_context():
    # Operations now auto-activate the global default budget instead of raising
    result = einsum("ij,jk->ik", numpy.ones((3, 4)), numpy.ones((4, 5)))
    assert result.shape == (3, 5)


def test_budget_exceeded():
    A = numpy.ones((256, 256))
    with pytest.raises(BudgetExhaustedError):
        with BudgetContext(flop_budget=100):
            einsum("ij,jk->ik", A, A)


def test_op_log_records_subscripts():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("ij,jk->ik", A, B)
        assert budget.op_log[0].subscripts == "ij,jk->ik"
        assert budget.op_log[0].op_name == "einsum"
