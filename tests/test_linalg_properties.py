"""Tests for linalg property wrappers with FLOP counting."""

import numpy

from flopscope._budget import BudgetContext


class TestTrace:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import trace

            assert trace(A) == numpy.trace(A)

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import trace

            trace(A)
            assert budget.flops_used == n


class TestDet:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import det

            assert numpy.isclose(det(A), numpy.linalg.det(A))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import det

            det(A)
            assert budget.flops_used == n**3


class TestSlogdet:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import slogdet

            sign, logdet = slogdet(A)
            sign_np, logdet_np = numpy.linalg.slogdet(A)
            assert numpy.isclose(sign, sign_np)
            assert numpy.isclose(logdet, logdet_np)

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import slogdet

            slogdet(A)
            assert budget.flops_used == n**3


class TestNorm:
    def test_vector_default(self):
        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import norm

            assert numpy.isclose(norm(x), 5.0)

    def test_vector_default_cost(self):
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import norm

            norm(x)
            assert budget.flops_used == 20  # FMA=2: 2*numel

    def test_matrix_fro_cost(self):
        # FMA=2: Frobenius norm costs 2*numel
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import norm

            norm(A)
            assert budget.flops_used == 40  # FMA=2: 2*numel

    def test_matrix_ord2_cost(self):
        # SVD-based: 4x baked into cost function
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import norm

            norm(A, ord=2)
            assert budget.flops_used == 4 * 4 * 5 * 4

    def test_matrix_ord1_cost(self):
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import norm

            norm(A, ord=1)
            assert budget.flops_used == 40  # FMA=2: 2*numel

    def test_vector_p_norm_cost(self):
        # FMA=2: p-norm costs 2*numel
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import norm

            norm(x, ord=3)
            assert budget.flops_used == 20  # FMA=2: 2*numel


class TestVectorNorm:
    def test_result_matches_numpy(self):
        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import vector_norm

            assert numpy.isclose(vector_norm(x), 5.0)

    def test_cost(self):
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import vector_norm

            vector_norm(x)
            assert budget.flops_used == 20  # FMA=2: 2*numel


class TestMatrixNorm:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(3, 4)
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import matrix_norm

            assert numpy.isclose(matrix_norm(A), numpy.linalg.matrix_norm(A))

    def test_fro_cost(self):
        # FMA=2: Frobenius norm costs 2*numel
        A = numpy.random.randn(3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import matrix_norm

            matrix_norm(A)
            assert budget.flops_used == 24  # FMA=2: 2*numel


class TestCond:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 0.0], [0.0, 2.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import cond

            assert numpy.isclose(cond(A), numpy.linalg.cond(A))

    def test_cost(self):
        m, n = 4, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import cond

            cond(A)
            assert budget.flops_used == m * n * min(m, n)


class TestMatrixRank:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 0.0], [0.0, 0.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import matrix_rank

            assert matrix_rank(A) == numpy.linalg.matrix_rank(A)

    def test_cost(self):
        m, n = 5, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.linalg import matrix_rank

            matrix_rank(A)
            assert budget.flops_used == m * n * min(m, n)
