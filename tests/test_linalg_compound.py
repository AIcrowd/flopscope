"""Tests for linalg compound operation wrappers with FLOP counting."""

import numpy

from flopscope._budget import BudgetContext


class TestMultiDot:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(10, 5)
        B = numpy.random.randn(5, 8)
        C = numpy.random.randn(8, 3)
        with BudgetContext(flop_budget=10**8):
            from flopscope.numpy.linalg import multi_dot

            result = multi_dot([A, B, C])
            expected = numpy.linalg.multi_dot([A, B, C])
            assert numpy.allclose(result, expected)

    def test_cost_positive(self):
        A = numpy.random.randn(10, 5)
        B = numpy.random.randn(5, 8)
        C = numpy.random.randn(8, 3)
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import multi_dot

            multi_dot([A, B, C])
            assert budget.flops_used > 0

    def test_op_log(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import multi_dot

            multi_dot([A, B])
            assert budget.op_log[-1].op_name == "linalg.multi_dot"


class TestMatrixPower:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.linalg import matrix_power

            result = matrix_power(A, 3)
            expected = numpy.linalg.matrix_power(A, 3)
            assert numpy.allclose(result, expected)

    def test_cost_power_3(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import matrix_power

            matrix_power(A, 3)
            # k=3: floor(log2(3))=1, popcount(3)=2, num_ops=(1+2-1)=2 matmuls
            # matmul_cost(4,4,4) = 2*64 - 16 = 112; total = 2*112 = 224
            assert budget.flops_used == 2 * (2 * n**3 - n**2)

    def test_cost_power_0(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import matrix_power

            matrix_power(A, 0)
            assert budget.flops_used == 0

    def test_cost_power_1(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import matrix_power

            matrix_power(A, 1)
            assert budget.flops_used == 0

    def test_cost_negative_power(self):
        n = 3
        A = numpy.random.randn(n, n) + numpy.eye(n) * 5
        with BudgetContext(flop_budget=10**8) as budget:
            from flopscope.numpy.linalg import matrix_power

            matrix_power(A, -2)
            # inv: n^3 + power k=2: num_ops=(1+1-1)=1 matmul
            # matmul_cost(3,3,3) = 2*27 - 9 = 45; total = 27 + 45 = 72
            assert budget.flops_used == n**3 + (2 * n**3 - n**2)

    def test_outside_context_uses_global_default(self):
        from flopscope.numpy.linalg import matrix_power

        # Operations now auto-activate the global default budget instead of raising
        result = matrix_power(numpy.eye(3), 2)
        assert result.shape == (3, 3)
