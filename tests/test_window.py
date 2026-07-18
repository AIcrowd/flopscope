import numpy

from flopscope._budget import BudgetContext


class TestBartlett:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import bartlett

            assert numpy.allclose(bartlett(10), numpy.bartlett(10))

    def test_cost(self):
        # Updated: compare+div+add+select per sample (FMA=2); 4 ops/point
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import bartlett

            bartlett(10)
            assert budget.flops_used == 4 * 10


class TestBlackman:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import blackman

            assert numpy.allclose(blackman(10), numpy.blackman(10))

    def test_cost(self):
        # Updated: 2 cos evals @16 + 8 arith per sample; 40 ops/point
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import blackman

            blackman(10)
            assert budget.flops_used == 40 * 10


class TestHamming:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import hamming

            assert numpy.allclose(hamming(10), numpy.hamming(10))

    def test_cost(self):
        # Updated (cost-model triage Task 10): derived-constant convention,
        # matching kaiser -- cos@16 + mul + sub per sample = 18 ops/point.
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import hamming

            hamming(10)
            assert budget.flops_used == 180


class TestHanning:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import hanning

            assert numpy.allclose(hanning(10), numpy.hanning(10))

    def test_cost(self):
        # Updated (cost-model triage Task 10): derived-constant convention,
        # matching kaiser -- cos@16 + mul + sub per sample = 18 ops/point.
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import hanning

            hanning(10)
            assert budget.flops_used == 180


class TestKaiser:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import kaiser

            assert numpy.allclose(kaiser(10, 5.0), numpy.kaiser(10, 5.0))

    def test_cost(self):
        # Updated: Bessel I0 @16 + 7 arith per sample; 23 ops/point
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import kaiser

            kaiser(10, 5.0)
            assert budget.flops_used == 23 * 10
