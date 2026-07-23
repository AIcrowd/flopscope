# tests/test_fft_free.py
import numpy

from flopscope._budget import BudgetContext


class TestFftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fftfreq

            assert numpy.allclose(fftfreq(8, d=1.0), numpy.fft.fftfreq(8, d=1.0))
            assert budget.flops_used == 8  # index grid scaled by 1/(n*d)


class TestRfftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import rfftfreq

            assert numpy.allclose(rfftfreq(8, d=1.0), numpy.fft.rfftfreq(8, d=1.0))
            assert budget.flops_used == 8 // 2 + 1


class TestFftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fftshift

            assert numpy.allclose(fftshift(x), numpy.fft.fftshift(x))
            assert budget.flops_used == 8  # numel(output); Task 4


class TestIfftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import ifftshift

            assert numpy.allclose(ifftshift(x), numpy.fft.ifftshift(x))
            assert budget.flops_used == 8  # numel(output); Task 4
