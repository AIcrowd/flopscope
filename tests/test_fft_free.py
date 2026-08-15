# tests/test_fft_free.py
"""Cost and numerics for the four zero-arithmetic FFT helpers.

Every test here snapshots ``flops_used`` immediately after the metered call and
compares values only after the budget has closed. The ordering is load-bearing,
not stylistic: since #193 these wrappers return a ``FlopscopeArray``, so
``numpy.allclose(result, ...)`` is itself a metered operation. Running the
comparison inside the budgeted region folds its cost into the number under
test, which is what the old form did -- it read ``flops_used`` after the
comparison and so would now measure the op plus the check.

The billed amounts below are unchanged from before the wrap. What moved is
where the comparison happens; see
``test_comparing_a_result_inside_the_budget_is_itself_billed``, which pins the
premise so this file cannot quietly drift back to the old shape.
"""

import numpy

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext


class TestFftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fftfreq

            result = fftfreq(8, d=1.0)
            billed = budget.flops_used

        assert billed == 8  # index grid scaled by 1/(n*d)
        assert numpy.allclose(numpy.asarray(result), numpy.fft.fftfreq(8, d=1.0))


class TestRfftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import rfftfreq

            result = rfftfreq(8, d=1.0)
            billed = budget.flops_used

        assert billed == 8 // 2 + 1
        assert numpy.allclose(numpy.asarray(result), numpy.fft.rfftfreq(8, d=1.0))


class TestFftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fftshift

            result = fftshift(x)
            billed = budget.flops_used

        assert billed == 8  # numel(output); Task 4
        assert numpy.allclose(numpy.asarray(result), numpy.fft.fftshift(x))


class TestIfftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import ifftshift

            result = ifftshift(x)
            billed = budget.flops_used

        assert billed == 8  # numel(output); Task 4
        assert numpy.allclose(numpy.asarray(result), numpy.fft.ifftshift(x))


def test_comparing_a_result_inside_the_budget_is_itself_billed():
    """The reason the four tests above snapshot before they compare.

    ``fnp.allclose`` is spelled out here because that is where a bare
    ``numpy.allclose`` on a flopscope array ends up anyway -- numpy dispatches
    it back into flopscope, with a warning about the auto-route. Either way the
    comparison is a billed op once the fftfreq result is a flopscope type, so a
    ``flops_used`` read taken after it measures the op *plus* the check.

    Asserted as "strictly greater", not as a literal: the point is that the
    comparison costs something, and pinning its exact cost here would make this
    file fail on an unrelated allclose repricing.
    """
    with BudgetContext(flop_budget=10**6) as budget:
        from flopscope.numpy.fft import fftfreq

        result = fftfreq(8, d=1.0)
        after_call = budget.flops_used
        assert fnp.allclose(result, fnp.asarray(numpy.fft.fftfreq(8, d=1.0)))
        after_compare = budget.flops_used

    assert after_call == 8, (
        f"fft.fftfreq(8) billed {after_call}, not 8 -- this file's numbers are "
        "unchanged by the #193 wrap, so a move here is a repricing"
    )
    assert after_compare > after_call, (
        "comparing an fft.fftfreq result against numpy no longer bills "
        "anything, so the snapshot-then-compare shape above is no longer "
        "needed -- but check first that the result is still a flopscope type"
    )
