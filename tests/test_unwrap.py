import numpy

from flopscope._budget import BudgetContext


class TestUnwrap:
    def test_result_matches_numpy(self):
        phase = numpy.array([0.0, 1.0, 2.0, 3.0, -3.0, -2.0])
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy import unwrap

            assert numpy.allclose(unwrap(phase), numpy.unwrap(phase))

    def test_cost(self):
        x = numpy.random.randn(20)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy import unwrap

            unwrap(x)
            assert budget.flops_used == 140  # 7 * numel (issue #69)

    def test_outside_context_uses_global_default(self):
        from flopscope.numpy import unwrap

        # Operations now auto-activate the global default budget instead of raising
        result = unwrap(numpy.ones(5))
        assert result.shape == (5,)
