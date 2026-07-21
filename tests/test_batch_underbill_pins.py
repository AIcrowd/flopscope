import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _bill(thunk):
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        thunk()
    return bc.flops_used


def test_solve_bills_broadcast_batch_from_b():
    rng = np.random.default_rng(0)
    C = fnp.asarray(rng.standard_normal((64, 64)))
    B1 = fnp.asarray(rng.standard_normal((64, 32)))
    B8 = fnp.asarray(rng.standard_normal((8, 64, 32)))
    b1 = _bill(lambda: fnp.linalg.solve(C, B1))
    b8 = _bill(lambda: fnp.linalg.solve(C, B8))
    assert b8 == 8 * b1  # batch of 8 independent solves


def test_tensorsolve_degenerate_scales_with_n():
    rng = np.random.default_rng(1)
    A50 = fnp.asarray(rng.standard_normal((50, 50)))
    b50 = fnp.asarray(rng.standard_normal(50))
    A100 = fnp.asarray(rng.standard_normal((100, 100)))
    b100 = fnp.asarray(rng.standard_normal(100))
    c50 = _bill(lambda: fnp.linalg.tensorsolve(A50, b50))
    c100 = _bill(lambda: fnp.linalg.tensorsolve(A100, b100))
    assert c50 > 100  # not the flat 4-FLOP floor
    assert c100 >= 7 * c50  # ~ (100/50)^3


def test_tensorsolve_axes_bills_true_solved_dimension():
    # Regression pin for the axes-reordering under-bill: numpy's
    # tensorsolve(a, b, axes=...) transposes `axes` to the tail of `a`
    # *before* reshaping to the real (n, n) system it solves, so a formula
    # that reads the split point off the UN-transposed `a.shape` (e.g. the
    # earlier `n = prod(a.shape[b.ndim:])`) can diverge arbitrarily from the
    # true solved dimension once `axes` is passed.
    #
    # a=(6,1,3,2), b=(1,3,2), axes=(0,): axis 0 moves to the tail, giving a
    # transposed shape (1,3,2,6); the real system solved is n=6 (matches
    # `honest_n` cross-checked against a manual transpose+reshape+solve in
    # .superpowers/sdd/reprobe_tensorsolve.py-style probes). The old
    # ind=b.ndim formula instead read n=prod(a.shape[3:])=2 from the
    # untransposed shape -> solve_cost(2,1)=13 (analytical), 26 under
    # production rates (float64 dtype_rate=2.0) -- an under-bill even
    # relative to the pre-`ind`-threading baseline for this shape.
    from tests.test_dtype_cost import _billed_with_production_rates

    rng = np.random.default_rng(7)
    a = fnp.asarray(rng.standard_normal((6, 1, 3, 2)))
    b = fnp.asarray(rng.standard_normal((1, 3, 2)))
    billed, dt = _billed_with_production_rates(
        lambda: fnp.linalg.tensorsolve(a, b, axes=(0,))
    )
    assert dt == "float64"
    assert billed == 432  # solve_cost(6, 1)=216 x dtype_rate(float64)=2.0
    assert billed != 26  # the old under-billed (ind-based) production value
