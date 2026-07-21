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
