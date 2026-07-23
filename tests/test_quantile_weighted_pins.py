import math
import numpy as np
import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


# NOTE: tests/conftest.py's autouse `reset_global_budget` fixture calls
# reset_weights() before EVERY test function, which would undo a one-time
# setup_module() load before the first test body even runs. So each test
# below calls load_weights() itself (the convention used throughout the
# rest of this test suite, e.g. tests/test_dtype_cost.py) rather than
# relying on a module-level setup hook.

N = 8192
RNG = np.random.default_rng(0)
X = RNG.standard_normal(N)          # float64 -> rate 2.0
W = np.ones(N)
Q = np.arange(1, N + 1) / N
L = math.ceil(math.log2(N))         # 13


def test_scalar_unweighted_unchanged():
    load_weights()
    # common path must not move: n + 4*1 = 8196 flops, x2 for f64
    assert _billed(lambda: fnp.quantile(X, 0.5)) == (N + 4) * 2


def test_dense_unweighted_clears_sort():
    load_weights()
    dense = _billed(lambda: fnp.quantile(X, Q, method="inverted_cdf"))
    srt = _billed(lambda: fnp.sort(X))
    assert dense >= srt                      # extraction closed
    assert dense == (N * (1 + 4 * L) + 4 * N) * 2   # 933,888


def test_scalar_weighted_clears_sort():
    load_weights()
    wq = _billed(lambda: fnp.quantile(X, 0.5, weights=W, method="inverted_cdf"))
    assert wq >= _billed(lambda: fnp.sort(X))
    assert wq == (4 * N * L + 3 * N + 1 * (L + 4)) * 2   # 901,154


def test_weighted_ge_unweighted_and_all_four_ops():
    load_weights()
    for op in (fnp.quantile, fnp.percentile, fnp.nanquantile, fnp.nanpercentile):
        qval = 0.5 if "percentile" not in op.__name__ else 50.0
        uw = _billed(lambda: op(X, qval))
        wt = _billed(lambda: op(X, qval, weights=W, method="inverted_cdf"))
        assert wt >= uw >= (N + 4) * 2 - 2, op.__name__
