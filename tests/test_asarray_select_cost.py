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
# rest of this test suite, e.g. tests/test_dtype_cost.py).

NC = np.arange(64, dtype=np.float32).reshape(8, 8)[
    :, ::2
]  # non-contiguous f32, 32 elems


def test_asarray_forced_copy_charges():
    load_weights()
    assert _billed(lambda: fnp.asarray(NC, copy=True)) == 32
    assert _billed(lambda: fnp.asarray(NC, order="C")) == 32
    assert _billed(lambda: fnp.asarray(NC, order="F")) == 32


def test_asarray_view_is_free():
    load_weights()
    contig = np.ascontiguousarray(NC)
    assert _billed(lambda: fnp.asarray(contig)) == 0  # no copy -> view
    assert _billed(lambda: fnp.asarray(contig, dtype=np.float32)) == 0


def test_asarray_list_matches_array():
    load_weights()
    data = [[1.0, 2.0], [3.0, 4.0]]
    assert _billed(lambda: fnp.asarray(data)) == _billed(lambda: fnp.array(data))


def test_select_default_promotes_bill():
    load_weights()
    cond = [np.array([True, False, True, False])]
    ch32 = [np.arange(4, dtype=np.float32)]
    honest_f64 = _billed(
        lambda: fnp.select(cond, [np.arange(4, dtype=np.float64)], default=0)
    )
    # np.float64 / f64-array default -> f64 output -> must bill like f64 choices
    assert (
        _billed(lambda: fnp.select(cond, ch32, default=np.float64(1.5))) == honest_f64
    )
    assert (
        _billed(lambda: fnp.select(cond, ch32, default=np.full(4, 1.5))) == honest_f64
    )
    # weak python float stays weak (no promotion)
    assert _billed(lambda: fnp.select(cond, ch32, default=1.5)) == _billed(
        lambda: fnp.select(cond, ch32, default=np.float32(1.5))
    )
