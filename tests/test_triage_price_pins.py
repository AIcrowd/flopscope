"""Production-price pins for the cost-model repricing pass ("triage").

Each test function locks in the billed price -- under the packaged
``default_weights.json`` (real weights x real dtype rates), not the
unit-weight ``_cost()``/``cost()`` helpers used elsewhere in this suite --
for one weight-tier family after its weight flip. Later repricing tasks
append their own families below.

``billed()`` wraps ``test_dtype_cost._billed_with_production_rates``, which
calls ``load_weights()`` itself, so every assertion here reads as
``weight * dtype_rate * <existing formula>``.
"""

import math

import numpy as np
from test_dtype_cost import _billed_with_production_rates

import flopscope.numpy as fnp


def billed(fn):
    return _billed_with_production_rates(fn)[0]


N = 1024  # power of two: ceil(log2) exact
LOG = math.ceil(math.log2(N))


# ---------------------------------------------------------------------------
# Task 1: x4 access tier -- sorts, sets, histograms, random reorder
# ---------------------------------------------------------------------------


def test_sort_family_bills_4x():
    a = np.random.default_rng(0).standard_normal(N).astype(np.float32)
    assert billed(lambda: fnp.sort(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.argsort(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.partition(fnp.asarray(a), 10)) == 4 * N
    assert billed(lambda: fnp.argpartition(fnp.asarray(a), 10)) == 4 * N
    assert (
        billed(lambda: fnp.searchsorted(fnp.asarray(np.sort(a)), fnp.asarray(a[:32])))
        == 4 * 32 * LOG
    )
    # lexsort: k keys x num_slices(=1, 1-D keys) x sort_cost(n); k=2 here.
    b = np.random.default_rng(10).standard_normal(N).astype(np.float32)
    assert (
        billed(lambda: fnp.lexsort((fnp.asarray(a), fnp.asarray(b)))) == 4 * 2 * N * LOG
    )


def test_set_and_histogram_family_bills_4x():
    a = np.random.default_rng(1).standard_normal(N).astype(np.float32)
    b = np.random.default_rng(2).standard_normal(N).astype(np.float32)
    assert billed(lambda: fnp.unique(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.unique_all(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.unique_counts(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.unique_inverse(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.unique_values(fnp.asarray(a))) == 4 * N * LOG
    assert billed(lambda: fnp.union1d(fnp.asarray(a), fnp.asarray(b))) == 4 * (
        2 * N
    ) * math.ceil(math.log2(2 * N))
    # intersect1d default (assume_unique=False): numpy unique()-sorts both
    # inputs first, so cost = sort_cost(n) + sort_cost(m) + sort_cost(n+m).
    assert billed(lambda: fnp.intersect1d(fnp.asarray(a), fnp.asarray(b))) == 4 * (
        2 * N * LOG + 2 * N * math.ceil(math.log2(2 * N))
    )
    assert billed(lambda: fnp.setdiff1d(fnp.asarray(a), fnp.asarray(b))) == 4 * (
        2 * N
    ) * math.ceil(math.log2(2 * N))
    assert billed(lambda: fnp.setxor1d(fnp.asarray(a), fnp.asarray(b))) == 4 * (
        2 * N
    ) * math.ceil(math.log2(2 * N))

    # histogram(bins=int) folds an int64 bin-count sentinel into its billing
    # dtypes tuple (mean_compute_dtype(a.dtype), platform-int); result_type
    # (float32, int64) == float64, so it always resolves at rate 2.0
    # regardless of the input's own dtype. Expected is 4(weight) x 2(rate) x
    # formula, not just 4x the formula.
    assert (
        billed(lambda: fnp.histogram(fnp.asarray(a), bins=16)) == 8 * N * 4
    )  # ceil(log2(16)) == 4
    # histogram2d/histogramdd always append an explicit float64 sentinel to
    # their billing dtypes too (counts are always float64) -- same x2 rate.
    assert billed(
        lambda: fnp.histogram2d(fnp.asarray(a), fnp.asarray(b), bins=16)
    ) == 8 * N * (4 + 4)
    assert billed(lambda: fnp.histogramdd(fnp.asarray(a), bins=16)) == 8 * N * 4
    # histogram_bin_edges has no int64/float64 sentinel; float32 input stays
    # float32 (mean_compute_dtype is a no-op for float dtypes) -> rate 1.0.
    assert billed(lambda: fnp.histogram_bin_edges(fnp.asarray(a), bins=16)) == 4 * N
    assert (
        billed(lambda: fnp.digitize(fnp.asarray(a), fnp.asarray(np.sort(b)[:16])))
        == 4 * N * 4
    )

    # bincount forces an int64 sentinel (heavier_billing_dtype(x.dtype,
    # platform int)), so -- like histogram -- it always resolves at rate 2.0,
    # even though the array below is int32 (own rate 1.0).
    x = np.random.default_rng(3).integers(0, 1000, size=N).astype(np.int32)
    assert billed(lambda: fnp.bincount(fnp.asarray(x))) == 8 * N


def test_random_reorder_bills_4x():
    assert billed(lambda: fnp.random.permutation(N)) == 4 * N
    g = fnp.random.default_rng(0)
    assert billed(lambda: g.permutation(N)) == 4 * N
    assert billed(lambda: g.choice(N, size=N)) == 4 * N

    arr = np.random.default_rng(4).standard_normal(N).astype(np.float32)
    assert billed(lambda: g.permuted(fnp.asarray(arr.copy()))) == 4 * N
    assert billed(lambda: g.shuffle(fnp.asarray(arr.copy()))) == 4 * N
    assert billed(lambda: fnp.random.shuffle(fnp.asarray(arr.copy()))) == 4 * N

    # random.sample is an alias of random_sample (uniform draws, see
    # src/flopscope/numpy/random/__init__.py:500) but always bills float64
    # (no dtype= parameter), so it resolves at rate 2.0 -- expected is
    # 4(weight) x 2(rate) x N, not just 4x N.
    assert billed(lambda: fnp.random.sample(N)) == 8 * N

    rs = fnp.random.RandomState(0)
    assert billed(lambda: rs.choice(N, size=N)) == 4 * N
    assert billed(lambda: rs.permutation(N)) == 4 * N
    assert billed(lambda: rs.shuffle(fnp.asarray(arr.copy()))) == 4 * N
