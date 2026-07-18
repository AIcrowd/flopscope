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


# ---------------------------------------------------------------------------
# Task 2: explicit indexing -- gathers x4 access tier; scatters/extract/
# compress x1 (elements touched)
# ---------------------------------------------------------------------------


def test_gather_tier_bills_4x_output():
    a = np.arange(1000, dtype=np.float32)
    idx = np.arange(0, 1000, 10)
    # built OUTSIDE the thunk as a plain numpy array: idx % 2 must not itself
    # be billed (it isn't -- plain numpy -- but keeping it out of the probe
    # matches this file's "inputs built outside the thunk" convention).
    idx_mod2 = idx % 2
    assert billed(lambda: fnp.take(fnp.asarray(a), fnp.asarray(idx))) == 4 * 100
    assert (
        billed(
            lambda: fnp.choose(
                fnp.asarray(idx_mod2),
                [fnp.asarray(a[:100]), fnp.asarray(a[100:200])],
            )
        )
        == 4 * 100
    )


def test_scatter_ops_bill_elements_touched():
    a = np.zeros(1000, dtype=np.float32)
    idx = np.arange(50, dtype=np.int32)
    vals = np.ones(50, dtype=np.float32)
    assert (
        billed(
            lambda: fnp.put(fnp.asarray(a.copy()), fnp.asarray(idx), fnp.asarray(vals))
        )
        == 50
    )
    sq = np.zeros((30, 40), dtype=np.float32)
    assert billed(lambda: fnp.fill_diagonal(fnp.asarray(sq.copy()), 7.0)) == 30


def test_extract_and_compress_bill_scan_plus_gather():
    a = np.zeros(1000, dtype=np.float32)
    mask = np.zeros(1000, dtype=bool)
    mask[:250] = (
        True  # built OUTSIDE the budget: the > comparison must not run in the thunk
    )
    assert billed(lambda: fnp.extract(fnp.asarray(mask), fnp.asarray(a))) == 1000
    assert (
        billed(lambda: fnp.compress(fnp.asarray(mask), fnp.asarray(a)))
        == 1000 + 4 * 250
    )


# ---------------------------------------------------------------------------
# Task 3: x1 materializing-copy tier -- array assembly & replication
# ---------------------------------------------------------------------------


def test_creation_and_copy_family_bills_output():
    a = np.arange(500, dtype=np.float32)
    b = np.arange(300, dtype=np.float32)
    assert billed(lambda: fnp.concatenate([fnp.asarray(a), fnp.asarray(b)])) == 800
    assert billed(lambda: fnp.vstack([fnp.asarray(a), fnp.asarray(a)])) == 1000
    assert billed(lambda: fnp.tile(fnp.asarray(b), 4)) == 1200
    assert billed(lambda: fnp.repeat(fnp.asarray(b), 3)) == 900
    assert billed(lambda: fnp.roll(fnp.asarray(a), 7)) == 500
    assert billed(lambda: fnp.full((20, 25), 3.0, dtype=np.float32)) == 500
    assert billed(lambda: fnp.delete(fnp.asarray(a), [0, 1])) == 498
    assert billed(lambda: fnp.append(fnp.asarray(a), fnp.asarray(b))) == 800
    assert billed(lambda: fnp.resize(fnp.asarray(b), (2, 300))) == 600


def test_creation_and_copy_family_remaining_ops_bill_output():
    """The rest of the 21 flipped keys, same x1 materializing-copy tier as
    ``test_creation_and_copy_family_bills_output`` above: concat/stack/hstack/
    dstack/column_stack (concatenate's siblings), row_stack (no ``deduct()``
    of its own -- bills exact parity with vstack), block/bmat (dtype-neutral:
    their ``deduct_after()`` declares ``dtypes=()``, so they always resolve at
    rate 1.0 regardless of the actual input dtype -- a pre-existing formula
    gap out of scope for a weight-only task, noted not fixed), insert,
    fromiter, full_like, and meshgrid's dense case.
    """
    a = np.arange(500, dtype=np.float32)
    b = np.arange(300, dtype=np.float32)
    assert billed(lambda: fnp.concat([fnp.asarray(a), fnp.asarray(b)])) == 800
    assert billed(lambda: fnp.stack([fnp.asarray(a), fnp.asarray(a)])) == 1000
    assert billed(lambda: fnp.hstack([fnp.asarray(a), fnp.asarray(b)])) == 800
    assert billed(lambda: fnp.dstack([fnp.asarray(a), fnp.asarray(a)])) == 1000
    assert billed(lambda: fnp.column_stack([fnp.asarray(a), fnp.asarray(a)])) == 1000
    # row_stack is a bare `return vstack(tup)`; the op_log record it produces
    # is literally vstack's, so its billed cost is vstack's by construction.
    row_stack_cost = billed(lambda: fnp.row_stack([fnp.asarray(a), fnp.asarray(a)]))
    vstack_cost = billed(lambda: fnp.vstack([fnp.asarray(a), fnp.asarray(a)]))
    assert row_stack_cost == vstack_cost == 1000
    # block/bmat: dtypes=() means rate is pinned to 1.0 regardless of dtype;
    # these values would be identical even on float64 inputs.
    assert billed(lambda: fnp.block([fnp.asarray(a), fnp.asarray(b)])) == 800
    m = np.arange(600, dtype=np.float32).reshape(20, 30)
    assert billed(lambda: fnp.bmat([[fnp.asarray(m), fnp.asarray(m)]])) == 1200
    ins_vals = np.array([9.0, 9.0], dtype=np.float32)  # built outside the thunk
    assert (
        billed(lambda: fnp.insert(fnp.asarray(a), [0, 1], fnp.asarray(ins_vals))) == 502
    )
    assert billed(lambda: fnp.fromiter(range(500), dtype=np.float32)) == 500
    assert billed(lambda: fnp.full_like(fnp.asarray(a), 3.0)) == 500
    # meshgrid: dense case only (sparse=True / copy=False are separate
    # argument-conditional branches of the same formula, unpinned here).
    x20 = np.arange(20, dtype=np.float32)
    y25 = np.arange(25, dtype=np.float32)
    assert billed(lambda: fnp.meshgrid(fnp.asarray(x20), fnp.asarray(y25))) == 1000


# ---------------------------------------------------------------------------
# Task 4: value-writing creation & layout copies -- diagonal length / numel
# ---------------------------------------------------------------------------


def test_writing_creation_bills_output_zeros_stay_free():
    assert billed(lambda: fnp.ones((40, 25), dtype=np.float32)) == 1000
    assert billed(lambda: fnp.eye(64, dtype=np.float32)) == 64
    assert (
        billed(lambda: fnp.eye(64, 32, k=40, dtype=np.float32)) == 0
    )  # k beyond width: no ones written
    assert billed(lambda: fnp.identity(50, dtype=np.float32)) == 50
    assert billed(lambda: fnp.zeros((40, 25), dtype=np.float32)) == 0
    assert billed(lambda: fnp.empty((40, 25), dtype=np.float32)) == 0
    base = np.ones((40, 25), dtype=np.float32)
    assert billed(lambda: fnp.zeros_like(fnp.asarray(base))) == 0
    assert billed(lambda: fnp.empty_like(fnp.asarray(base))) == 0
    # ones_like: not in the brief's given pin block above, added here so all
    # ten Task 4 ops have a dedicated pin (mirrors ones' numel(output)).
    assert billed(lambda: fnp.ones_like(fnp.asarray(base))) == 1000


def test_fft_shifts_bill_their_copy():
    a = np.arange(640, dtype=np.float32)
    assert billed(lambda: fnp.fft.fftshift(fnp.asarray(a))) == 640
    assert billed(lambda: fnp.fft.ifftshift(fnp.asarray(a))) == 640


def test_conditional_view_copies_bill_numel():
    a = np.arange(600, dtype=np.float32)
    assert billed(lambda: fnp.copy(fnp.asarray(a))) == 600
    assert billed(lambda: fnp.reshape(fnp.asarray(a), (20, 30))) == 600
    assert billed(lambda: fnp.ravel(fnp.asarray(a.reshape(20, 30)))) == 600
    assert billed(lambda: fnp.require(fnp.asarray(a), requirements=["C"])) == 600
