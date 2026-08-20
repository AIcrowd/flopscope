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

import array
import json
import math

import numpy as np
import pytest
from test_dtype_cost import _billed_with_production_rates

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import get_dtype_rate, get_weight, load_weights


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

    # Module-level random.choice must join the x4 reorder tier: numpy's module
    # surface delegates to the legacy RandomState singleton (the same selection
    # machinery as RandomState.choice / default_rng().choice), so weight 1.0 was
    # a cheaper alias route around the agreed reorder tier. An int32 pool bills
    # dtype-neutral (rate 1): 4 x 100 = 400, at parity with the Generator
    # surface for the identical draw.
    pool = np.arange(1000, dtype=np.int32)
    assert (
        billed(lambda: fnp.random.choice(fnp.asarray(pool), size=100, replace=True))
        == 400
    )
    assert billed(
        lambda: fnp.random.choice(fnp.asarray(pool), size=100, replace=True)
    ) == billed(lambda: g.choice(fnp.asarray(pool), size=100, replace=True))

    # random.sample is an alias of random_sample (uniform draws, see
    # src/flopscope/numpy/random/__init__.py:500) -- a plain draw, not a
    # selection -- so it bills at the same 1x tier as its twin. It always
    # bills float64 (no dtype= parameter), so expected is 1(weight) x 2(rate) x N.
    assert billed(lambda: fnp.random.sample(N)) == 2 * N

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


def test_place_and_putmask_bill_numel_input_not_mask_count():
    """place/putmask both declare ``dtypes=(arr's own dtype,)`` and cost
    ``numel(input)`` -- the WHOLE array is scanned regardless of how many
    positions the mask actually selects (unlike put's numel(indices)
    formula above). A float64 array resolves at dtype_rate 2.0, proving the
    dtype declaration is live (a dtype-neutral bug would still read 1000)."""
    a = np.zeros(1000, dtype=np.float32)
    vals = np.ones(50, dtype=np.float32)
    mask = np.zeros(1000, dtype=bool)
    mask[:50] = True  # built outside the thunk: the slice-assign must not bill
    assert (
        billed(
            lambda: fnp.place(
                fnp.asarray(a.copy()), fnp.asarray(mask), fnp.asarray(vals)
            )
        )
        == 1000
    )
    assert (
        billed(
            lambda: fnp.putmask(
                fnp.asarray(a.copy()), fnp.asarray(mask), fnp.asarray(vals)
            )
        )
        == 1000
    )
    a64 = np.zeros(1000, dtype=np.float64)
    vals64 = np.ones(50, dtype=np.float64)
    assert (
        billed(
            lambda: fnp.place(
                fnp.asarray(a64.copy()), fnp.asarray(mask), fnp.asarray(vals64)
            )
        )
        == 2000
    )
    assert (
        billed(
            lambda: fnp.putmask(
                fnp.asarray(a64.copy()), fnp.asarray(mask), fnp.asarray(vals64)
            )
        )
        == 2000
    )


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
    of its own -- bills exact parity with vstack), block/bmat (dtype-aware as
    of the final-review fix below: ``deduct_after()`` reads the promoted
    output dtype via ``set_dtypes()`` instead of declaring ``dtypes=()``, so
    they resolve at the real dtype rate -- see
    ``test_choose_block_bmat_bill_wide_dtypes`` for the float64/complex128
    proof), insert, fromiter, full_like, and meshgrid's dense case.
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
    # These two stay float32 (rate 1.0), so the values below are unaffected by
    # the dtype-awareness fix; test_choose_block_bmat_bill_wide_dtypes below
    # is what actually exercises the fixed rate/complex-factor path.
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
    # require with an explicit dtype= casts/materializes at the REQUESTED
    # dtype (mirrors full_like's resolution): int32 input (rate 1.0)
    # requested as float64 resolves rate 2.0 -- 1000 x 2.0(f64) x 1.0 = 2000,
    # not the input width's 1000.
    b = np.arange(1000, dtype=np.int32)
    assert (
        billed(
            lambda: fnp.require(fnp.asarray(b), dtype=np.float64, requirements=["C"])
        )
        == 2000
    )


# ---------------------------------------------------------------------------
# Task 5: views stay free -- split family, broadcast_to, diagonal, unstack
# ---------------------------------------------------------------------------


def test_split_broadcast_and_diagonal_stay_free():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    v = np.arange(24, dtype=np.float32)
    sq = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert billed(lambda: fnp.split(fnp.asarray(a), 2, axis=0)) == 0
    assert billed(lambda: fnp.hsplit(fnp.asarray(a), 2)) == 0
    assert billed(lambda: fnp.vsplit(fnp.asarray(a), 2)) == 0
    assert billed(lambda: fnp.array_split(fnp.asarray(v), 7)) == 0
    assert billed(lambda: fnp.broadcast_to(fnp.asarray(v[:6]), (4, 6))) == 0
    assert billed(lambda: fnp.diagonal(fnp.asarray(sq))) == 0


@pytest.mark.skipif(not hasattr(np, "unstack"), reason="requires numpy >= 2.1")
def test_unstack_stays_free():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert billed(lambda: fnp.unstack(fnp.asarray(a))) == 0


# ---------------------------------------------------------------------------
# Task 6: where / select / piecewise / apply_along_axis rework
# ---------------------------------------------------------------------------


def test_where_three_arg_bills_4x_broadcast_output():
    cond = np.ones(1000, dtype=bool)
    x = np.arange(1000, dtype=np.float32)
    assert (
        billed(lambda: fnp.where(fnp.asarray(cond), fnp.asarray(x), fnp.asarray(x)))
        == 4 * 1000
    )
    # cond smaller than result: broadcast size is what's billed
    cond_row = np.ones((1, 4), dtype=bool)
    big = np.ones((300, 4), dtype=np.float32)
    assert (
        billed(
            lambda: fnp.where(fnp.asarray(cond_row), fnp.asarray(big), fnp.asarray(big))
        )
        == 4 * 1200
    )


def test_where_one_arg_prices_as_nonzero():
    cond = np.ones(1000, dtype=bool)
    one_arg = billed(lambda: fnp.where(fnp.asarray(cond)))
    nz = billed(lambda: fnp.nonzero(fnp.asarray(cond)))
    assert one_arg == nz == 1000


def test_select_and_piecewise_bill_per_condition():
    x = np.arange(500, dtype=np.float32)
    c1, c2 = np.zeros(500, dtype=bool), np.ones(500, dtype=bool)
    assert (
        billed(
            lambda: fnp.select(
                [fnp.asarray(c1), fnp.asarray(c2)], [fnp.asarray(x), fnp.asarray(x)]
            )
        )
        == 2 * 500
    )
    assert (
        billed(
            lambda: fnp.piecewise(
                fnp.asarray(x), [fnp.asarray(c1), fnp.asarray(c2)], [0.0, 1.0]
            )
        )
        == 2 * 500
    )


def test_piecewise_ndarray_condlist_counts_rows():
    """numpy promotion parity: a 2-D bool ndarray condlist is a stack of
    per-ROW conditions (bit-identical results to the list form), so it must
    bill numel * n_rows -- not collapse to one condition. A bare 1-D bool
    array really is ONE condition (numpy wraps it as [condlist])."""
    x = np.arange(500, dtype=np.float32)
    c1, c2 = np.zeros(500, dtype=bool), np.ones(500, dtype=bool)
    stacked = np.stack([c1, c2])  # (2, 500): two row-conditions
    assert (
        billed(lambda: fnp.piecewise(fnp.asarray(x), fnp.asarray(stacked), [0.0, 1.0]))
        == 2 * 500
    )
    assert (
        billed(lambda: fnp.piecewise(fnp.asarray(x), fnp.asarray(c2), [1.0])) == 1 * 500
    )


def test_apply_along_axis_wrapper_bills_1x_output():
    m = np.arange(1000, dtype=np.float32).reshape(20, 50)
    # scalar-returning pure-python func1d: no separately billed fnp ops inside.
    # Return np.float32 so heavier-of-dtype billing stays at rate 1 (a python
    # float would make the result float64 and double the pin).
    assert (
        billed(
            lambda: fnp.apply_along_axis(lambda row: np.float32(0.0), 1, fnp.asarray(m))
        )
        == 20
    )


# ---------------------------------------------------------------------------
# Task 7: diag family + triangular constructors
# ---------------------------------------------------------------------------


def test_diag_family_bills_written_values_only():
    sq = np.arange(64.0, dtype=np.float32).reshape(8, 8)
    v = np.arange(10, dtype=np.float32)
    assert billed(lambda: fnp.diag(fnp.asarray(sq))) == 0  # 2-D extract: view
    assert billed(lambda: fnp.diag(fnp.asarray(v), k=3)) == 10  # 1-D construct: len(v)
    assert billed(lambda: fnp.diagflat(fnp.asarray(v))) == 10
    assert billed(lambda: fnp.triu(fnp.asarray(sq))) == 36  # 8*9/2 kept
    assert billed(lambda: fnp.tril(fnp.asarray(sq), k=-1)) == 28  # below diagonal
    assert billed(lambda: fnp.triu(fnp.asarray(sq), k=2)) == 21


def test_triu_batch_leading_dims_multiply():
    """A (3, 8, 8) stack bills 3x the per-matrix kept-triangle count -- numpy
    applies triu/tril to the final two axes and leaves leading batch
    dimensions alone, so the cost must multiply in the batch size rather
    than treat the whole stack as one flat triangle."""
    stacked = np.ones((3, 8, 8), dtype=np.float32)
    assert billed(lambda: fnp.triu(fnp.asarray(stacked))) == 3 * 36


# ---------------------------------------------------------------------------
# Task 8: index generators -- tri family, unravel/ravel_multi_index,
# mask_indices, broadcast_shapes
# ---------------------------------------------------------------------------


def test_index_generators_bill_their_outputs():
    assert billed(lambda: fnp.tri(20, 30, dtype=np.float32)) == 600
    assert billed(lambda: fnp.tril_indices(8)) == 2 * 36
    assert billed(lambda: fnp.triu_indices(8, k=2)) == 2 * 21
    assert billed(lambda: fnp.diag_indices(9, ndim=3)) == 3 * 9
    assert (
        billed(lambda: fnp.unravel_index(fnp.asarray(np.arange(50)), (10, 20)))
        == 2 * 50
    )
    assert (
        billed(
            lambda: fnp.ravel_multi_index(
                (
                    fnp.asarray(np.zeros(50, dtype=np.int64)),
                    fnp.asarray(np.zeros(50, dtype=np.int64)),
                ),
                (10, 20),
            )
        )
        == 50
    )
    # np.triu as mask_func: plain-numpy callable, NOT billed -> isolates
    # mask_indices' own cost, which is the scanned n x n mask (the numpy
    # ``nonzero`` convention): numel(mask) = n*n, priced at the mask's own
    # (int) dtype rate. The returned index count plays no part.
    n = 8
    int_rate = 2 if np.dtype(int).itemsize == 8 else 1  # int64 -> 2.0, int32 -> 1.0
    # n*n * int_rate = 8*8 * 2 = 128 (int64) / 64 (int32)
    mask_indices_cost = n * n * int_rate
    assert billed(lambda: fnp.mask_indices(n, np.triu)) == mask_indices_cost
    # fnp.triu as mask_func: see test_mask_indices_fnp_mask_func_bills_on_top below.
    assert billed(lambda: fnp.broadcast_shapes((4, 6), (6,))) == 3


def test_mask_indices_fnp_mask_func_bills_on_top():
    """An fnp-wrapped mask_func (e.g. fnp.triu) bills its own cost on top of
    mask_indices' own scan cost, per this op's docstring -- it no longer raises.

    numpy's ``mask_indices`` body runs ``a = mask_func(m, k)`` (m = an (n,n)
    int matrix) then ``nonzero(a != 0)``. Previously an fnp mask_func returned
    a FlopscopeArray that leaked into that bare top-level ``nonzero`` and
    tripped the wrapper-depth guard. The wrapper now strips mask_func's result
    to a base ndarray, so numpy's own ``nonzero`` runs on a plain array while
    the fnp mask_func still bills its own cost.

    n=8, triu at offset 0 -> k = 8*9/2 = 36 selected pairs:
      - mask_indices' own cost: the scanned (n, n) mask, i.e. numel(mask) =
        n*n = 64 elements at the mask's int dtype rate (int64 -> 2.0 on
        Linux/mac, int32 -> 1.0 on Windows) -- the numpy ``nonzero``
        convention; the returned index count plays no part.
      - fnp.triu on numpy's internal ``ones((n,n), int)``: kept upper triangle
        = 36 elements at that same int dtype's rate
      total = 64 * int_rate + 36 * int_rate = 100 * int_rate -> 200 where
      default int is int64.
    """
    n, k = 8, 8 * 9 // 2  # 36
    # numpy builds ones((n,n), int); triu bills 36 kept elements at that rate.
    int_rate = 2 if np.dtype(int).itemsize == 8 else 1  # int64 -> 2.0, int32 -> 1.0
    # n*n * int_rate = 8*8 * 2 = 128 (int64) / 64 (int32)
    mask_indices_cost = n * n * int_rate
    triu_own_cost = k * int_rate  # triu's own kept-triangle
    expected = mask_indices_cost + triu_own_cost
    assert billed(lambda: fnp.mask_indices(n, fnp.triu)) == expected


def test_np_nonzero_top_level_routes_to_fnp():
    """Top-level ``np.nonzero(FlopscopeArray)`` routes through fnp.nonzero
    (billed numel(input)) instead of raising numpy's NEP-18 "no implementation
    found" TypeError -- matching its set/unique dispatch siblings and the
    already-overridden ``.nonzero()`` method. The auto-route emits the standard
    UserWarning (nudging callers to fnp.nonzero), which we positively assert."""
    a = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.int32)  # built outside the thunk
    with pytest.warns(UserWarning, match="nonzero"):
        used = billed(lambda: np.nonzero(fnp.asarray(a)))
    assert used == a.size  # nonzero = numel(input)


def test_tril_indices_from_and_triu_indices_from_bill_their_outputs():
    """*_from siblings take an array argument (only its shape matters) but
    bill the same numel-of-returned-index-arrays formula as the non-_from
    forms, dtype-neutral."""
    a = np.ones((8, 8), dtype=np.float32)
    assert billed(lambda: fnp.tril_indices_from(fnp.asarray(a))) == 2 * 36
    assert billed(lambda: fnp.triu_indices_from(fnp.asarray(a), k=2)) == 2 * 21


def test_diag_indices_from_bills_its_output():
    a = np.ones((9, 9, 9), dtype=np.float32)
    assert billed(lambda: fnp.diag_indices_from(fnp.asarray(a))) == 3 * 9


def test_indices_already_charged_weight_one_no_change():
    """indices was ALREADY counted_custom at weight 1.0 before Task 8 -- this
    pin confirms no change. Unlike the index-generator family above, indices
    bills its REAL output dtype (not dtype-neutral): dense (2,3,4) int64 grid
    -> numel(output)=24 elements x dtype_rate(int64)=2.0 x weight 1.0 = 48."""
    assert billed(lambda: fnp.indices((3, 4))) == 48


# ---------------------------------------------------------------------------
# Task 9: pad -- writes-consistent base (numel(output)) + mode extras
# ---------------------------------------------------------------------------


def test_pad_bills_full_output_plus_mode_extras():
    a = np.arange(100, dtype=np.float32)
    assert billed(lambda: fnp.pad(fnp.asarray(a), 10)) == 120  # constant: numel(out)
    assert billed(lambda: fnp.pad(fnp.asarray(a), 10, mode="edge")) == 120
    assert (
        billed(lambda: fnp.pad(fnp.asarray(a), 10, mode="linear_ramp")) == 140
    )  # out + (out - in)
    stat = billed(lambda: fnp.pad(fnp.asarray(a), 10, mode="mean"))
    assert stat > 120  # out + stat cost


def test_pad_movement_and_odd_reflect_modes():
    """wrap and (default even) reflect are movement modes -- 0 extra, so they
    bill the same numel(out)=120 as constant/edge above. symmetric with
    reflect_type='odd' takes the same +(out-in) extra as linear_ramp."""
    a = np.arange(100, dtype=np.float32)
    assert billed(lambda: fnp.pad(fnp.asarray(a), 10, mode="wrap")) == 120
    assert billed(lambda: fnp.pad(fnp.asarray(a), 10, mode="reflect")) == 120
    assert (
        billed(
            lambda: fnp.pad(fnp.asarray(a), 10, mode="symmetric", reflect_type="odd")
        )
        == 140
    )


def test_pad_2d_multiplies_the_full_output_base():
    """A (10, 10) input padded by 2 on every side of every axis produces a
    14x14=196-element output; the writes-consistent base charges every one of
    those cells (not just the padded border), locking multi-axis numel(out)."""
    a2 = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert billed(lambda: fnp.pad(fnp.asarray(a2), 2)) == 196


def test_pad_empty_input_floors_at_one():
    """A 0-element input padded by 0 (numpy accepts mode='constant'/'empty' on
    an empty axis; every other mode rejects it) produces a 0-element output --
    the writes-consistent base would charge 0 flop_cost, but pad still ran, so
    the shared floor-of-1 convention applies."""
    a = np.zeros(0, dtype=np.float32)
    assert billed(lambda: fnp.pad(fnp.asarray(a), 0)) == 1


def test_pad_broadcastable_pad_width_bills_the_real_output_not_zero():
    """Regression: pad_width forms that numpy.pad broadcasts -- a per-axis
    column ``[[1], [2]]`` (shape (ndim, 1)) and a single ``[[5]]`` (shape
    (1, 1)) -- must bill the full output they actually produce.

    numpy.pad normalizes these via its own ``_as_pairs`` broadcasting; an
    earlier flopscope normalizer was NARROWER and raised on them, and the
    cost path swallowed that into a 0 FLOP bill while numpy.pad went on to
    return a full-size padded array -- a real-output-for-0-FLOPs budget
    bypass, reproducible for EVERY mode. The normalizer now mirrors
    ``_as_pairs`` exactly, so the bill equals the true output size."""
    m = np.zeros((3, 4), dtype=np.float32)  # -> out (3+1+1, 4+2+2) = (5, 8) = 40
    assert billed(lambda: fnp.pad(fnp.asarray(m), [[1], [2]])) == 40
    # mean/linear_ramp still bill numel(out)=40 as their base + a mode extra,
    # so strictly more than the movement-mode 40.
    assert billed(lambda: fnp.pad(fnp.asarray(m), [[1], [2]], mode="mean")) > 40
    assert billed(lambda: fnp.pad(fnp.asarray(m), [[1], [2]], mode="linear_ramp")) > 40
    # single (1, 1) pad_width broadcasts to (5, 5) on a length-10 vector -> 20
    v = np.zeros(10, dtype=np.float32)
    assert billed(lambda: fnp.pad(fnp.asarray(v), [[5]])) == 20


def test_pad_malformed_pad_width_still_raises_numpy_error():
    """A pad_width numpy.pad itself rejects must surface numpy's own error
    (the mirrored normalizer raises the same ValueError), NOT get swallowed
    into a 0 bill. ``[[1, 2], [3, 4]]`` (shape (2, 2)) is not broadcastable to
    (ndim=1, 2)."""
    v = np.arange(10.0)
    with pytest.raises(ValueError):
        billed(lambda: fnp.pad(fnp.asarray(v), [[1, 2], [3, 4]], mode="constant"))


# ---------------------------------------------------------------------------
# Task 10: windows bill their derived per-sample constant; save/savez bill
# the bytes they write (load and from_dlpack stay free)
# ---------------------------------------------------------------------------


def test_windows_bill_derived_constants():
    # windows return float64 by design; dtype_rate 2 applies -> 2 * 18 * M
    assert billed(lambda: fnp.hamming(100)) == 2 * 18 * 100
    assert billed(lambda: fnp.hanning(100)) == 2 * 18 * 100
    assert billed(lambda: fnp.kaiser(100, 14.0)) == 2 * 23 * 100  # unchanged


def test_io_save_bills_load_stays_free(tmp_path):
    a = np.arange(250, dtype=np.float32)
    f = str(tmp_path / "w.npy")
    # numel(250) + shape-header (ndim=1 -> 1*8 bytes).
    assert billed(lambda: fnp.save(f, fnp.asarray(a))) == 4 * (250 + 8)
    assert billed(lambda: fnp.load(f)) == 0
    src = np.arange(64, dtype=np.float32)
    assert billed(lambda: fnp.from_dlpack(src)) == 0  # stays free everywhere (Q9)


def test_io_savez_bills_sum_of_saved_arrays_including_meta(tmp_path):
    """__meta__ is serialized to a uint8 byte blob and written to the archive
    like any other array (see ``_prepare``), so it bills the same 4*numel
    egress cost -- excluding it was a budget-bypass (a participant could
    round-trip unlimited data through __meta__ for a flat, size-independent
    cost); see test_io_savez_large_meta_bills_proportionally_not_flat below.
    The archive's MEMBER NAMES ("a", "b", and "__meta__") are billed too --
    see test_io_savez_name_channel_bills_proportionally_to_name_bytes below.
    Every billed array also bills an 8-byte-per-dimension shape-header channel
    (the ``.npy``/``.npz`` header encodes shape ints a participant controls),
    and the names blob -- ingested server-side as one synthetic 1-D uint8
    array -- bills one more 8-byte header of its own."""
    a = np.arange(250, dtype=np.float32)
    b = np.arange(150, dtype=np.float32)
    meta = {"k": 1}
    meta_len = len(json.dumps(meta).encode("utf-8"))
    name_bytes = len("a") + len("b") + len("__meta__")
    shape_header_bytes = 3 * 8  # a, b, __meta__ blob: all 1-D
    names_shape_header = 8  # non-empty name blob
    fz = str(tmp_path / "wz.npz")
    assert billed(
        lambda: fnp.savez(fz, a=fnp.asarray(a), b=fnp.asarray(b), __meta__=meta)
    ) == 4 * (400 + meta_len + name_bytes + shape_header_bytes + names_shape_header)
    # savez_compressed shares savez's exact formula (4*sum(numel), meta+names included).
    fzc = str(tmp_path / "wzc.npz")
    assert billed(
        lambda: fnp.savez_compressed(
            fzc, a=fnp.asarray(a), b=fnp.asarray(b), __meta__=meta
        )
    ) == 4 * (400 + meta_len + name_bytes + shape_header_bytes + names_shape_header)


def test_io_savez_large_meta_bills_proportionally_not_flat(tmp_path):
    """Exploit regression (budget bypass): before the fix, __meta__ was
    excluded from billing entirely, so ``savez(path, __meta__={...huge...})``
    billed only the floor-of-1 cost (4 FLOPs) no matter how much data the
    blob smuggled to disk -- e.g. a 2,000,000-float payload (~10MB on disk)
    billed 4 FLOPs. A large __meta__ must now bill 4*len(json-encoded-blob),
    dominating a small named array's own cost, not a flat 4-FLOP floor."""
    payload = {"payload": [0.0] * 100_000}
    meta_len = len(json.dumps(payload).encode("utf-8"))
    small = np.arange(10, dtype=np.float32)
    name_bytes = len("a") + len("__meta__")
    shape_header_bytes = 2 * 8  # small + __meta__ blob: both 1-D
    names_shape_header = 8  # non-empty name blob
    f = str(tmp_path / "exploit.npz")
    total = billed(lambda: fnp.savez(f, a=fnp.asarray(small), __meta__=payload))
    assert total == 4 * (
        10 + meta_len + name_bytes + shape_header_bytes + names_shape_header
    )
    array_only_cost = 4 * 10
    assert total > 1000 * array_only_cost  # dominated by meta, not the tiny array
    assert total != 4  # the pre-fix floor-of-1 exploit value


def test_io_savez_name_channel_bills_proportionally_to_name_bytes(tmp_path):
    """Exploit regression (budget bypass), NAME channel: before this fix, the
    archive's MEMBER NAMES (savez's kwargs keys) were written to the archive
    and recovered verbatim by ``load``, but never counted toward billing --
    only array VALUES were billed (4*numel each). A participant could
    smuggle megabytes of data through many tiny 1-element arrays given huge
    names (a .npz zip member name can run to tens of thousands of bytes) for
    near-zero cost -- e.g. 200 one-element arrays with 60,000-char names
    smuggle ~12MB of recoverable data while the array values alone bill only
    4*200=800 FLOPs. Member names must now bill 4*sum(len(name bytes)),
    dominating the tiny arrays' own cost, not ride along for free."""
    n_arrays = 200
    name_len = 60_000
    names = [f"{i:06d}" + "x" * (name_len - 6) for i in range(n_arrays)]
    assert all(len(n) == name_len for n in names)
    raw = {name: np.array([1.0], dtype=np.float32) for name in names}
    name_bytes = sum(len(n.encode("utf-8")) for n in names)
    shape_header_bytes = n_arrays * 8  # 200 arrays, each 1-D
    names_shape_header = 8  # non-empty name blob
    f = str(tmp_path / "smuggle.npz")
    total = billed(lambda: fnp.savez(f, **{k: fnp.asarray(v) for k, v in raw.items()}))
    array_only_cost = 4 * n_arrays  # the pre-fix exploit value (800 FLOPs)
    assert total == 4 * (
        n_arrays + name_bytes + shape_header_bytes + names_shape_header
    )
    assert total > 1000 * array_only_cost  # dominated by names, not array values
    assert total != array_only_cost  # the pre-fix name-channel exploit value


# ---------------------------------------------------------------------------
# Task 11: __getitem__ bills advanced indexing (new surface). Basic indexing
# (int/slice/newaxis/Ellipsis, or a tuple thereof) stays free -- it returns a
# view. Advanced indexing bills under "getitem": integer-array (fancy)
# indexing costs 4*numel(output) (matching take); a boolean-mask part ALSO
# adds numel(mask) for the scan (matching compress).
# ---------------------------------------------------------------------------


def test_getitem_slices_free_fancy_4x_mask_scan_plus_4x():
    a = np.arange(1000, dtype=np.float32)
    idx = np.arange(0, 1000, 10)
    mask = np.zeros(1000, dtype=bool)
    mask[:250] = True
    # Build every FlopscopeArray input outside the billed() thunk so
    # construction (free, but still routed through require_budget()) never
    # contaminates the single-op measurement below.
    fa = fnp.asarray(a)
    fidx = fnp.asarray(idx)
    fmask = fnp.asarray(mask)
    m = fnp.asarray(a.reshape(20, 50))
    assert billed(lambda: fa[10:500:2]) == 0
    assert billed(lambda: fa[fidx]) == 4 * 100
    assert billed(lambda: fa[fmask]) == 1000 + 4 * 250
    assert billed(lambda: m[3]) == 0
    assert billed(lambda: m[:, ::5]) == 0
    # 2-D fancy on axis 0: output (3, 50) = 150 elements -> 4*150.
    assert billed(lambda: m[[0, 2, 4]]) == 4 * 150


def test_getitem_bool_scalar_and_0d_array_indices_are_advanced():
    """Regression: numpy routes a bare boolean scalar AND a 0-d int/bool array
    through the advanced-index COPY path (shares_memory False), not a view --
    so both must bill, not fall through the ndim>0 / ndarray-only gate as free.

    - ``fa[True]`` / ``fa[np.bool_(True)]``: a (1, 1000) copy -> 1 (scalar scan)
      + 4*1000 (gather) = 4001.
    - ``fbig[np.array(7)]`` on a (500, 2000) array: a 2000-elem row COPY (not a
      view, unlike ``fbig[7]``) -> 4*2000 = 8000.
    - ``fa[np.array(7)]`` on a 1-D array: gathers a single scalar -> 4*1 = 4.
    """
    a = np.arange(1000, dtype=np.float32)
    big = np.arange(500 * 2000, dtype=np.float32).reshape(500, 2000)
    # Index operands (bool scalars, 0-d int arrays) built outside billed().
    bt = np.bool_(True)
    zi = np.array(7)
    fa = fnp.asarray(a)
    fbig = fnp.asarray(big)
    assert billed(lambda: fa[True]) == 1 + 4 * 1000
    assert billed(lambda: fa[bt]) == 1 + 4 * 1000
    assert billed(lambda: fbig[zi]) == 4 * 2000
    assert billed(lambda: fa[zi]) == 4 * 1
    # An integer SCALAR stays basic (a view) -- it must NOT bill.
    assert billed(lambda: fbig[7]) == 0


# ---------------------------------------------------------------------------
# Task 12 (final-review fix, CRITICAL): __getitem__ classifies a tuple/range
# PART as advanced too, not just list/ndarray/bool-scalar. numpy performs
# advanced (copying) indexing for ANY array-like integer/bool sequence part;
# a tuple or range part gathers exactly like the list form, so under the
# list-only check this was an unbounded 0-FLOP gather -- ``arr[range(n)]``
# for any n, or ``arr[tuple_a, tuple_b]``, billed nothing for a real copy.
# The top-level key, when it IS a tuple, is unchanged: it is multi-axis
# BASIC indexing, already split into per-axis parts before this check runs
# -- see test_getitem_top_level_tuple_key_still_splits_to_basic_parts below
# for that distinction.
# ---------------------------------------------------------------------------


def test_getitem_tuple_and_range_parts_are_advanced():
    fv = fnp.asarray(np.arange(500, dtype=np.float64))
    # A bare range key: one part, a range -- gathers all 500 elements.
    # 4*500 * dtype_rate(float64)=2.0.
    assert billed(lambda: fv[range(500)]) == 4 * 500 * 2

    big = fnp.asarray(np.arange(500 * 2000, dtype=np.float64).reshape(500, 2000))
    rows = tuple(range(100))
    cols = tuple(range(100))
    rows_l = list(rows)
    cols_l = list(cols)
    # Two top-level parts, each a plain tuple: numpy pairs them elementwise
    # (NOT an outer product), gathering 100 elements at (rows[i], cols[i]).
    # The list form was already correct; both must agree bit-exact.
    assert billed(lambda: big[rows, cols]) == 4 * 100 * 2
    assert billed(lambda: big[rows_l, cols_l]) == billed(lambda: big[rows, cols])

    v = fnp.asarray(np.arange(5, dtype=np.float64))
    mask_tuple = (True, False, True, False, True)
    mask_list = list(mask_tuple)
    # A tuple PART containing bools (wrapped in a 1-tuple so it survives the
    # top-level split as ONE part, not five bool-scalar parts -- see the
    # regression test below) is a boolean mask gather: 5 scan + 4*3 gather =
    # 17, at dtype_rate(float64)=2.0. Must agree with the list form.
    assert billed(lambda: v[(mask_tuple,)]) == (5 + 4 * 3) * 2
    assert billed(lambda: v[(mask_tuple,)]) == billed(lambda: v[mask_list])


def test_getitem_top_level_tuple_key_still_splits_to_basic_parts():
    """Guard the subtlety the CRITICAL fix must not disturb: whether a tuple
    is "advanced" depends on WHERE it sits, not that it IS a tuple. A
    top-level tuple key is multi-axis indexing, unpacked into per-axis parts
    before the advanced check runs, so broadening what counts as an advanced
    *part* (previous test) leaves these untouched.
    """
    m = fnp.asarray(np.arange(1000, dtype=np.float64).reshape(20, 50))
    assert billed(lambda: m[1:3, ::2]) == 0  # parts are slices: basic
    assert billed(lambda: m[3]) == 0  # single int: basic
    assert billed(lambda: m[3, 4]) == 0  # two int parts: basic

    v = fnp.asarray(np.arange(10, dtype=np.float64))
    # key IS the 3-tuple (0, 2, 4) -> splits into three INT parts -> basic ->
    # falls through to numpy, which raises its own "too many indices" error
    # for a 1-D array (3 indices, 1 axis). Must not be billed.
    with pytest.raises(IndexError, match="too many indices"):
        v[(0, 2, 4)]
    # A 1-tuple *containing* the 3-tuple is a DIFFERENT key: one part, a
    # genuine advanced gather, matching the list form bit-exact.
    assert billed(lambda: v[(0, 2, 4),]) == 4 * 3 * 2
    assert billed(lambda: v[(0, 2, 4),]) == billed(lambda: v[[0, 2, 4]])


def test_getitem_any_arraylike_sequence_part_is_advanced_others_fall_through():
    """Hardening: the advanced-part classifier is numpy-faithful (any part that
    coerces to an integer/bool array gathers), not an enumerated type list --
    so array.array, memoryview, and any future sequence/buffer bill like the
    list form, while parts numpy can't turn into a b/i/u index array fall
    through to numpy unbilled. Not grader-reachable (the client wire rejects
    these before the server sees them); this closes the in-process robustness
    gap after the list-only then list/tuple/range allow-lists each missed a
    silent-budget sequence kind.
    """
    fv = fnp.asarray(np.arange(500, dtype=np.float64))
    # array.array of int64 ('q') indices -> 500-elem gather, dtype_rate(f64)=2.
    ia = array.array("q", range(500))
    assert billed(lambda: fv[ia]) == 4 * 500 * 2
    # memoryview over an int64 buffer -> same 500-elem gather. (np.ndarray
    # satisfies the buffer protocol at runtime; the memoryview() stub only
    # declares ReadableBuffer, hence the arg-type ignore.)
    mv = memoryview(np.arange(500))  # type: ignore[arg-type]
    assert billed(lambda: fv[mv]) == 4 * 500 * 2

    # A numpy integer SCALAR stays BASIC (a view): np.asarray would make it a
    # 0-d int array (kind 'i'), but the fast path catches np.integer first so
    # it is NOT misbilled as an advanced 0-d-array gather.
    npint = np.int64(7)
    assert billed(lambda: fv[npint]) == 0

    # Fall-through parts numpy cannot coerce to a b/i/u index array: a float
    # sequence (kind 'f'), a ragged sequence (np.asarray raises, caught), and a
    # generator (kind 'O', NOT consumed by the classification asarray) all fall
    # through to super().__getitem__ -> numpy raises its own error, unbilled.
    with pytest.raises(IndexError):
        fv[[1.5, 2.5]]
    with pytest.raises(ValueError):
        fv[[[0, 1], [2, 3, 4]]]

    def _gen():
        yield 0
        yield 1
        yield 2

    with pytest.raises(IndexError):
        fv[_gen()]


# ---------------------------------------------------------------------------
# Task 13 (final-review fix, IMPORTANT): choose/block/bmat activate their
# dtype rate + complex factor. Their deduct_after() used to declare
# dtypes=(), which resolves to the dtype-neutral rate 1.0 / complex factor
# 1.0 regardless of the registry's declared complex_factor=2.0 -- silently
# discounting every float64 or complex call. They now read the promoted
# dtype off the actual result via set_dtypes(), matching take's (choose) and
# concatenate's (block/bmat) sibling formulas exactly at every dtype.
# ---------------------------------------------------------------------------


def test_choose_block_bmat_bill_wide_dtypes():
    idx = fnp.asarray(np.zeros(100, dtype=np.int64))
    c_f32 = [fnp.asarray(np.ones(100, dtype=np.float32))] * 2
    c_f64 = [fnp.asarray(np.ones(100, dtype=np.float64))] * 2
    c_c128 = [fnp.asarray(np.ones(100, dtype=np.complex128))] * 2
    # choose: 4*numel(output) at weight 4.0, matching take exactly at every
    # dtype (100-elem gather).
    assert billed(lambda: fnp.choose(idx, c_f32)) == 4 * 100
    assert billed(lambda: fnp.choose(idx, c_f64)) == 4 * 100 * 2
    assert billed(lambda: fnp.choose(idx, c_c128)) == 4 * 100 * 2 * 2

    a_f32 = fnp.asarray(np.ones((10, 10), dtype=np.float32))
    b_f32 = fnp.asarray(np.ones((10, 10), dtype=np.float32))
    a_f64 = fnp.asarray(np.ones((10, 10), dtype=np.float64))
    b_f64 = fnp.asarray(np.ones((10, 10), dtype=np.float64))
    a_c128 = fnp.asarray(np.ones((10, 10), dtype=np.complex128))
    b_c128 = fnp.asarray(np.ones((10, 10), dtype=np.complex128))
    # block/bmat: numel(output) at weight 1.0, matching concatenate exactly
    # at every dtype (200-elem assembly).
    assert billed(lambda: fnp.block([[a_f32, b_f32]])) == 200
    assert billed(lambda: fnp.block([[a_f64, b_f64]])) == 400
    assert billed(lambda: fnp.block([[a_c128, b_c128]])) == 800
    assert billed(lambda: fnp.bmat([[a_f32, b_f32]])) == 200
    assert billed(lambda: fnp.bmat([[a_f64, b_f64]])) == 400
    assert billed(lambda: fnp.bmat([[a_c128, b_c128]])) == 800


# ---------------------------------------------------------------------------
# Task 14 (whole-branch review fix F1a, IMPORTANT): FlopscopeArray METHOD
# forms of .copy/.ravel/.reshape/.choose/.flatten bill identically to their
# fnp.* FUNCTION forms. This branch repriced copy/ravel/reshape/choose as
# functions (0 -> numel@w1, choose 0 -> 4x numel) but left the ndarray
# METHODS unoverridden -- they fell through to numpy's raw C implementation
# and billed 0, a fresh regression diverging from fnp.copy(x) == x.copy()
# parity. flatten has no fnp.flatten function at all (numpy.ndarray.flatten
# is always a real copy, unlike .ravel() which may return a view); it now
# bills once under the "copy" op name rather than composing copy-then-
# reshape, which would double-bill the same data movement.
# ---------------------------------------------------------------------------


def test_ndarray_method_forms_match_function_forms():
    """x.copy()/.ravel()/.reshape()/.choose() must bill exactly like their
    fnp.* function counterparts, at production rates. All were 0 before this
    fix (the method fell through to numpy's unbilled C implementation)."""
    x = fnp.asarray(np.ones(1_000_000, dtype=np.float64))
    assert billed(lambda: x.copy()) == 2_000_000
    assert billed(lambda: fnp.copy(x)) == 2_000_000
    assert billed(lambda: x.ravel()) == 2_000_000
    assert billed(lambda: fnp.ravel(x)) == 2_000_000
    # .reshape() must accept BOTH the varargs form (x.reshape(1000, 1000))
    # and the tuple form (x.reshape((1000, 1000))) -- unlike fnp.reshape /
    # np.reshape, whose 2nd positional parameter is `order`, not another
    # shape dimension.
    assert billed(lambda: x.reshape(1000, 1000)) == 2_000_000
    assert billed(lambda: x.reshape((1000, 1000))) == 2_000_000
    assert billed(lambda: fnp.reshape(x, (1000, 1000))) == 2_000_000

    # choose's index array selects among `choices`; billing is dtype-aware
    # off the promoted CHOICES dtype (read from the result), not off the
    # (integer) index array -- a 1000-element float64 choose bills
    # 4 (access-tier weight) * 2.0 (float64 rate) * 1000 = 8000.
    idx = fnp.asarray(np.arange(1000) % 2)
    choices = [
        fnp.asarray(np.zeros(1000, dtype=np.float64)),
        fnp.asarray(np.ones(1000, dtype=np.float64)),
    ]
    assert billed(lambda: idx.choose(choices)) == 8000
    assert billed(lambda: fnp.choose(idx, choices)) == 8000


def test_copy_method_and_function_are_self_consistent_every_dtype():
    """fnp.copy(x) == x.copy() at every dtype, not just float64 -- the method
    now forwards to the same fnp.copy backend, so this must hold generally."""
    for dt in (np.float32, np.float64, np.complex64, np.complex128):
        x = fnp.asarray(np.arange(37, dtype=dt) + 1)
        cost_fn = billed(lambda x=x: fnp.copy(x))
        cost_method = billed(lambda x=x: x.copy())
        assert cost_fn == cost_method
        assert cost_fn > 0


def test_ndarray_flatten_bills_once_like_copy_not_double():
    """.flatten() has no fnp.flatten function to delegate to (numpy.flatten
    is always a real copy). It must bill exactly once at numel(input) --
    the same price as .copy() -- not twice (e.g. a naive copy-then-reshape
    composition)."""
    x = fnp.asarray(np.ones(1_000_000, dtype=np.float64))
    assert billed(lambda: x.flatten()) == 2_000_000
    assert billed(lambda: x.flatten()) == billed(lambda: x.copy())

    # And it must actually be a real copy (OWNDATA), not a view -- matching
    # numpy.ndarray.flatten's documented contract.
    with flops.BudgetContext(flop_budget=10**18, quiet=True):
        two_d = fnp.asarray(np.ones((1000, 1000), dtype=np.float64))
        flat = two_d.flatten()
    assert flat.flags["OWNDATA"]
    assert flat.base is None
    assert flat.shape == (1_000_000,)


# ---------------------------------------------------------------------------
# Task 14b (F1a follow-up regression): outer/vdot flatten multi-D operands
# through a PRIVATE internal ravel whose result only feeds the einsum cost
# resolver. Task 14 (adding FlopscopeArray.ravel) made a bare a.ravel() there
# bill numel@w1, over-charging the operand's size on top of the real
# contraction. The internal flatten is now subclass-stripped so it stays a
# free numpy view. Production-rate pins for the exact honest cost.
# ---------------------------------------------------------------------------


def test_vdot_matrix_operands_bill_core_not_ravel_surcharge():
    """vdot of two 500x500 f64 matrices (Frobenius inner product) bills only
    the flattened i,i-> contraction (2*250000-1 = 499999, x2.0 f64 = 999998),
    NOT 999998 + two 500000 ravel surcharges (the pre-fix 1999998)."""
    a = fnp.asarray(np.ones((500, 500), dtype=np.float64))
    b = fnp.asarray(np.ones((500, 500), dtype=np.float64))
    assert billed(lambda: fnp.vdot(a, b)) == (2 * 250_000 - 1) * 2  # 999998


def test_outer_2d_operand_bills_same_as_1d_twin():
    """outer(2x3, 4) flattens the 2x3 operand internally; it must bill the
    IDENTICAL amount as outer(6, 4) -- both produce a 6x4 output, so the bill
    depends only on the output, never on the operand's shape. f64: 6*4*2.0=48
    (pre-fix outer(2x3,4) was 60: 48 + a 12-FLOP ravel of the 6-elem operand)."""
    m2x3 = fnp.asarray(np.ones((2, 3), dtype=np.float64))
    v6 = fnp.asarray(np.ones(6, dtype=np.float64))
    v4 = fnp.asarray(np.ones(4, dtype=np.float64))
    assert billed(lambda: fnp.outer(m2x3, v4)) == 6 * 4 * 2  # 48
    assert billed(lambda: fnp.outer(v6, v4)) == 6 * 4 * 2  # 48 (identical)
    assert billed(lambda: fnp.outer(m2x3, v4)) == billed(lambda: fnp.outer(v6, v4))


# ---------------------------------------------------------------------------
# Issue #192 item 4: ldexp's exponent operand is a control parameter, not an
# arithmetic one. numpy's fi->f / el->e loops keep the mantissa's precision
# whatever the exponent's width, so an int32-or-wider exponent must not lift
# the call to the float64 rate. Production-rate pins for the repriced cells.
# ---------------------------------------------------------------------------


def test_ldexp_exponent_width_does_not_lift_the_production_rate():
    """A float32 mantissa bills the float32 rate at every exponent width.

    Pre-fix, int32/int64/uint32 exponents billed 2x this (the float64 rate),
    for the same loop producing the same float32 values.
    """
    load_weights()
    weight = get_weight("ldexp")
    mantissas = np.ones(N, dtype=np.float32)
    expected = int(N * get_dtype_rate("float32") * weight)

    for exponent_dtype in ("int8", "int16", "int32", "int64", "uint16", "uint32"):
        exponents = np.ones(N, dtype=exponent_dtype)
        assert (
            billed(
                lambda e=exponents: fnp.ldexp(fnp.asarray(mantissas), fnp.asarray(e))
            )
            == expected
        ), f"float32 mantissa repriced by a {exponent_dtype} exponent"

    # A float64 mantissa still bills the float64 rate -- unchanged by the fix.
    wide = np.ones(N, dtype=np.float64)
    assert billed(
        lambda: fnp.ldexp(fnp.asarray(wide), fnp.asarray(np.ones(N, "int32")))
    ) == int(N * get_dtype_rate("float64") * weight)
