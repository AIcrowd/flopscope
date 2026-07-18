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
import pytest
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
    # np.triu as mask_func: plain-numpy callable, NOT billed -> isolates mask_indices' own 2k
    assert billed(lambda: fnp.mask_indices(8, np.triu)) == 2 * 36
    # fnp.triu as mask_func: see test_mask_indices_fnp_mask_func_hits_preexisting_nonzero_bug
    # below -- it does NOT reach a `> 2*36` bill (a separate, pre-existing crash).
    assert billed(lambda: fnp.broadcast_shapes((4, 6), (6,))) == 3


def test_mask_indices_fnp_mask_func_hits_preexisting_nonzero_bug():
    """An fnp-wrapped mask_func (e.g. fnp.triu) does NOT reach a `> 2*k`
    combined bill -- it raises instead. The ledger's IMPLEMENTATION CAVEAT
    ("confirm the mask_func callable is restricted or its own fnp ops bill
    separately ... flag before committing (do not fix here)") anticipated an
    fnp callable "billing separately"; this pins the ACTUAL observed
    behavior instead, which is worse: numpy's own ``mask_indices`` body ends
    with a bare top-level ``nonzero(a != 0)`` call, and ``a`` (mask_func's
    FlopscopeArray return value, auto-wrapped by ``wrap_module_returns``)
    hits it. ``nonzero`` is NOT in ``FlopscopeArray._get_array_function_
    dispatch``'s map (unlike the ``.nonzero()`` METHOD, which IS overridden,
    and unlike ``fnp.nonzero`` itself, which exists) -- so this always
    raises, independent of Task 8's changes here (reproduced against the
    pre-Task-8 HEAD, commit c76ec237f, via `git stash`). Flagged per the
    ledger instruction, not fixed -- out of this task's "index generators
    pricing" scope; the underlying gap is in flopscope._ndarray's NEP-18
    dispatch map, not in mask_indices' own cost formula.
    """
    with pytest.raises(RuntimeError, match="nonzero"):
        billed(lambda: fnp.mask_indices(8, fnp.triu))


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
