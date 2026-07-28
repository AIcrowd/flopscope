"""Index-form accounting for ufunc.at.

Ground truth: ``ufunc.at`` applies the ufunc once per selected cell and does
NOT deduplicate repeated indices, so the application count is exactly the size
of the indexing result. We assert against that oracle rather than against magic
numbers, so these tests keep their value as the cost model evolves.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp
from flopscope._pointwise import _canonical_index, _ufunc_at_touched_cells


def oracle(shape, indices) -> int:
    """True ufunc.at application count = size of the indexing result."""
    return int(np.empty(shape, np.uint8)[indices].size)


# (shape, indices, label)
INDEX_FORMS = [
    ((6,), 3, "int scalar"),
    ((6,), -1, "negative int"),
    ((6,), np.array([0, 1, 2]), "int ndarray"),
    ((6,), [0, 0, 0, 0], "repeated list"),
    ((6,), np.zeros(1000, np.intp), "heavy repeat"),
    ((6,), range(3), "range"),
    (
        (6,),
        # memoryview() over an ndarray is runtime-supported (ndarray implements
        # the buffer protocol); numpy's stubs don't declare __buffer__, so the
        # stub is narrower than the implementation here.
        memoryview(np.array([1, 2, 3], np.intp)),  # pyright: ignore[reportArgumentType]
        "memoryview",
    ),
    ((6,), np.array([], np.intp), "empty ndarray"),
    ((6,), [], "empty list"),
    ((6,), slice(None), "full slice"),
    ((6,), slice(1, 5, 2), "stepped slice"),
    ((6,), slice(None, None, -1), "negative step"),
    ((6,), slice(4, 2), "empty slice"),
    ((6,), True, "bool scalar True"),
    ((6,), False, "bool scalar False"),
    ((6,), np.bool_(True), "np.bool_ True"),
    ((6,), np.array(True), "0-d bool array"),
    ((6,), np.array(2), "0-d int array"),
    ((6,), np.array([True, False, True, True, False, True]), "1-d mask"),
    ((6,), [True, False, True, True, False, True], "list of bools"),
    ((6,), Ellipsis, "bare Ellipsis"),
    ((4, 5), (slice(None), slice(None)), "two full slices"),
    ((4, 5), (slice(0, 3), slice(1, 4)), "two partial slices"),
    ((4, 5), (np.array([0, 1]), np.array([2, 3])), "two int arrays"),
    ((4, 5), (np.array([0, 1]), slice(None)), "advanced + slice"),
    ((4, 5), (np.array([0, 1]), 2), "advanced + int"),
    ((4, 5), (np.array([0, 1]), Ellipsis), "advanced + Ellipsis"),
    ((4, 5), (Ellipsis, np.array([0, 1])), "Ellipsis + advanced"),
    ((4, 5), np.ones((4, 5), bool), "2-d full mask"),
    ((4, 5), np.zeros((4, 5), bool), "2-d empty mask"),
    ((4, 5), (None, 0), "newaxis leading"),
    ((4, 5), (0, None), "newaxis trailing"),
    ((2, 3, 4), (np.array([0, 1]), Ellipsis, np.array([0, 1])), "adv-Ellipsis-adv"),
    ((2, 3, 4), np.ones((2, 3), bool), "k-d partial mask"),
    ((1, 4096), (np.zeros(1000, np.intp), Ellipsis), "repeat + Ellipsis"),
    ((1, 4096), (np.zeros(1000, np.intp), slice(None)), "repeat + slice"),
]


@pytest.mark.parametrize(
    "shape,indices,label",
    INDEX_FORMS,
    ids=[lbl for _, _, lbl in INDEX_FORMS],
)
def test_touched_cells_matches_oracle(shape, indices, label):
    canon = _canonical_index(indices)
    expected = max(oracle(shape, indices), 1)
    assert _ufunc_at_touched_cells(np.empty(shape, np.float32), canon) == expected


@pytest.mark.parametrize(
    "shape,indices,label",
    INDEX_FORMS,
    ids=[lbl for _, _, lbl in INDEX_FORMS],
)
def test_canonical_index_preserves_write(shape, indices, label):
    """The canonical index must write exactly what the raw index writes."""
    raw = np.zeros(shape, np.float64)
    canon_target = np.zeros(shape, np.float64)
    np.add.at(raw, indices, 1.0)
    # slices, memoryviews, etc. are runtime-supported index forms that
    # numpy's stubs type more narrowly than the implementation accepts.
    canon = _canonical_index(indices)
    np.add.at(canon_target, canon, 1.0)  # pyright: ignore[reportArgumentType]
    np.testing.assert_array_equal(raw, canon_target)


def test_bool_is_not_treated_as_int():
    """Python bool subclasses int; numpy treats it as a 0-d mask adding an axis."""
    assert (
        _ufunc_at_touched_cells(np.empty((6,), np.float32), _canonical_index(True)) == 6
    )
    assert _ufunc_at_touched_cells(np.empty((6,), np.float32), _canonical_index(1)) == 1


def test_non_integer_array_index_raises_indexerror():
    with pytest.raises(IndexError):
        _canonical_index(np.array([1.5, 2.5]))


def test_canonical_index_does_not_freeze_caller_array():
    """We must never make a participant's own array read-only."""
    caller = np.array([0, 1, 2], np.intp)

    class Holder:
        def __array__(self, dtype=None, copy=None):
            return caller

    _canonical_index(Holder())
    assert caller.flags.writeable is True


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        before = ctx.flops_used
        fn()
        return ctx.flops_used - before


def test_list_index_bills_every_application():
    """A plain python list index must bill per application, not per destination cell."""
    dst = fnp.asarray(np.zeros(6, np.float64))
    idx = [0] * 5000
    assert billed(lambda: np.add.at(dst, idx, 1.0)) == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(5000, np.float64)), 1.0)
    )


def test_stateful_array_index_bills_what_it_writes():
    """An index resolved twice could bill one value and write another."""
    n = 200_000

    class Shifting:
        def __init__(self):
            self.calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            return np.zeros(1 if self.calls == 1 else n, np.intp)

    dst = fnp.asarray(np.zeros(4, np.float64))
    probe = Shifting()
    cost = billed(lambda: np.add.at(dst, probe, 1.0))
    written = float(np.asarray(dst)[0])
    assert probe.calls == 1, "index must be resolved exactly once"
    assert cost == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(int(written), np.float64)), 1.0)
    )


def test_stateful_slice_index_bills_what_it_writes():
    n = 100_000

    class Shifting:
        def __init__(self):
            self.calls = 0

        def __index__(self):
            self.calls += 1
            return 1 if self.calls == 1 else n

    dst = fnp.asarray(np.zeros(n, np.float64))
    idx = slice(0, Shifting())

    def do_at():
        # a slice stop with a stateful __index__ is runtime-supported; numpy's
        # stubs type slice indices narrower than the implementation accepts.
        np.add.at(dst, idx, 1.0)  # pyright: ignore[reportArgumentType]

    cost = billed(do_at)
    written = int((np.asarray(dst) != 0).sum())
    assert cost == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(written, np.float64)), 1.0)
    )


def test_tuple_index_entries_are_stripped():
    """flopscope-typed tuple entries must not reach numpy's index parser."""
    dst = fnp.asarray(np.zeros((4, 5), np.float64))
    rows = fnp.asarray(np.array([0, 1, 2], np.intp))
    cols = fnp.asarray(np.array([0, 1, 2], np.intp))
    assert billed(lambda: np.add.at(dst, (rows, cols), 1.0)) == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(3, np.float64)), 1.0)
    )


def test_at_write_through_alias_voids_symmetry_tag():
    """ufunc.at mutates its target, so the write must be recorded."""
    from flopscope._write_epoch import epoch_of

    z = fnp.zeros((32, 32))
    alias = fnp.asarray(z)
    before = epoch_of(z)
    np.subtract.at(
        alias,
        (np.arange(32), np.zeros(32, np.intp)),
        np.arange(1, 33).astype(z.dtype),
    )
    assert epoch_of(z) != before, "the write must advance the buffer epoch"


@pytest.mark.parametrize(
    "name,invoke",
    [
        ("mask_indices", lambda cb: fnp.mask_indices(4, cb)),
        ("fromfunction", lambda cb: fnp.fromfunction(cb, (4, 4))),
        (
            "apply_along_axis",
            lambda cb: fnp.apply_along_axis(
                cb, 0, fnp.asarray(np.zeros((4, 4), np.float32))
            ),
        ),
    ],
)
def test_callback_wall_time_books_to_residual(name, invoke):
    """Participant callback time must not land in the free overhead bucket."""
    flopscope.configure(callback_warnings=False)
    sleep_s = 0.20

    def callback(*args, **kwargs):
        time.sleep(sleep_s)
        if name == "mask_indices":
            return np.zeros((4, 4), bool)
        if name == "apply_along_axis":
            return np.float32(0.0)
        return (
            np.zeros(np.shape(args[0]), np.float32)
            if args
            else np.zeros((4, 4), np.float32)
        )

    try:
        with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
            invoke(callback)
    finally:
        flopscope.configure(callback_warnings=True)
    summary = ctx.summary_dict()
    residual = float(summary.get("residual_wall_time_s") or 0.0)
    assert residual >= 0.8 * sleep_s, (
        f"{name}: callback slept {sleep_s}s but only {residual:.3f}s booked to residual"
    )


def test_values_array_protocol_resize_of_index_does_not_shrink_the_bill():
    """The ``vals`` operand's ``__array__`` runs while resolving the billing
    dtype, and ``_canon_entry`` hands an already-canonicalized integer index
    array back BY IDENTITY (not a copy). If the touched-cell count were taken
    before that resolution, participant code reachable from ``__array__``
    could enlarge the index in place (``ndarray.resize(refcheck=False)``)
    after the count but before the write, and the bill would reflect the
    index's smaller pre-resize size instead of what ``ufunc.at`` actually
    applies against.
    """
    idx = np.zeros(1, np.intp)

    class Vals:
        def __array__(self, dtype=None, copy=None):
            idx.resize(1_000_000, refcheck=False)
            return np.ones(1, np.float64)

    dst = fnp.asarray(np.zeros(4, np.float64))
    cost = billed(lambda: np.add.at(dst, idx, Vals()))
    written = float(np.asarray(dst)[0])
    assert written == 1_000_000.0, "sanity: the resized index must actually be applied"
    assert cost == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(int(written), np.float64)), 1.0)
    )


def test_at_composed_matmul_is_not_cheaper_than_matmul():
    """A matmul assembled out of ufunc.at calls must not undercut fnp.matmul."""
    k = 32
    X = fnp.asarray(np.random.randn(k, k).astype(np.float32))
    W = fnp.asarray(np.random.randn(k, k).astype(np.float32))
    honest = billed(lambda: fnp.matmul(X, W))
    one = np.array([0], np.intp)

    def at_route():
        D = fnp.asarray(np.zeros((1, k, k, k), np.float32))
        np.add.at(D, one, W[None, None, :, :])
        np.multiply.at(D, one, X[None, :, :, None])
        V = D[0]
        m = k
        while m > 1:
            h = m // 2
            np.add.at(V[:, :h, :][None], one, V[:, h : 2 * h, :][None])
            V = V[:, :h, :]
            m = h

    assert billed(at_route) >= honest
