"""Index-form accounting for ufunc.at.

Ground truth: ``ufunc.at`` applies the ufunc once per selected cell and does
NOT deduplicate repeated indices, so the application count is exactly the size
of the indexing result. We assert against that oracle rather than against magic
numbers, so these tests keep their value as the cost model evolves.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp
from flopscope._pointwise import _canon_entry, _canonical_index, _ufunc_at_touched_cells
from flopscope._weights import load_weights


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


class _LyingDtypeIndex(np.ndarray):
    """An ndarray subclass whose ``.dtype`` PROPERTY claims ``intp`` no
    matter what the real underlying buffer is. numpy's own C-level index
    parser reads the true descriptor regardless of what this property
    reports, so classifying an index off ``.dtype`` here (rather than the
    real buffer) would accept -- and silently truncate -- a float-backed
    index real numpy rejects outright.
    """

    @property
    def dtype(self):
        return np.dtype(np.intp)


def test_lying_dtype_float_index_matches_raw_numpy_rejection():
    """A ``.dtype``-lying subclass over a FLOAT buffer must be rejected with
    the exact exception type (and, here, class of message) raw numpy raises
    for the equivalent honest float index -- not silently truncated into an
    accepted integer index.
    """
    honest_float_index = np.array([0.0, 1.0, 2.0])
    lying = honest_float_index.view(_LyingDtypeIndex)
    assert lying.dtype == np.dtype(np.intp), "sanity: the property really does lie"

    dst = np.zeros(5)
    with pytest.raises(IndexError) as raw_exc_info:
        np.add.at(dst, honest_float_index, 1.0)

    with pytest.raises(type(raw_exc_info.value)):
        _canonical_index(lying)


def test_lying_dtype_narrow_int_index_does_not_truncate_values():
    """An integer-backed subclass reporting a NARROWER dtype than its real
    buffer must not have its real (wide) index values truncated by the
    lying property -- that would touch different cells than the caller's
    actual data implies. 300 overflows int8 but is a valid value at the
    real (int64) dtype.
    """

    class _LyingNarrowInt(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(np.int8)

    real_index = np.array([300], dtype=np.int64)
    lying = real_index.view(_LyingNarrowInt)

    canon = _canon_entry(lying)
    assert isinstance(canon, np.ndarray)
    assert canon.dtype == np.dtype(np.int64), (
        "the canonical dtype must be the real buffer's, not the lying int8 claim"
    )
    assert int(canon[0]) == 300, (
        "the real (wide) value must survive, not get truncated via the lie"
    )


def test_canonical_index_does_not_freeze_caller_array():
    """We must never make a participant's own array read-only."""
    caller = np.array([0, 1, 2], np.intp)

    class Holder:
        def __array__(self, dtype=None, copy=None):
            return caller

    _canonical_index(Holder())
    assert caller.flags.writeable is True


@pytest.mark.parametrize("dtype", [np.intp, np.int32, np.uint16])
def test_canon_entry_snapshots_integer_index_array(dtype):
    """An integer ndarray index must canonicalize to an owned, read-only
    copy -- not the caller's own array by identity. A live-identity result
    would let the caller change what the array describes (e.g. via
    ``ndarray.resize(refcheck=False)``) after canonicalization but before
    the count and the write that are supposed to use this frozen form.
    """
    original = np.array([0, 1, 2], dtype)
    result = _canon_entry(original)
    assert isinstance(result, np.ndarray)
    assert result is not original
    assert result.flags.writeable is False
    assert result.dtype == dtype, "the copy must not coerce the index dtype"
    np.testing.assert_array_equal(result, original)


def test_canon_entry_honest_ndarray_subclass_index_still_works():
    """A plain ndarray subclass that does NOT override ``.dtype`` -- i.e.
    reports its real dtype honestly -- must canonicalize exactly like a
    bare ndarray of the same dtype: subclassing itself is legitimate, only
    lying about ``.dtype`` is the problem this module guards against.
    """

    class _HonestIndexSubclass(np.ndarray):
        pass

    original = np.array([0, 1, 2], np.int32).view(_HonestIndexSubclass)
    result = _canon_entry(original)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, np.array([0, 1, 2], np.int32))


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


def test_foreign_ufunc_at_callback_records_successful_write():
    """A foreign ``ufunc.at`` handler can mutate the target before returning."""
    from flopscope._write_epoch import epoch_of

    class MutatingDuck:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def __array__(self, dtype=None, copy=None):
            return np.asarray([0.0], dtype=dtype)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            assert (ufunc, method) == (np.add, "at")
            self.calls += 1
            inputs[0][0] += 7
            return self.result

    raw_result = object()
    raw_target = np.zeros(2)
    raw_duck = MutatingDuck(raw_result)
    expected = np.add.at(raw_target, [0], raw_duck)

    result = object()
    target = fnp.zeros(2)
    before = epoch_of(target)
    duck = MutatingDuck(result)
    with flopscope.BudgetContext(flop_budget=10**9):
        actual = np.add.at(target, [0], duck)

    assert expected is raw_result
    assert actual is result
    assert raw_duck.calls == duck.calls == 1
    np.testing.assert_array_equal(target, raw_target)
    assert epoch_of(target) != before


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
    dtype, and can be reached again later when numpy resolves ``vals`` a
    second time inside its own ``ufunc.at`` execution. ``_canon_entry``
    snapshots an integer index array into an owned, read-only copy taken
    BEFORE either of those calls, so participant code reachable from
    ``__array__`` that resizes the ORIGINAL index object in place
    (``ndarray.resize(refcheck=False)``) -- on every call, as here -- changes
    neither the count nor what actually gets applied: both stay pinned to
    the frozen snapshot taken at canonicalization time, regardless of what
    the participant does to their own array afterward.
    """
    idx = np.zeros(1, np.intp)

    class Vals:
        def __array__(self, dtype=None, copy=None):
            idx.resize(1_000_000, refcheck=False)
            return np.ones(1, np.float64)

    dst = fnp.asarray(np.zeros(4, np.float64))
    cost = billed(lambda: np.add.at(dst, idx, Vals()))
    written = float(np.asarray(dst)[0])
    assert written == 1.0, (
        "the frozen snapshot taken at canonicalization must be applied, "
        "not the index resized afterward"
    )
    assert cost == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(int(written), np.float64)), 1.0)
    )


def test_values_array_protocol_is_resolved_at_most_once():
    """``vals`` (the ``ufunc.at`` ``values`` operand) is resolved through the
    array protocol AT MOST ONCE. flopscope used to read ``vals.__array__``
    once here for billing-dtype resolution and then hand numpy the
    caller's live, unresolved object, which let numpy's own ``ufunc.at``
    implementation independently re-derive ``vals`` a second time --
    opening the same billed-vs-applied gap the frozen index snapshot
    below closes for indices (a ``vals.__array__`` reporting one dtype on
    its first call and a pricier one on numpy's own second call would
    bill the cheap one while numpy actually computed the pricier loop).
    flopscope now resolves ``vals`` exactly once and hands numpy that SAME
    resolved array, so there is no second, independent read left for
    numpy to perform. The resize below (fired on every call ``vals``
    gets, including this single one) still can't touch the index snapshot
    already frozen at canonicalization time.
    """
    n = 500_000
    idx = np.zeros(1, np.intp)

    class Vals:
        def __init__(self):
            self.calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            idx.resize(n, refcheck=False)
            return np.ones(1, np.float64)

    dst = fnp.asarray(np.zeros(4, np.float64))
    vals = Vals()
    cost = billed(lambda: np.add.at(dst, idx, vals))
    written = float(np.asarray(dst)[0])
    assert vals.calls == 1, (
        "flopscope must resolve vals exactly once and hand numpy that same "
        "resolved array -- numpy re-deriving vals on its own would let it "
        "see a different value than what was billed"
    )
    assert written == 1.0, (
        "the frozen snapshot taken at canonicalization must be applied, "
        "not the index resized during vals resolution"
    )
    assert cost == billed(
        lambda: fnp.add(fnp.asarray(np.zeros(int(written), np.float64)), 1.0)
    )


def test_values_dtype_array_protocol_is_resolved_once():
    """The ``vals`` operand's DTYPE, not just its identity, must come from a
    single resolution shared by the billing rate and the real call.
    ``vals.__array__`` returning a cheap dtype on its (only) call and numpy
    then independently re-deriving a pricier one from the caller's live
    object would bill the cheap rate while the loop that actually ran was
    the pricier one -- ``_resolve_at_operand`` materializes ``vals`` once,
    up front, so numpy only ever sees that same resolved array.
    """
    load_weights()
    n = 2_000_000
    calls = [0]

    class Vals:
        def __array__(self, dtype=None, copy=None):
            calls[0] += 1
            cheap = np.ones(n, np.int8)
            expensive = np.ones(n, np.float64)
            return cheap if calls[0] == 1 else expensive

    dst = fnp.asarray(np.ones(n, np.int8))
    cost = billed(lambda: np.multiply.at(dst, np.arange(n), Vals()))

    honest_int8 = billed(
        lambda: np.multiply.at(
            fnp.asarray(np.ones(n, np.int8)), np.arange(n), np.ones(n, np.int8)
        )
    )
    honest_float64 = billed(
        lambda: np.multiply.at(
            fnp.asarray(np.ones(n, np.int8)), np.arange(n), np.ones(n, np.float64)
        )
    )

    assert calls[0] == 1, "sanity: vals must be resolved exactly once"
    assert cost == honest_int8, "the bill must match the single read actually used"
    assert cost < honest_float64, (
        "sanity: float64 is the genuinely pricier loop dtype here"
    )


def test_values_ndarray_subclass_cannot_lie_about_its_own_dtype():
    """A ``vals`` operand that is ALREADY an ``np.ndarray`` (not merely
    ``__array__``-duck-typed) takes a different path than the test above:
    ``_resolve_at_operand`` used to just strip any flopscope wrapper and
    otherwise leave it alone, so reading ``vals.dtype`` for billing ran
    against the caller's own object. An arbitrary OTHER ndarray subclass
    can override ``.dtype`` as a plain Python property -- numpy's own
    ufunc dispatch reads the TRUE underlying descriptor at the C level
    regardless of what that property reports, so a subclass instance can
    report a cheap dtype to whatever reads ``.dtype`` in Python while
    numpy computes at its real, pricier one.
    """
    load_weights()
    n = 2_000_000

    class LiesAboutDtype(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(np.int8)

    dst = fnp.asarray(np.zeros(n, dtype=np.int8))
    real_vals = np.full(n, 1 + 2j, dtype=np.complex128).view(LiesAboutDtype)
    idx = np.arange(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # int8 destination truncates complex vals
        cost = billed(lambda: np.add.at(dst, idx, real_vals))

        honest = billed(
            lambda: np.add.at(
                fnp.asarray(np.zeros(n, dtype=np.int8)),
                idx,
                np.full(n, 1 + 2j, dtype=np.complex128),
            )
        )

    assert cost == honest, (
        "the bill must reflect the TRUE dtype numpy computes at, not the "
        "cheap one the subclass's .dtype property reports"
    )


def test_at_destination_ndarray_subclass_cannot_lie_about_its_own_dtype():
    """Same exposure as the ``vals`` test above, but for ``a`` -- the
    in-place destination -- itself: dispatch to ``.at`` happens as soon as
    ``a`` is flopscope-aware, but nothing stops ``a`` from ALSO being a
    foreign ndarray subclass whose ``.dtype`` property lies (e.g. reached
    via ``np.add.at(foreign_array, idx, fnp_array)`` dispatching off the
    ``values`` operand). ``_counted_ufunc_at`` reads ``a.dtype`` directly
    for the billing rate, so it must be exercised beneath numpy's own
    ufunc-method dispatch (which would otherwise mask the lie the same way
    the C-level execution does).
    """
    from flopscope._pointwise import _counted_ufunc_at

    load_weights()
    n = 2_000_000

    class LiesAboutDtype(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(np.int8)

    idx = np.arange(n)
    lying_dst = np.zeros(n, dtype=np.float64).view(LiesAboutDtype)
    vals = np.ones(n, dtype=np.int8)

    cost = billed(lambda: _counted_ufunc_at(np.subtract, lying_dst, idx, vals))
    honest_float64 = billed(
        lambda: _counted_ufunc_at(np.subtract, np.zeros(n, dtype=np.float64), idx, vals)
    )
    honest_int8 = billed(
        lambda: _counted_ufunc_at(np.subtract, np.zeros(n, dtype=np.int8), idx, vals)
    )

    assert cost == honest_float64, (
        "the bill must reflect the TRUE dtype numpy computes at, not the "
        "cheap one the subclass's .dtype property reports"
    )
    assert cost > honest_int8, "sanity: float64 is the genuinely pricier dtype here"


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
