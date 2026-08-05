"""Cost accounting for ``ufunc.reduceat``.

Ground truth (NumPy's own documented per-segment semantics): for segment
``i``, if ``indices[i] < indices[i+1]`` then
``result[i] = reduce(a[indices[i]:indices[i+1]])`` -- a length-``L`` segment
costs ``L-1`` applications, the same ``n-1`` convention ``.reduce`` uses.
Otherwise ``result[i] = a[indices[i]]``, a plain element copy with no
arithmetic (0 applications). The final segment always runs to the end of the
axis. We assert against an independently-computed oracle (a plain Python loop
over that definition, not flopscope's own vectorized formula) rather than
against magic numbers, so these tests keep their value as the cost model
evolves.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp
from flopscope._weights import load_weights


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        before = ctx.flops_used
        with warnings.catch_warnings():
            # np.subtract.reduce/.reduceat on a FlopscopeArray auto-routes
            # with a UserWarning notice; harmless here, just noise.
            warnings.simplefilter("ignore", UserWarning)
            fn()
        return ctx.flops_used - before


def billed_raising(fn, exc_type) -> int:
    """Like :func:`billed`, but for a call expected to raise ``exc_type``.

    The deduct commits before the wrapped real call runs (flopscope never
    refunds), so the flops delta is still meaningful even though ``fn()``
    itself raises.
    """
    with flopscope.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        before = ctx.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(exc_type):
                fn()
        return ctx.flops_used - before


def application_count_oracle(n_axis: int, indices) -> int:
    """Honest per-lane application count, from NumPy's documented semantics.

    A direct, independent Python loop over the per-segment definition in
    the module docstring above -- deliberately NOT the vectorized
    ``np.maximum(lengths - 1, 0)`` trick flopscope's own cost formula uses,
    so this is a genuine check on that formula rather than a restatement
    of it.
    """
    idx = list(indices)
    k = len(idx)
    if k == 0 or n_axis == 0:
        return 0
    total = 0
    for i in range(k):
        start = idx[i]
        end = idx[i + 1] if i + 1 < k else n_axis
        if start < end:
            total += (end - start) - 1
        # else: a plain copy (indices[i] >= indices[i+1]) -- 0 applications.
    return total


def test_reduceat_work_per_lane_accumulates_applications_as_python_int():
    from flopscope._pointwise import _reduceat_work_per_lane

    n = 2**62
    indices = np.array([0, n - 1, 0, n - 1, 0], dtype=np.int64)

    applications, output_cells = _reduceat_work_per_lane(indices, n)

    expected_applications = 3 * n - 5
    assert type(applications) is int
    assert applications == expected_applications
    assert applications > np.iinfo(np.int64).max
    assert type(output_cells) is int
    assert output_cells == len(indices)


def honest_reduceat_cost(dtype, lanes: int, billed_units_per_lane: int) -> int:
    """Independently price either per-lane work measure used for billing.

    ``lanes`` independent ``subtract.reduce`` calls over
    ``billed_units_per_lane + 1``-length rows bill exactly
    ``lanes * billed_units_per_lane`` units at whatever rate/weight is
    currently configured. Comparing against this independently prices either
    reduceat work measure without hardcoding any rate or weight number, so it
    survives cost-model-wide rate changes.
    """
    length = billed_units_per_lane + 1
    arr = fnp.asarray(np.full((lanes, length), 2, dtype=dtype))
    return billed(lambda: np.subtract.reduce(arr, axis=-1))


def reduceat_actual_cost(shape, axis, indices, dtype) -> int:
    a = fnp.asarray(np.full(shape, 2, dtype=dtype))
    return billed(lambda: np.subtract.reduceat(a, indices, axis=axis))


# (shape, axis, indices, label)
CASES = [
    ((20,), 0, [0, 5, 10, 15], "monotonic partition"),
    ((30,), 0, [0], "single index, whole axis"),
    ((10,), 0, [4], "single index, not at start"),
    ((10,), 0, [5, 2, 8], "overlapping / non-monotonic"),
    ((10,), 0, [3, 3, 7], "indices[i] == indices[i+1]"),
    ((10,), 0, list(range(10)), "all singleton segments"),
    ((1, 5), 0, [0, 0, 0, 0], "multi-lane copy-only output floor"),
    ((10,), 0, [], "empty index list"),
    ((4, 5), 0, [0, 2], "2-D, axis 0"),
    ((4, 5), 1, [0, 3], "2-D, axis 1"),
    ((4, 5), -1, [0, 3], "2-D, axis -1"),
    ((2, 3, 4), 0, [0, 1], "3-D, axis 0"),
    ((2, 3, 4), 1, [0, 2], "3-D, axis 1"),
    ((2, 3, 4), -1, [0, 1, 3], "3-D, axis -1"),
]


@pytest.mark.parametrize("shape,axis,indices,label", CASES, ids=[c[3] for c in CASES])
def test_reduceat_cost_matches_application_count_oracle(shape, axis, indices, label):
    dtype = np.int64
    norm_axis = axis % len(shape)
    n = shape[norm_axis]
    lanes = 1
    for i, d in enumerate(shape):
        if i != norm_axis:
            lanes *= d
    billed_units = max(application_count_oracle(n, indices), len(indices))
    actual = reduceat_actual_cost(shape, axis, indices, dtype)
    expected = honest_reduceat_cost(dtype, lanes, billed_units)
    assert actual == expected


def test_reduceat_output_floor_is_deducted_before_numpy(monkeypatch):
    a = fnp.asarray(np.ones(1, dtype=np.float64))
    old_global_minimum = honest_reduceat_cost(
        np.float64, lanes=1, billed_units_per_lane=0
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("numpy must not run after budget preflight fails")

    monkeypatch.setattr(flopscope._pointwise, "_call_numpy", fail_if_called)
    with flopscope.BudgetContext(flop_budget=old_global_minimum, quiet=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(flopscope.BudgetExhaustedError):
                np.subtract.reduceat(a, [0, 0])


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((37,), 0),
        ((6, 4), 0),
        ((6, 4), 1),
        ((6, 4), -1),
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), -1),
    ],
    ids=[
        "1-D",
        "2-D axis0",
        "2-D axis1",
        "2-D axis-1",
        "3-D axis0",
        "3-D axis1",
        "3-D axis-1",
    ],
)
def test_reduceat_whole_axis_single_segment_matches_reduce(shape, axis):
    """``<ufunc>.reduceat(a, [0])`` over the whole axis performs exactly the
    same work as ``<ufunc>.reduce(a, axis=axis)`` -- one full-axis
    reduction per lane -- so it must bill exactly the same.
    """
    a = fnp.asarray(np.full(shape, 2, dtype=np.int64))
    reduceat_cost = billed(lambda: np.subtract.reduceat(a, [0], axis=axis))
    reduce_cost = billed(lambda: np.subtract.reduce(a, axis=axis))
    assert reduceat_cost == reduce_cost


def test_indices_array_protocol_second_read_does_not_escape_the_snapshot():
    """A more direct TOCTOU probe than the ``a``-mutates-``idx`` case
    above: here ``indices`` ITSELF is an arbitrary array-like exposing
    ``__array__`` (not a plain ndarray), and that method returns a
    DIFFERENT, larger array if it is ever invoked a second time.

    Flopscope's own converter (``_np.array(indices, dtype=intp,
    copy=True)``) is documented as the ONLY read of ``indices`` -- both the
    cost math and the real ``ufunc.reduceat`` call must run against that
    one snapshot. An implementation that instead costs off one read but
    hands the ORIGINAL ``indices`` object (rather than the snapshot) to the
    real call -- letting numpy re-invoke ``__array__`` itself from inside
    ``ufunc.reduceat`` -- would silently execute against a second,
    different array than the one it billed for.

    The first call returns a single index spanning the whole axis (the
    maximally expensive shape for this ``n``); a second call, if it ever
    happened, would return ``n`` singleton segments (a larger array, but
    near-zero cost). Both the read count and the bill must reflect only
    the first (and, per the single-read discipline, only) call.
    """
    n = 4000
    calls = 0

    class Indices:
        def __array__(self, dtype=None, copy=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.array([0], dtype=np.intp)  # whole-axis single segment
            return np.arange(n, dtype=np.intp)  # n singletons: a larger, cheap array

    a = fnp.asarray(np.full(n, 2, dtype=np.int64))
    cost = billed(lambda: np.subtract.reduceat(a, Indices()))

    honest = billed(
        lambda: np.subtract.reduce(fnp.asarray(np.full(n, 2, dtype=np.int64)))
    )

    assert calls == 1, "sanity: array-like indices must be read exactly once"
    assert cost == honest, "the bill must reflect the single read actually executed"


def test_reduceat_axis_none_on_1d_matches_axis_zero():
    """Real ``ufunc.reduceat`` accepts ``axis=None`` on a 1-D array and
    treats it as axis 0 -- it performs the full reduction, not a no-op.
    Use a size where a dropped (near-floor) bill would be unmistakable
    against the honest one.
    """
    n = 1_000_000
    dtype = np.int64
    none_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(n, 2, dtype=dtype)),
            [0],
            axis=None,  # type: ignore[arg-type]
        )
    )
    zero_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(n, 2, dtype=dtype)), [0], axis=0
        )
    )
    floor_cost = honest_reduceat_cost(dtype, lanes=1, billed_units_per_lane=0)
    assert none_cost == zero_cost
    assert none_cost > floor_cost


def test_reduceat_axis_none_on_2d_matches_numpy_rejection():
    """``axis=None`` is only valid for reduceat on a 1-D array; real numpy
    rejects it on higher-rank input (``reduceat does not allow multiple
    axes``). flopscope must not invent a cost for an axis the real call is
    about to refuse -- it must raise the same error numpy raises directly,
    not silently bill and succeed.
    """
    a_raw = np.full((4, 5), 2, dtype=np.int64)
    with pytest.raises(ValueError):
        np.add.reduceat(a_raw, [0], axis=None)  # type: ignore[arg-type]  # confirms raw numpy's own behavior

    billed_raising(
        lambda: np.subtract.reduceat(
            fnp.asarray(a_raw.copy()),
            [0],
            axis=None,  # type: ignore[arg-type]
        ),
        ValueError,
    )


def test_reduceat_out_of_range_indices_do_not_inflate_the_bill():
    """Real ``ufunc.reduceat`` requires every index in ``[0, n)`` -- it
    rejects negative indices outright (no Python-style wraparound) as well
    as indices ``>= n``. flopscope cannot guess a segment length for an
    index the real call is about to refuse, so an out-of-range index, in
    EITHER direction, must floor to the minimum bill rather than inflate
    it from the (potentially huge) phantom segment length the raw index
    arithmetic would otherwise produce.
    """
    n = 10
    dtype = np.int64
    a_raw = np.full(n, 2, dtype=dtype)

    # Confirm numpy's own accept/reject boundary before pinning flopscope
    # to it.
    with pytest.raises(IndexError):
        np.subtract.reduceat(a_raw, [-(10**9)])
    with pytest.raises(IndexError):
        np.subtract.reduceat(a_raw, [n])

    floor_cost = honest_reduceat_cost(dtype, lanes=1, billed_units_per_lane=0)

    huge_negative_cost = billed_raising(
        lambda: np.subtract.reduceat(fnp.asarray(a_raw.copy()), [-(10**9)]),
        IndexError,
    )
    positive_oob_cost = billed_raising(
        lambda: np.subtract.reduceat(fnp.asarray(a_raw.copy()), [n]),
        IndexError,
    )

    assert huge_negative_cost == floor_cost
    assert positive_oob_cost == floor_cost


def test_reduceat_axis_as_length_one_tuple_matches_bare_axis():
    """Real ``ufunc.reduceat`` accepts a length-1 ``axis`` tuple and
    unwraps it to the contained integer (``axis=(0,)`` behaves exactly
    like ``axis=0``); flopscope must bill it the same way rather than
    crashing on the tuple/int comparison or falling through to the floor.
    """
    shape = (1000,)
    dtype = np.int64
    tuple_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(shape, 2, dtype=dtype)),
            [0, 3],
            axis=(0,),  # type: ignore[arg-type]
        )
    )
    bare_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(shape, 2, dtype=dtype)), [0, 3], axis=0
        )
    )
    floor_cost = honest_reduceat_cost(dtype, lanes=1, billed_units_per_lane=0)
    assert tuple_cost == bare_cost
    assert tuple_cost > floor_cost


def test_reduceat_negative_axis_on_multidim_bills_correctly():
    """A negative ``axis`` (Python-style, counting from the end) on a
    multi-dimensional array must resolve to the same axis -- and bill the
    same -- as its positive equivalent.
    """
    shape = (6, 4)
    dtype = np.int64
    neg_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(shape, 2, dtype=dtype)), [0, 2], axis=-1
        )
    )
    pos_cost = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full(shape, 2, dtype=dtype)), [0, 2], axis=1
        )
    )
    floor_cost = honest_reduceat_cost(dtype, lanes=1, billed_units_per_lane=0)
    assert neg_cost == pos_cost
    assert neg_cost > floor_cost


def test_axis_array_protocol_second_read_does_not_escape_the_snapshot():
    """A more direct TOCTOU probe than the ``indices`` case above: here
    ``axis`` ITSELF is an arbitrary object exposing ``__index__``, and that
    method returns a DIFFERENT axis if it is ever invoked a second time.

    ``_resolve_reduceat_axis`` is documented as the ONLY read of ``axis`` --
    both the cost math and the real ``ufunc.reduceat`` call must run against
    that one resolved axis. An implementation that costs off one read but
    hands the ORIGINAL ``axis`` object (rather than the resolved int) to the
    real call -- letting numpy re-invoke ``__index__`` itself from inside
    ``ufunc.reduceat`` -- would silently execute along a second, different
    axis than the one it billed for.

    The first call returns axis 1 (length 2, near-zero cost for this shape);
    a second call, if it ever happened, would return axis 0 (length N, the
    maximally expensive axis for this shape). Both the read count and the
    bill must reflect only the first (and, per the single-read discipline,
    only) call -- and the array must actually be reduced along the axis
    that was billed for, not the other one.
    """
    N = 200_000
    calls = 0

    class Axis:
        def __index__(self):
            nonlocal calls
            calls += 1
            return 1 if calls == 1 else 0

    a = fnp.asarray(np.full((N, 2), 2, dtype=np.int64))
    cost = billed(lambda: np.subtract.reduceat(a, [1], axis=Axis()))

    honest = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full((N, 2), 2, dtype=np.int64)), [1], axis=1
        )
    )
    expensive = billed(
        lambda: np.subtract.reduceat(
            fnp.asarray(np.full((N, 2), 2, dtype=np.int64)), [1], axis=0
        )
    )

    assert calls == 1, "sanity: axis must be resolved exactly once"
    assert cost == honest, "the bill must reflect the single read actually executed"
    assert cost < expensive, (
        "sanity: axis 0 is the genuinely expensive axis for this shape"
    )


def test_dtype_kwarg_array_protocol_is_resolved_once():
    """An explicit ``dtype=`` naming a dtype-like object (one exposing a
    ``.dtype`` property, which ``np.dtype()`` honours) must be resolved
    EXACTLY once, with that SAME resolved dtype used both for the billing
    rate and for the real ``ufunc.reduceat`` call. Reading it once for
    billing and then handing the original object to numpy -- which
    resolves it again independently -- would let a property that reports a
    cheap dtype on an early call and a pricier one later bill at the cheap
    rate while numpy actually runs the pricier loop.

    Uses production dtype rates (``load_weights()``) rather than this
    module's unit-rate ``billed()`` default -- float32 and float64 bill
    identically under unit rates, which would hide exactly the divergence
    this test exists to catch.
    """
    load_weights()
    N = 2_000_000
    calls = 0

    class StatefulDtype:
        @property
        def dtype(self):
            nonlocal calls
            calls += 1
            return np.dtype(np.float32) if calls == 1 else np.dtype(np.float64)

    a = fnp.asarray(np.full((N,), 2, dtype=np.int32))
    cost = billed(lambda: np.add.reduceat(a, [0], dtype=StatefulDtype()))

    honest_float32 = billed(
        lambda: np.add.reduceat(
            fnp.asarray(np.full((N,), 2, dtype=np.int32)), [0], dtype=np.float32
        )
    )
    honest_float64 = billed(
        lambda: np.add.reduceat(
            fnp.asarray(np.full((N,), 2, dtype=np.int32)), [0], dtype=np.float64
        )
    )

    assert calls == 1, "sanity: dtype= must be resolved exactly once"
    assert cost == honest_float32, "the bill must match the single read actually used"
    assert cost < honest_float64, "sanity: float64 is the genuinely pricier dtype here"


def test_axis_that_fails_resolution_first_and_succeeds_second_cannot_buy_a_free_scan():
    """A caller-supplied ``axis`` whose ``__index__`` RAISES on its first
    call and SUCCEEDS on a second must not be able to slip past
    flopscope's own resolution (which floors the cost when it can't pin
    ``axis`` down) only to have the real ``ufunc.reduceat`` call re-resolve
    the SAME object a second time and execute successfully.

    flopscope's own resolution is now authoritative: a form it cannot pin
    down raises HERE, before any budget is deducted and before numpy is
    ever called, using the very first (and only) read of ``axis`` -- so
    there is no second call left for a stateful object to succeed on.
    """
    N = 1_000_000
    calls = 0

    class FlakyAxis:
        def __index__(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TypeError("boom on first call")
            return 0

    a = fnp.asarray(np.full(N, 2, dtype=np.int64))
    cost = billed_raising(
        lambda: np.subtract.reduceat(a, [0], axis=FlakyAxis()),  # type: ignore[arg-type]
        TypeError,
    )

    assert calls == 1, "sanity: axis must be read exactly once, not retried"
    assert cost == 0, "a form flopscope refuses must never reach the real numpy call"


@pytest.mark.parametrize(
    "make_axis,ndim,expect_ok",
    [
        pytest.param(lambda: None, 1, True, id="none-on-1d-ok"),
        pytest.param(lambda: None, 2, False, id="none-on-2d-rejected"),
        pytest.param(lambda: -1, 2, True, id="negative-axis-ok"),
        pytest.param(lambda: (0,), 2, True, id="one-tuple-ok"),
        pytest.param(lambda: (0, 1), 2, False, id="two-tuple-rejected"),
        pytest.param(lambda: True, 2, False, id="bool-axis-rejected"),
        pytest.param(lambda: 5, 2, False, id="out-of-range-rejected"),
        pytest.param(lambda: np.int64(0), 2, True, id="np-int64-ok"),
        pytest.param(lambda: np.array(0), 2, True, id="0d-int-array-ok"),
        # A LIST axis: real ``ufunc.reduceat`` accepts only a bare int or a
        # length-1 tuple, never a list -- regardless of length or content
        # (even a single in-range int). This must raise the same
        # ``TypeError`` raw numpy raises, not be silently normalized into
        # a tuple and executed.
        pytest.param(lambda: [0], 2, False, id="single-elem-list-rejected"),
        pytest.param(lambda: [0, 1], 2, False, id="multi-elem-list-rejected"),
        pytest.param(lambda: [[0], [1]], 2, False, id="nested-list-rejected"),
    ],
)
def test_flopscope_accept_reject_boundary_matches_raw_numpy(make_axis, ndim, expect_ok):
    """Pin flopscope's own accept/reject boundary for ``axis`` against RAW
    numpy's, for every form this module's other tests rely on plus the
    forms that must keep being rejected: flopscope must neither refuse a
    form numpy accepts nor silently accept one numpy refuses.
    """
    shape = (6,) if ndim == 1 else (6, 4)
    axis = make_axis()
    a_raw = np.full(shape, 2, dtype=np.int64)

    if expect_ok:
        raw_result = np.subtract.reduceat(a_raw, [0], axis=axis)  # type: ignore[arg-type]
        fnp_cost = billed(
            lambda: np.subtract.reduceat(
                fnp.asarray(a_raw.copy()),
                [0],
                axis=axis,  # type: ignore[arg-type]
            )
        )
        assert fnp_cost > 0
        assert raw_result is not None  # sanity: raw numpy really did accept this form
    else:
        with pytest.raises(Exception) as raw_exc_info:
            np.subtract.reduceat(a_raw, [0], axis=axis)  # type: ignore[arg-type]
        cost = billed_raising(
            lambda: np.subtract.reduceat(
                fnp.asarray(a_raw.copy()),
                [0],
                axis=axis,  # type: ignore[arg-type]
            ),
            type(raw_exc_info.value),
        )
        assert cost == 0, "a refused axis form must never reach the real numpy call"


@pytest.mark.parametrize(
    "axis",
    [[0], [0, 1], [[0], [1]]],
    ids=["single-elem", "multi-elem", "nested"],
)
def test_reduceat_list_axis_is_never_normalized_into_a_tuple_and_executed(axis):
    """A LIST axis must not be silently converted to a tuple and executed:
    real ``ufunc.reduceat`` rejects every list form outright (regardless of
    length or content), so flopscope must reject it too, with the exact
    same exception type raw numpy raises, and without ever reaching the
    real reduceat call (which is what would happen if a list were quietly
    normalized into an accepted tuple).
    """
    a_raw = np.full((6, 4), 2, dtype=np.int64)
    with pytest.raises(Exception) as raw_exc_info:
        np.subtract.reduceat(a_raw, [0], axis=axis)  # type: ignore[arg-type]

    cost = billed_raising(
        lambda: np.subtract.reduceat(
            fnp.asarray(a_raw.copy()),
            [0],
            axis=axis,  # type: ignore[arg-type]
        ),
        type(raw_exc_info.value),
    )
    assert cost == 0


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


class _HonestIndexSubclass(np.ndarray):
    """A plain ndarray subclass that does NOT override ``.dtype`` -- legitimate
    subclassing must keep working exactly like a bare ndarray index.
    """


@pytest.mark.parametrize(
    "index_dtype",
    [np.float64, np.uint64],
    ids=["float64", "uint64"],
)
@pytest.mark.parametrize(
    "raw_indices",
    [
        pytest.param([4, 4, 4], id="copy-only-global-minimum"),
        pytest.param([0, 4], id="nonzero-arithmetic"),
    ],
)
def test_reduceat_rejected_ndarray_index_dtype_retains_existing_bill(
    index_dtype, raw_indices
):
    n = 5
    dtype = np.int64
    a_raw = np.full(n, 2, dtype=dtype)
    indices = np.array(raw_indices, dtype=index_dtype)

    with pytest.raises(TypeError):
        np.subtract.reduceat(a_raw, indices)

    actual = billed_raising(
        lambda: np.subtract.reduceat(fnp.asarray(a_raw.copy()), indices),
        TypeError,
    )
    applications = int(application_count_oracle(n, indices))
    expected = honest_reduceat_cost(dtype, lanes=1, billed_units_per_lane=applications)
    assert actual == expected


def test_reduceat_lying_dtype_float_index_matches_raw_numpy_rejection():
    """A ``.dtype``-lying subclass over a FLOAT buffer must be rejected with
    the exact exception type raw numpy raises for the equivalent honest
    float index -- not silently truncated into an accepted integer index.
    """
    a_raw = np.ones((5, 2))
    honest_float_index = np.array([0.0, 1.0])
    lying = honest_float_index.view(_LyingDtypeIndex)
    assert lying.dtype == np.dtype(np.intp), "sanity: the property really does lie"

    with pytest.raises(Exception) as raw_exc_info:
        np.add.reduceat(a_raw, honest_float_index)

    billed_raising(
        lambda: np.add.reduceat(fnp.asarray(a_raw.copy()), lying),
        type(raw_exc_info.value),
    )


def test_reduceat_lying_dtype_narrow_int_index_does_not_truncate_values():
    """An integer-backed subclass reporting a NARROWER dtype than its real
    buffer must not have its real (wide) index values truncated by the
    lying property -- that would touch different cells than the caller's
    actual data implies. Use an index value that overflows int8 but is a
    valid axis position at the real (wider) dtype.
    """

    class _LyingNarrowInt(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(np.int8)

    n = 300
    a_raw = np.arange(n, dtype=np.int64)
    real_index = np.array([200], dtype=np.int64)  # 200 overflows int8
    lying = real_index.view(_LyingNarrowInt)

    raw_result = np.add.reduceat(a_raw, real_index)
    cost = billed(lambda: np.add.reduceat(fnp.asarray(a_raw.copy()), lying))
    honest_cost = billed(lambda: np.add.reduceat(fnp.asarray(a_raw.copy()), real_index))
    assert cost == honest_cost, (
        "the bill must reflect the real (wide) index, not a truncated one"
    )
    assert raw_result is not None  # sanity: raw numpy accepts the real (wide) index


def test_reduceat_honest_ndarray_subclass_index_still_works():
    """A plain ndarray subclass that reports its REAL dtype honestly must
    keep working exactly like a bare ndarray index, and bill identically --
    subclassing is legitimate, only lying about ``.dtype`` is the problem.
    """
    a_raw = np.full((5, 2), 2, dtype=np.int64)
    honest = np.array([0, 1], dtype=np.int32).view(_HonestIndexSubclass)

    raw_result = np.add.reduceat(a_raw, honest)
    cost = billed(lambda: np.add.reduceat(fnp.asarray(a_raw.copy()), honest))
    honest_plain_cost = billed(
        lambda: np.add.reduceat(
            fnp.asarray(a_raw.copy()), np.array([0, 1], dtype=np.int32)
        )
    )
    assert cost == honest_plain_cost
    assert raw_result is not None


class _LyingDtypeOperand(np.ndarray):
    """An ndarray subclass whose ``.dtype`` property claims ``int8`` no
    matter what the real underlying buffer is -- used to probe the ``a`` and
    ``out=`` operands (not the index), which are exposed to the same
    property-override trick whenever the OTHER operand is what triggers
    flopscope's dispatch.
    """

    @property
    def dtype(self):
        return np.dtype(np.int8)


def test_reduceat_lying_dtype_a_operand_bills_the_real_dtype():
    """``a`` itself (not just ``indices``) can be a foreign lying subclass --
    dispatch happens as soon as EITHER ``a`` or ``out=`` is flopscope-aware.
    The bill must reflect the REAL buffer dtype numpy computes at, not the
    cheap one a lying ``.dtype`` property reports. Uses ``subtract`` so
    there is no sum-accumulator widening (which would put int8 and float64
    at the same rate and mask the divergence).
    """
    load_weights()
    n = 4000
    real_float64 = np.ones(n, dtype=np.float64).view(_LyingDtypeOperand)
    assert real_float64.dtype == np.dtype(np.int8), "sanity: the property lies"

    from flopscope._pointwise import _counted_ufunc_reduceat

    lying_cost = billed(lambda: _counted_ufunc_reduceat(np.subtract, real_float64, [0]))
    honest_float64_cost = billed(
        lambda: _counted_ufunc_reduceat(np.subtract, np.ones(n, dtype=np.float64), [0])
    )
    honest_int8_cost = billed(
        lambda: _counted_ufunc_reduceat(np.subtract, np.ones(n, dtype=np.int8), [0])
    )
    assert lying_cost == honest_float64_cost
    assert lying_cost > honest_int8_cost


def test_reduceat_lying_dtype_out_operand_bills_the_real_dtype():
    """Same exposure as the ``a`` operand test above, but for ``out=``: its
    dtype participates in the billing rate (via ``store_billing_dtypes``)
    and must be read off the real buffer, not a Python-level override.
    Uses ``subtract`` (rather than ``add``/``multiply``) so there is no
    sum-accumulator widening on ``a`` to dominate the rate and mask the
    ``out=`` divergence.
    """
    load_weights()
    n = 2000
    a = np.full(n, 2, dtype=np.int8)
    lying_out = np.zeros(1, dtype=np.float64).view(_LyingDtypeOperand)
    assert lying_out.dtype == np.dtype(np.int8), "sanity: the property lies"

    from flopscope._pointwise import _counted_ufunc_reduceat

    lying_cost = billed(
        lambda: _counted_ufunc_reduceat(np.subtract, a, [0], out=lying_out)
    )
    honest_int8_cost = billed(
        lambda: _counted_ufunc_reduceat(
            np.subtract, a, [0], out=np.zeros(1, dtype=np.int8)
        )
    )
    honest_float64_cost = billed(
        lambda: _counted_ufunc_reduceat(
            np.subtract, a, [0], out=np.zeros(1, dtype=np.float64)
        )
    )
    assert lying_cost == honest_float64_cost
    assert lying_cost > honest_int8_cost


# ----- stale billing snapshot: participant-controlled resolution (axis=,
# dtype=) mutating ``a`` in place AFTER its billing view was already read -----


class _OwningFloat64(np.ndarray):
    """A plain ndarray subclass that OWNS its data (is not a view of
    something else), constructed the same way ``np.empty``/``np.zeros``
    would build one. That ownership is exactly what makes
    ``resize(n, refcheck=False)`` a legitimate in-place grow rather than a
    ``ValueError`` -- any array a caller builds directly, not a slice or
    view of something else, has it.
    """

    def __new__(cls, n, fill=1.0):
        obj = super().__new__(cls, (n,), dtype=np.float64)
        obj[...] = fill
        return obj


class _ResizingAxis:
    """An ``axis=`` object whose ``__index__`` -- before returning a valid
    axis -- resizes ``a`` in place via an OWNING ndarray subclass's
    ``resize(n, refcheck=False)``.
    """

    def __init__(self, a, new_size, axis=0):
        self._a = a
        self._new_size = new_size
        self._axis = axis

    def __index__(self):
        self._a.resize(self._new_size, refcheck=False)
        return self._axis


def test_reduceat_resizing_axis_bills_the_post_resize_array():
    """Regression pin for the confirmed stale-billing-snapshot defect:
    ``a``'s billing view used to be captured BEFORE ``axis=`` was
    resolved. Resolving ``axis`` invokes ``__index__``, participant code,
    which here grows ``a`` from 4 elements to ``n`` in place -- a
    legitimate operation for an array that owns its data. The bill must
    reflect the ``n``-element array ``ufunc.reduceat`` actually reduces,
    not the 4-element array that existed before ``axis`` was resolved.
    ``out=`` is a flopscope array so dispatch reaches flopscope's wrapper
    even though ``a`` itself is a foreign (non-flopscope) subclass, mirroring
    how a real caller would trigger this path.
    """
    n = 1_000_000
    a = _OwningFloat64(4)
    axis = _ResizingAxis(a, n)

    cost = billed(lambda: np.add.reduceat(a, [0], axis=axis, out=fnp.zeros((1,))))
    assert a.size == n, "sanity: the resize actually ran"

    honest = billed(lambda: np.add.reduceat(fnp.asarray(np.full(n, 1.0)), [0]))
    assert cost == honest, (
        "the bill must match the post-resize array numpy actually reduced, "
        "not a snapshot taken before axis resolution ran"
    )


def test_reduceat_resizing_dtype_bills_the_post_resize_array():
    """Same defect, a different participant-controlled vector: a ``dtype=``
    object whose ``.dtype`` PROPERTY -- which ``np.dtype()`` honours --
    resizes ``a`` as a side effect of reporting a valid dtype, instead of
    ``axis=``'s ``__index__``. Calls the wrapper directly (like the
    lying-dtype tests above) so the attack is exercised even though ``a``
    itself never becomes flopscope-aware.
    """
    n = 1_000_000
    a = _OwningFloat64(4)

    class _ResizingDtype:
        @property
        def dtype(self):
            a.resize(n, refcheck=False)
            return np.dtype(np.float64)

    from flopscope._pointwise import _counted_ufunc_reduceat

    cost = billed(
        lambda: _counted_ufunc_reduceat(np.add, a, [0], dtype=_ResizingDtype())
    )
    assert a.size == n, "sanity: the resize actually ran"

    honest = billed(
        lambda: _counted_ufunc_reduceat(np.add, np.full(n, 1.0), [0], dtype=np.float64)
    )
    assert cost == honest


# ----- stale billing snapshot: ``out=``'s billing view captured too early -----


class _OwningFloat32(np.ndarray):
    """A plain ndarray subclass that OWNS its data, built at float32 -- the
    width the reproduction below needs so that widening to float64 in
    place (``obj.dtype = np.float64``) is a legal same-byte-count
    reinterpretation (numpy requires the last axis's byte count to be
    divisible by the new itemsize).
    """

    def __new__(cls, n):
        obj = super().__new__(cls, (n,), dtype=np.float32)
        obj[...] = 0.0
        return obj


def test_reduceat_out_view_captured_before_indices_resolves_bills_the_post_widen_store():
    """Regression pin for the confirmed stale-``out=``-billing-view defect:
    ``out``'s billing view used to be captured BEFORE ``indices`` was
    resolved (on the reasoning that ``out`` has no shape/dtype dependency
    on ``a`` or ``indices`` -- true, but irrelevant: the exposure is
    participant code reachable from ANY later resolution step, not a data
    dependency). Resolving a non-ndarray ``indices`` invokes ``__array__``,
    participant code, which here widens ``out`` from float32 to float64 IN
    PLACE (a legitimate reinterpretation for an array that owns its data).
    The bill must reflect the float64 store ``ufunc.reduceat`` actually
    writes into, not the float32 rate that existed before ``indices`` was
    ever touched.

    Uses production dtype rates (``load_weights()``) rather than this
    module's unit-rate ``billed()`` default -- float32 and float64 bill
    identically under unit rates, which would hide exactly the divergence
    this test exists to catch.
    """
    load_weights()
    a = fnp.asarray(np.ones(8, np.float32))
    out = _OwningFloat32(4)  # 16 bytes -> widens to 2 x float64

    class Idx:
        def __array__(self, dtype=None, copy=None):
            # widen AFTER a naive meter would have viewed `out`
            out.dtype = np.float64  # pyright: ignore[reportAttributeAccessIssue]
            return np.array([0, 1], np.intp)

    cost = billed(lambda: np.add.reduceat(a, Idx(), out=out))
    assert out.dtype == np.float64, "sanity: the widen actually ran"

    honest_float32 = billed(
        lambda: np.add.reduceat(
            fnp.asarray(np.ones(8, np.float32)), [0, 1], out=np.zeros(2, np.float32)
        )
    )
    honest_float64 = billed(
        lambda: np.add.reduceat(
            fnp.asarray(np.ones(8, np.float32)), [0, 1], out=np.zeros(2, np.float64)
        )
    )
    assert honest_float64 == 2 * honest_float32, (
        "sanity: the widened store really is twice the honest price"
    )
    assert cost == honest_float64, (
        "the bill must reflect the post-widen store, not a snapshot taken "
        "before indices resolved"
    )
