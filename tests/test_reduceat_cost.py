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


def honest_reduceat_cost(dtype, lanes: int, applications_per_lane: int) -> int:
    """The expected bill, measured via an INDEPENDENT flopscope call.

    ``lanes`` independent ``subtract.reduce`` calls over
    ``applications_per_lane + 1``-length rows bill exactly
    ``lanes * applications_per_lane`` applications at whatever rate/weight
    is currently configured -- the same "n-1 per lane" reduction convention
    reduceat itself follows. Comparing against this pins reduceat's bill to
    ``.reduce``'s without hardcoding any rate or weight number, so it
    survives cost-model-wide rate changes.
    """
    length = applications_per_lane + 1
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
    applications = application_count_oracle(n, indices)
    actual = reduceat_actual_cost(shape, axis, indices, dtype)
    expected = honest_reduceat_cost(dtype, lanes, applications)
    assert actual == expected


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
    floor_cost = honest_reduceat_cost(dtype, lanes=1, applications_per_lane=0)
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

    floor_cost = honest_reduceat_cost(dtype, lanes=1, applications_per_lane=0)

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
    floor_cost = honest_reduceat_cost(dtype, lanes=1, applications_per_lane=0)
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
    floor_cost = honest_reduceat_cost(dtype, lanes=1, applications_per_lane=0)
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
