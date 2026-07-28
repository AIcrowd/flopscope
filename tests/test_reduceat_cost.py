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


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        before = ctx.flops_used
        with warnings.catch_warnings():
            # np.subtract.reduce/.reduceat on a FlopscopeArray auto-routes
            # with a UserWarning notice; harmless here, just noise.
            warnings.simplefilter("ignore", UserWarning)
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


def test_a_array_protocol_index_resize_does_not_shrink_the_bill():
    """``a``'s ``__array__`` runs before flopscope ever looks at
    ``indices`` -- ``a`` has to be resolved into a concrete array before
    ``indices`` can be interpreted against its shape at all. A participant
    who closes over the SAME ``indices`` object from inside ``a.__array__``
    and resizes it there (``ndarray.resize(..., refcheck=False)``) changes
    what flopscope's later snapshot describes. The bill must track
    whatever ``indices`` looks like when the snapshot is actually taken --
    the exact state the real ``ufunc.reduceat`` call then executes against
    -- not whatever it looked like when the caller first built it.

    Here ``indices`` starts as ``n`` singleton segments (near-zero real
    work) and is shrunk, from inside ``a.__array__``, down to a single
    index spanning the whole axis -- the maximally expensive shape for this
    ``n``. The bill must reflect the expensive, POST-mutation shape, not
    the cheap one the caller originally constructed.
    """
    n = 4000
    idx = np.arange(n, dtype=np.intp)

    class A:
        def __array__(self, dtype=None, copy=None):
            idx.resize(1, refcheck=False)
            return np.full(n, 2, dtype=np.int64)

    # ``out=`` is a FlopscopeArray purely to route the call through
    # flopscope's ``__array_ufunc__`` override -- neither ``A()`` nor the
    # plain ``idx`` ndarray carries the protocol on its own.
    out = fnp.asarray(np.zeros(1, dtype=np.int64))
    cost = billed(lambda: np.subtract.reduceat(A(), idx, out=out))

    honest_a = fnp.asarray(np.full(n, 2, dtype=np.int64))
    honest = billed(lambda: np.subtract.reduce(honest_a))

    assert idx.shape == (1,), "sanity: the resize inside __array__ must have landed"
    assert cost == honest, "the bill must track the resized (expensive) index array"
