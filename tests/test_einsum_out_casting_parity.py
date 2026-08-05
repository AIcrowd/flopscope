"""``fnp.einsum(..., out=)`` must accept exactly what ``np.einsum`` accepts.

flopscope used to write its answer into ``out=`` with
``np.copyto(..., casting="unsafe")``. numpy's einsum resolves the destination
under ``casting='safe'``, so flopscope was silently *looser* than numpy in two
directions at once:

* two float64 operands into an int64 destination truncated to whole numbers
  where numpy raises ``TypeError``;
* a complex result into a float64 destination dropped the imaginary part with
  nothing louder than a ``ComplexWarning``.

Being looser is the defect. Being *stricter* would be worse: it breaks working
participant code. A first attempt at this fix shipped a hand-written guard that
refused 138 cells plain numpy accepts, and was reverted. So these tests are
driven off the dtype matrix and differential against plain numpy -- accept set,
exception class, and the values actually written -- rather than off a
hand-listed set of cases someone believed to be right.

The rule flopscope implements, measured to agree with numpy 2.0.2 through
2.4.2 over the whole matrix::

    op_dtype = np.result_type(*operand_dtypes, out.dtype)   # out PARTICIPATES
    accept  iff np.can_cast(op_dtype, out.dtype, 'safe')

and the contraction then runs in ``op_dtype``, which is why the values need
their own assertion: casting the *answer* into ``out`` is a different operation
from computing in the destination's dtype.
"""

import itertools
import warnings

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope.errors import UnsupportedDtypeError

# The 16 dtypes of the reference matrix.
MATRIX_DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
    np.complex64,
    np.complex128,
    np.clongdouble,
]

N = 3
_BUDGET = 10**14


def _operand(dtype, shape):
    """A constant 10 (or True) -- small enough to be exact in float16, large
    enough that an int8 contraction overflows where an int16 one does not, so
    the value assertions actually discriminate between computing in the
    operands' dtype and computing in the destination's."""
    base = np.ones(shape) if np.dtype(dtype) == np.bool_ else np.ones(shape) * 10
    return base.astype(dtype)


def _attempt(call):
    """Run ``call``; return ``("ok", written)`` or ``(exception name, str)``.

    ``ComplexWarning`` is promoted to an error: numpy's own contract is that
    dropping an imaginary part on the way into ``out`` is a refusal, not a
    warning, and the old flopscope behaviour was exactly that warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        try:
            return ("ok", np.asarray(call()).copy())
        except Exception as exc:  # noqa: BLE001 -- the class is the assertion
            return (type(exc).__name__, str(exc))


def _numpy_ref(subscripts, operands, out):
    # optimize=False is numpy's default and the only reference stable across
    # 2.0 -> 2.4: numpy 2.4's optimize=True path regressed into the same
    # unsafe-store defect being fixed here, and is not even self-consistent
    # across arities.
    return _attempt(lambda: np.einsum(subscripts, *operands, out=out, optimize=False))


def _flopscope(subscripts, operands, out):
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        return _attempt(lambda: fnp.einsum(subscripts, *operands, out=out))


def _compare(subscripts, dtypes, out_dtype, shapes, out_shape):
    operands = [_operand(d, s) for d, s in zip(dtypes, shapes, strict=True)]
    ref = _numpy_ref(subscripts, operands, np.zeros(out_shape, dtype=out_dtype))
    got = _flopscope(subscripts, operands, np.zeros(out_shape, dtype=out_dtype))
    return ref, got


def _assert_agrees(subscripts, dtypes, out_dtype, shapes, out_shape):
    ref, got = _compare(subscripts, dtypes, out_dtype, shapes, out_shape)
    names = [np.dtype(d).name for d in dtypes]
    where = f"{subscripts} {names} -> {np.dtype(out_dtype).name}"

    assert (ref[0] == "ok") == (got[0] == "ok"), (
        f"accept/refuse disagreement for {where}: numpy={ref}, flopscope={got}"
    )
    if ref[0] == "ok":
        assert np.array_equal(ref[1], got[1]), (
            f"value disagreement for {where}: numpy={ref[1]}, flopscope={got[1]}"
        )
    else:
        assert ref[0] == got[0], (
            f"exception class disagreement for {where}: "
            f"numpy={ref[0]}, flopscope={got[0]}"
        )


# --- the full 16 x 16 x 16 matrix, one parametrization per destination -----


@pytest.mark.parametrize("out_dtype", MATRIX_DTYPES, ids=lambda d: np.dtype(d).name)
def test_matmul_matrix_matches_numpy_cell_by_cell(out_dtype):
    """256 operand pairs per destination, 4096 cells in total: accept/refuse,
    exception class, and written values, all against plain numpy."""
    shapes = ((N, N), (N, N))
    for da, db in itertools.product(MATRIX_DTYPES, repeat=2):
        _assert_agrees("ij,jk->ik", (da, db), out_dtype, shapes, (N, N))


def test_matrix_accept_count_tracks_numpys_own():
    """A tripwire on the SIZE of the accept set, measured against numpy rather
    than frozen.

    A literal count cannot work here: the accept set is numpy-version
    dependent -- 1154 on numpy 2.2 against 1069 on 2.3 -- so a frozen number
    passes on the machine it was measured on and fails the matrix. Asking
    numpy for its own count on the same cells keeps the tripwire (a fix that
    drifts either way still has to be deliberate) without pinning a constant
    that only holds for one numpy.

    The cell-by-cell tests above are the real assertion; this catches a
    wholesale collapse, e.g. a guard that starts refusing everything and
    still agrees per-cell because the comparison itself broke."""
    accepted = expected = 0
    for da, db in itertools.product(MATRIX_DTYPES, repeat=2):
        for dout in MATRIX_DTYPES:
            want, got = _compare("ij,jk->ik", (da, db), dout, ((N, N), (N, N)), (N, N))
            expected += want[0] == "ok"
            accepted += got[0] == "ok"
    assert expected > 0, "test bug: numpy accepted nothing, the matrix is broken"
    assert accepted == expected


# --- other subscripts and arities -----------------------------------------

_SHAPED_CASES = [
    ("ij->ji", 1, ((N, N),), (N, N)),
    ("ii->", 1, ((N, N),), ()),
    ("ij->i", 1, ((N, N),), (N,)),
    ("ij,ij->", 2, ((N, N), (N, N)), ()),
    ("i,j->ij", 2, ((N,), (N,)), (N, N)),
    ("ij,jk,kl->il", 3, ((N, N), (N, N), (N, N)), (N, N)),
    ("ij,jk,kl,lm->im", 4, ((N, N),) * 4, (N, N)),
]


@pytest.mark.parametrize(
    "subscripts,arity,shapes,out_shape", _SHAPED_CASES, ids=lambda v: str(v)[:24]
)
def test_other_subscripts_and_arities_match_numpy(subscripts, arity, shapes, out_shape):
    """The rule is arity- and subscript-independent, including the operand
    index quoted in the refusal message (it is the number of *inputs*)."""
    rng = np.random.default_rng(0)
    if arity == 1:
        combos = [(d,) for d in MATRIX_DTYPES]
    elif arity == 2:
        combos = list(itertools.product(MATRIX_DTYPES, repeat=2))
    else:
        combos = [
            tuple(MATRIX_DTYPES[i] for i in rng.integers(0, len(MATRIX_DTYPES), arity))
            for _ in range(40)
        ]
    for combo in combos:
        for dout in MATRIX_DTYPES:
            _assert_agrees(subscripts, combo, dout, shapes, out_shape)


# --- the regression pin for the reverted first attempt --------------------

# Mixed-signedness narrow integers whose pairwise promotion overshoots the
# destination, aimed at a float/complex destination that holds each operand
# exactly. `np.result_type` is a lattice minimum, so it picks the destination;
# a left-fold over the operands alone picks something wider and then wrongly
# concludes the store is unsafe. All ten are accepted by numpy and write exact
# values. Attempt #1 refused all ten.
DISCRIMINATORS = [
    (np.int8, np.uint8, np.float16),
    (np.uint8, np.int8, np.float16),
    (np.int8, np.uint16, np.float32),
    (np.int8, np.uint16, np.complex64),
    (np.int16, np.uint16, np.float32),
    (np.int16, np.uint16, np.complex64),
    (np.uint16, np.int8, np.float32),
    (np.uint16, np.int8, np.complex64),
    (np.uint16, np.int16, np.float32),
    (np.uint16, np.int16, np.complex64),
]


@pytest.mark.parametrize("da,db,dout", DISCRIMINATORS, ids=lambda d: np.dtype(d).name)
def test_lattice_promotion_cells_are_accepted(da, db, dout):
    """These are the cells that reverted the first attempt at this fix."""
    ref, got = _compare("ij,jk->ik", (da, db), dout, ((N, N), (N, N)), (N, N))
    assert ref[0] == "ok", "precondition: numpy accepts this"
    assert got[0] == "ok", (
        f"flopscope refused {np.dtype(da).name} x {np.dtype(db).name} -> "
        f"{np.dtype(dout).name}, which numpy accepts: {got[1]}"
    )
    assert np.array_equal(ref[1], got[1])


def test_int8_uint8_into_float16_regression_pin():
    """The named case from the reverted attempt, spelled out verbatim.

    ``result_type(int8, uint8)`` is int16, which does not cast safely into
    float16 -- but ``result_type(int8, uint8, float16)`` is float16, which
    does, and float16 represents every int8 and every uint8 exactly. numpy
    accepts this and writes exact values; the first attempt raised TypeError.
    """
    rng = np.random.default_rng(0)
    a = rng.integers(1, 5, (6, 6)).astype(np.int8)
    b = rng.integers(1, 5, (6, 6)).astype(np.uint8)

    expected = np.zeros((6, 6), np.float16)
    np.einsum("ij,jk->ik", a, b, out=expected, optimize=False)

    out = np.zeros((6, 6), np.float16)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        returned = fnp.einsum("ij,jk->ik", a, b, out=out)

    assert np.array_equal(np.asarray(returned), expected)
    # and exact against an object-dtype reference, not just against numpy
    exact = np.einsum("ij,jk->ik", a.astype(object), b.astype(object))
    assert np.array_equal(expected.astype(object), exact)


# --- the defect itself ----------------------------------------------------


@pytest.mark.parametrize(
    "da,db,dout,why",
    [
        (np.float64, np.float64, np.int64, "float64 truncated to int64"),
        (np.float64, np.float64, np.int32, "float64 truncated to int32"),
        (np.float64, np.float64, np.float32, "float64 narrowed to float32"),
        (np.int64, np.int64, np.int32, "int64 narrowed to int32"),
        (np.complex128, np.complex128, np.float64, "imaginary part dropped"),
        (np.complex64, np.complex64, np.float32, "imaginary part dropped"),
        (np.float64, np.float64, np.bool_, "everything collapsed to a flag"),
    ],
)
def test_narrowing_destinations_now_raise(da, db, dout, why):
    """Each of these used to succeed and write a wrong number."""
    a = _operand(da, (N, N))
    b = _operand(db, (N, N))
    out = np.zeros((N, N), dtype=dout)

    with pytest.raises(TypeError, match="could not be cast"):
        np.einsum("ij,jk->ik", a, b, out=out, optimize=False)

    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(TypeError, match="could not be cast"):
            fnp.einsum("ij,jk->ik", a, b, out=out)


def test_complex_into_float_does_not_truncate_with_a_warning():
    """The old behaviour was a ``ComplexWarning`` and a real, wrong write."""
    a = (np.ones((N, N)) + 2j).astype(np.complex128)
    out = np.zeros((N, N), dtype=np.float64)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, a, out=out)
    assert np.array_equal(out, np.zeros((N, N)))  # untouched


# --- computing in the destination's dtype, not casting the answer ---------


@pytest.mark.parametrize(
    "da,db,dout,expected",
    [
        # 3 * 10 * 10 = 300 overflows int8; numpy contracts in the
        # destination's dtype and returns 300, so flopscope must too.
        (np.int8, np.int8, np.int16, 300),
        (np.int8, np.int8, np.int32, 300),
        (np.int8, np.int8, np.float32, 300),
        (np.uint8, np.uint8, np.int32, 300),
        # bool x bool contracted in bool is a logical or (1); contracted in a
        # numeric destination it counts the matches (3).
        (np.bool_, np.bool_, np.int8, 3),
        (np.bool_, np.bool_, np.float64, 3),
    ],
)
def test_contraction_runs_in_the_resolved_dtype(da, db, dout, expected):
    ref, got = _compare("ij,jk->ik", (da, db), dout, ((N, N), (N, N)), (N, N))
    assert ref[0] == "ok" and got[0] == "ok"
    assert np.all(ref[1] == expected), "precondition: numpy computes in out's dtype"
    assert np.array_equal(got[1], ref[1])


# --- refusals are free ----------------------------------------------------


def test_refusal_costs_zero_flops():
    """The guard sits above ``budget.deduct``. flops_used never decreases and
    nothing is refunded, so a refused call must never have been billed."""
    a = np.ones((200, 200), dtype=np.float64)
    with f.BudgetContext(flop_budget=10**12, quiet=True) as budget:
        before = budget.flops_used
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, a, out=np.zeros((200, 200), np.int64))
        assert budget.flops_used == before == 0

        # the context is still usable, and a legal call still bills
        fnp.einsum("ij,jk->ik", a, a, out=np.zeros((200, 200), np.float64))
        assert budget.flops_used > 0


def test_refusal_is_free_even_when_the_budget_could_not_afford_it():
    """A refusal must not turn into a budget-exhaustion error either: the
    dtype question is settled before the budget is consulted at all."""
    a = np.ones((200, 200), dtype=np.float64)
    with f.BudgetContext(flop_budget=1, quiet=True) as budget:
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, a, out=np.zeros((200, 200), np.int64))
        assert budget.flops_used == 0


def test_refused_destination_is_left_untouched():
    a = np.ones((4, 4), dtype=np.float64) * 3
    out = np.full((4, 4), 7, dtype=np.int64)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, a, out=out)
    assert np.array_equal(out, np.full((4, 4), 7, dtype=np.int64))


# --- message and exception-class parity -----------------------------------


def test_refusal_message_matches_numpy_verbatim():
    a = np.ones((N, N), dtype=np.float64)
    out = np.zeros((N, N), dtype=np.int64)
    with pytest.raises(TypeError) as numpy_exc:
        np.einsum("ij,jk->ik", a, a, out=out, optimize=False)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(TypeError) as flopscope_exc:
            fnp.einsum("ij,jk->ik", a, a, out=out)
    assert str(flopscope_exc.value) == str(numpy_exc.value)


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_refusal_message_operand_index_is_the_input_count(arity):
    """numpy names the destination by its position in the iterator, which is
    the number of input operands."""
    subscripts = {
        1: "ij->ij",
        2: "ij,jk->ik",
        3: "ij,jk,kl->il",
    }[arity]
    ops = [np.ones((N, N), dtype=np.float64) for _ in range(arity)]
    out = np.zeros((N, N), dtype=np.int64)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(TypeError, match=f"the operand {arity} dtype"):
            fnp.einsum(subscripts, *ops, out=out)


def test_datetime_destination_is_refused():
    """Plain numpy's ``np.result_type`` fails outright rather than reporting
    a cast for a float64-vs-datetime64 destination, and used to let
    ``DTypePromotionError`` out; flopscope let the same error out too, since
    at the time datetime64 was still a priced, non-refused destination kind.

    Under the numeric-allowlist dtype ban, a datetime64 destination is
    refused before ``np.result_type`` is ever consulted -- the ban's own
    ``UnsupportedDtypeError`` fires first, not numpy's promotion error. It is
    a ``TypeError`` subclass either way, so the accept/refuse contract is
    unaffected; only which exception class explains the refusal changes."""
    a = np.ones((N, N), dtype=np.float64)
    out = np.zeros((N, N), dtype="M8[ns]")
    with pytest.raises(np.exceptions.DTypePromotionError):
        np.einsum("ij,jk->ik", a, a, out=out, optimize=False)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("ij,jk->ik", a, a, out=out)
        assert budget.flops_used == 0


@pytest.mark.parametrize("out_dtype", ["U64", "S64", "U32", "S16"])
def test_string_destinations_are_refused(out_dtype):
    """einsum has no string inner loop, and numpy itself either fails the
    cast rule or reaches the missing loop and mis-reports the resulting
    ``TypeError('invalid data type for einsum')`` as a ``SystemError``.

    Under the numeric-allowlist dtype ban, flopscope refuses a string
    destination via its own ``UnsupportedDtypeError`` before numpy's einsum
    call runs at all, rather than by propagating whatever numpy's own
    casting/loop failure happens to be. ``UnsupportedDtypeError`` is a
    ``TypeError`` subclass, so this ``pytest.raises(TypeError)`` still holds
    either way, and refusal is still free."""
    a = np.ones((N, N), dtype=np.float64)
    out = np.empty((N, N), dtype=out_dtype)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True) as budget:
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, a, out=out)
        assert budget.flops_used == 0


def test_object_destination_is_refused():
    """object was a legal einsum destination through 0.10.0. As of 0.10.1 it is
    refused: a destination dtype carrying Python objects cannot be priced, so
    allowing it back in reopens the same unbounded-cost gap
    ``refuse_non_numeric_dtype`` closes everywhere else -- widened since to a
    numeric allowlist covering every non-numeric kind, not just object."""
    a = np.ones((N, N), dtype=np.float64) * 2
    out = np.zeros((N, N), dtype=object)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("ij,jk->ik", a, a, out=out)


# --- the destination still behaves like a destination ---------------------


def test_accepted_widening_still_writes_through_and_returns_out():
    a = np.ones((4, 4), dtype=np.float32) * 2
    out = np.zeros((4, 4), dtype=np.float64)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True):
        returned = fnp.einsum("ij,jk->ik", a, a, out=out)
    assert returned is out
    assert np.array_equal(out, np.full((4, 4), 16.0))


@pytest.mark.parametrize("optimize", [True, False, "greedy", "optimal"])
def test_parity_holds_for_every_optimize_setting(optimize):
    """The accept set is a property of the dtypes, not of the path. (numpy's
    own optimize=True path disagrees with itself across versions, which is why
    the reference is optimize=False.)"""
    a = np.ones((4, 4), dtype=np.float64)
    with f.BudgetContext(flop_budget=_BUDGET, quiet=True) as budget:
        with pytest.raises(TypeError):
            fnp.einsum(
                "ij,jk,kl->il",
                a,
                a,
                a,
                out=np.zeros((4, 4), np.int64),
                optimize=optimize,
            )
        assert budget.flops_used == 0
        result = fnp.einsum(
            "ij,jk,kl->il",
            a,
            a,
            a,
            out=np.zeros((4, 4), np.float64),
            optimize=optimize,
        )
    assert np.array_equal(np.asarray(result), np.full((4, 4), 16.0))


def test_a_promoting_contraction_does_not_materialize_its_operands(monkeypatch):
    """Casting an operand materializes its LOGICAL shape.

    A broadcast view has O(1) storage and O(numel) logical size, so promoting
    one turned a 4-byte view into an allocation the size of the contraction
    (~80GB at 100000 square; measured 128MB against numpy's 0.1MB at 4000).
    ``dtype=`` casts inside numpy's iterator, per element, so the view is
    never materialized.

    Asserted on the ARRAY numpy is handed rather than on process RSS. A
    broadcast view carries a zero stride and a materialized copy does not, so
    the distinction is exact and identical on every platform -- where
    ``ru_maxrss`` is KiB on Linux and bytes on macOS, and an earlier version
    of this test divided by 1e6 unconditionally, making a 128MB regression
    read as 0.13 and pass the threshold on the very runners CI uses.
    """
    calls = []
    real_einsum = np.einsum

    def spy(*args, **kwargs):
        arrays = [a for a in args[1:] if isinstance(a, np.ndarray)]
        calls.append((args[0], [a.strides for a in arrays], kwargs.get("dtype")))
        return real_einsum(*args, **kwargs)

    view = np.broadcast_to(np.float32(1.0), (4000, 4000))
    assert 0 in view.strides, "test bug: the fixture is not a broadcast view"

    with f.BudgetContext(flop_budget=10**14, quiet=True):
        operand = fnp.asarray(view)
        dest = np.empty((), np.float64)
        monkeypatch.setattr(np, "einsum", spy)
        fnp.einsum("ij->", operand, out=dest)

    assert float(dest) == 16000000.0
    # The contraction under test, not whichever call happened to come first:
    # asarray and friends issue their own einsums, and keying on calls[0] is
    # how the first version of this test looked at the wrong array entirely.
    contraction = [c for c in calls if c[0] == "ij->"]
    assert contraction, f"the contraction never reached numpy; saw {calls}"
    _, strides, dtype_kwarg = contraction[0]
    assert strides and 0 in strides[0], (
        f"the operand handed to numpy has strides {strides} with no zero -- "
        f"the broadcast view was materialized rather than cast in the iterator"
    )
    assert dtype_kwarg is not None, (
        "the promotion was not passed as dtype=, so numpy could not cast per element"
    )


def test_a_promoted_dense_contraction_still_follows_the_planned_path(monkeypatch):
    """Avoiding materialization must not cost the plan.

    Bypassing the pairwise stepper for every promoting contraction would swap
    a planned O(N^3) chain for one direct O(N^5) call, and ignore an
    explicitly supplied path. Only operands that actually expand when copied
    take the direct route; dense ones keep their steps.
    """
    calls = []
    real_einsum = np.einsum

    def spy(*args, **kwargs):
        calls.append(args[0])
        return real_einsum(*args, **kwargs)

    size = 24
    rng = np.random.default_rng(0)
    raw = [rng.random((size, size)).astype("float32") for _ in range(4)]
    dest = np.zeros((size, size), np.float64)

    with f.BudgetContext(flop_budget=10**14, quiet=True):
        operands = [fnp.asarray(m) for m in raw]
        monkeypatch.setattr(np, "einsum", spy)
        fnp.einsum("ij,jk,kl,lm->im", *operands, out=dest, optimize=True)

    # Following the plan means numpy is handed the individual STEPS and never
    # the whole 4-operand subscript; bypassing it means exactly the opposite.
    # Counting calls instead does not work -- asarray issues einsums of its
    # own, and they inflate the count enough to hide the bypass.
    assert "ij,jk,kl,lm->im" not in calls, (
        f"the whole contraction went to numpy in one call ({calls}); the "
        f"planned pairwise path was bypassed"
    )
    assert len(calls) >= 2, f"no pairwise steps reached numpy at all: {calls}"
    expected = np.einsum(
        "ij,jk,kl,lm->im", *raw, out=np.zeros((size, size), np.float64), optimize=True
    )
    np.testing.assert_allclose(dest, expected, rtol=1e-6)
