"""``out=`` containers: unwrap the one numpy accepts, refuse the rest for free.

numpy's ufunc protocol lets a caller write ``np.multiply(a, b, out=(dest,))``
as well as ``out=dest``; the two mean the same thing. flopscope has to see
through that tuple for itself, because it reads ``out`` several times before
numpy ever gets it — to pick the billing dtype, to check symmetry, and to
decide what to hand back.

A tuple slipping past those reads was not cosmetic. The destination's dtype
stopped participating in the rate, so a contraction into a wider buffer billed
as if the buffer were not there: ``matvec`` of float32 operands into a
complex128 destination billed 130,816 instead of 1,047,552 — one eighth
price, for a correctly computed and correctly written result.

Worse on the einsum path, which never forwards ``out`` to numpy at all. There a
container reached ``_np.asarray(container)``, which builds a NEW array, so the
result landed in that temporary, the real destination kept its old contents,
and the caller got the untouched container back having paid the full
contraction. A plausible-looking array of the wrong values at full price is the
worst failure class in a metering system.

Lists stay refused, because numpy refuses them everywhere. Refusals happen
before ``budget.deduct``, so they cost nothing — the property the whole fix
turns on, and the one the earlier version of this guard did not have.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

import flopscope as fl
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor


@pytest.fixture()
def budget():
    with fl.BudgetContext(flop_budget=10**14, quiet=True) as ctx:
        yield ctx


def _rng():
    return np.random.default_rng(20260727)


def _f32(*shape):
    # Asymmetric random data on purpose: a constant fill on a square shape
    # picks up an inferred symmetry tag, which changes the accumulation cost
    # and would pin symmetry-discounted numbers instead of the real ones.
    return fnp.asarray(_rng().standard_normal(shape).astype("float32"))


def _billed(ctx, fn):
    before = ctx.flops_used
    result = fn()
    return ctx.flops_used - before, result


# ---------------------------------------------------------------------------
# The tuple bills exactly like the bare array — the point of the change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,call,make_dest",
    [
        (
            "matvec",
            lambda o: fnp.matvec(_f32(256, 256), _f32(256), out=o),
            lambda: fnp.zeros(256, dtype="complex128"),
        ),
        (
            "vecmat",
            lambda o: fnp.vecmat(_f32(256), _f32(256, 256), out=o),
            lambda: fnp.zeros(256, dtype="complex128"),
        ),
        (
            "vecdot",
            lambda o: fnp.vecdot(_f32(256), _f32(256), out=o),
            lambda: fnp.zeros((), dtype="complex128"),
        ),
        (
            "multiply",
            lambda o: fnp.multiply(_f32(1000), _f32(1000), out=o),
            lambda: fnp.zeros(1000, dtype="complex128"),
        ),
        (
            "add",
            lambda o: fnp.add(_f32(1000), 0.0, out=o),
            lambda: fnp.zeros(1000, dtype="float64"),
        ),
        (
            "clip",
            lambda o: fnp.clip(_f32(1000), -1.0, 1.0, out=o),
            lambda: fnp.zeros(1000, dtype="complex128"),
        ),
        (
            "einsum",
            lambda o: fnp.einsum("ij,jk->ik", _f32(64, 64), _f32(64, 32), out=o),
            lambda: fnp.zeros((64, 32), dtype="complex128"),
        ),
        (
            "fft.fft",
            lambda o: fnp.fft.fft(_f32(64), out=o),
            lambda: fnp.zeros(64, dtype="complex128"),
        ),
    ],
)
def test_a_one_tuple_out_bills_exactly_like_a_bare_out(budget, name, call, make_dest):
    # The regression net. The guard's own tests going green is NOT evidence
    # the under-bill closed: a prototype had every wrapped-out test passing
    # while add(f32, 0.0, out=(f64,)) still billed half price.
    bare_cost, _ = _billed(budget, lambda: call(make_dest()))
    tuple_cost, _ = _billed(budget, lambda: call((make_dest(),)))

    assert tuple_cost == bare_cost, (
        f"{name}: out=(dest,) billed {tuple_cost} where out=dest billed "
        f"{bare_cost} — the destination's dtype is not reaching the rate"
    )


#: The ops whose ``out=`` arrives inside **kwargs rather than as a declared
#: parameter, which is how they escaped the normalization every sibling gets.
#: ``take`` is the odd one out: it declares ``out``, strips it, and still
#: refused ``out=(dest,)`` — the only op in the codebase inconsistent with its
#: siblings on the one-tuple.
_KWARGS_OUT_OPS = [
    (
        "concatenate",
        lambda o: fnp.concatenate([_f32(64, 32), _f32(64, 32)], out=o),
        lambda: fnp.zeros((128, 32), dtype="float32"),
    ),
    (
        "stack",
        lambda o: fnp.stack([_f32(64, 32), _f32(64, 32)], out=o),
        lambda: fnp.zeros((2, 64, 32), dtype="float32"),
    ),
    (
        "concat",
        lambda o: fnp.concat([_f32(64, 32), _f32(64, 32)], out=o),
        lambda: fnp.zeros((128, 32), dtype="float32"),
    ),
    (
        "isnan",
        lambda o: fnp.isnan(_f32(64, 32), out=o),
        lambda: fnp.zeros((64, 32), dtype="bool"),
    ),
    (
        "isinf",
        lambda o: fnp.isinf(_f32(64, 32), out=o),
        lambda: fnp.zeros((64, 32), dtype="bool"),
    ),
    (
        "isfinite",
        lambda o: fnp.isfinite(_f32(64, 32), out=o),
        lambda: fnp.zeros((64, 32), dtype="bool"),
    ),
    (
        "compress",
        lambda o: fnp.compress(
            np.array([True, False] * 32), _f32(64, 32), axis=0, out=o
        ),
        lambda: fnp.zeros((32, 32), dtype="float32"),
    ),
    (
        "take",
        lambda o: fnp.take(_f32(64, 32), np.arange(32), axis=0, out=o),
        lambda: fnp.zeros((32, 32), dtype="float32"),
    ),
    (
        "outer",
        lambda o: fnp.outer(_f32(64), _f32(64), out=o),
        lambda: fnp.zeros((64, 64), dtype="float32"),
    ),
]


@pytest.mark.parametrize(
    "name,call,make_dest", _KWARGS_OUT_OPS, ids=[c[0] for c in _KWARGS_OUT_OPS]
)
def test_a_kwargs_out_op_bills_a_one_tuple_like_a_bare_array(
    budget, name, call, make_dest
):
    bare_cost, _ = _billed(budget, lambda: call(make_dest()))
    tuple_cost, _ = _billed(budget, lambda: call((make_dest(),)))
    assert tuple_cost == bare_cost, (
        f"{name}: out=(dest,) billed {tuple_cost} where out=dest billed {bare_cost}"
    )


@pytest.mark.parametrize(
    "name,call,make_dest", _KWARGS_OUT_OPS, ids=[c[0] for c in _KWARGS_OUT_OPS]
)
def test_a_kwargs_out_op_refuses_a_list_for_free(budget, name, call, make_dest):
    # Built before the measurement: allocating the destination costs FLOPs of
    # its own, and building it inside is how "refusal is free" measures the
    # wrong thing. Every one of these charged in full before refusing.
    dest = make_dest()
    before = budget.flops_used
    with pytest.raises(TypeError, match="out= must be an array"):
        call([dest])
    assert budget.flops_used == before, f"{name} was billed for refusing out=[dest]"


@pytest.mark.parametrize(
    "name,call,make_dest", _KWARGS_OUT_OPS, ids=[c[0] for c in _KWARGS_OUT_OPS]
)
def test_a_kwargs_out_op_accepts_a_flopscope_destination(budget, name, call, make_dest):
    # The destination a participant actually holds is a FlopscopeArray. Every
    # one of these forwarded it to numpy still wrapped, tripping the internal
    # "missing _to_base_ndarray() strip" guard — after the deduct, so the
    # caller paid in full and then got a RuntimeError.
    dest = make_dest()
    result = call(dest)
    assert result is dest, f"{name} did not hand back the destination"


@pytest.mark.parametrize(
    "name,call,make_dest", _KWARGS_OUT_OPS, ids=[c[0] for c in _KWARGS_OUT_OPS]
)
def test_a_kwargs_out_op_treats_a_none_holding_tuple_as_no_destination(
    budget, name, call, make_dest
):
    # out=(None,) is numpy's "allocate this slot for me", not a destination.
    result = call((None,))
    assert result is not None and result.dtype != np.dtype(object)


# ---------------------------------------------------------------------------
# The destination's dtype has to reach the rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,call,shape",
    [
        ("concatenate", lambda o, a, b: fnp.concatenate([a, b], out=o), (128, 32)),
        ("stack", lambda o, a, b: fnp.stack([a, b], out=o), (2, 64, 32)),
        ("concat", lambda o, a, b: fnp.concat([a, b], out=o), (128, 32)),
    ],
)
def test_a_join_into_a_wider_destination_bills_the_destinations_loop(
    budget, name, call, shape
):
    """float32 blocks joined into a complex128 buffer convert on the way in.

    numpy's default casting for these is ``same_kind``, which admits
    float -> complex, so the copy loop that runs is the destination's. It used
    to bill as though the destination were not there — the same price as the
    all-float32 join, for a complex128 result.
    """
    a, b = _f32(64, 32), _f32(64, 32)
    native, _ = _billed(budget, lambda: call(fnp.zeros(shape, dtype="float32"), a, b))
    wider, _ = _billed(budget, lambda: call(fnp.zeros(shape, dtype="complex128"), a, b))

    # The reference is the same join with complex128 INPUTS: writing that many
    # complex128 elements costs what it costs, whichever end declares the dtype.
    ac, bc = fnp.astype(a, "complex128"), fnp.astype(b, "complex128")
    reference, _ = _billed(
        budget, lambda: call(fnp.zeros(shape, dtype="complex128"), ac, bc)
    )

    assert wider > native, f"{name}: a complex128 destination billed as float32"
    assert wider == reference, (
        f"{name}: complex128 destination billed {wider}, but the same write from "
        f"complex128 inputs bills {reference}"
    )


@pytest.mark.parametrize("op", ["isnan", "isinf", "isfinite"])
def test_a_predicate_into_a_wider_destination_bills_the_cast_it_performs(budget, op):
    """The predicates return bool, and numpy does NOT widen their loop.

    Every loop these ufuncs publish ends in ``?``, and
    ``np.isnan.resolve_dtypes((float32, float64))`` reports ``(float32, bool)``:
    the float32 loop runs and the bool answer is cast into the caller's buffer.
    The destination is a cast target, not a wider predicate.

    It is still charged, because that cast is real work flopscope already
    prices: over 4e6 float32 values, into bool 0.179 ms, into float64 1.314 ms,
    and the explicit two-step (predicate into bool, then ``astype(float64)``)
    1.269 ms — the fused spelling IS the two-step. Leaving the destination out
    of the rate handed back a free ``astype``.
    """
    call = getattr(fnp, op)
    x = _f32(64, 32)
    bool_dest = fnp.zeros((64, 32), dtype="bool")
    f64_dest = fnp.zeros((64, 32), dtype="float64")
    c128_dest = fnp.zeros((64, 32), dtype="complex128")
    # Same values, declared wide by the OPERAND instead of by the destination.
    x_f64 = fnp.astype(x, "float64")
    x_c128 = fnp.astype(x, "complex128")

    plain, _ = _billed(budget, lambda: call(x))
    natural, _ = _billed(budget, lambda: call(x, out=bool_dest))
    assert natural == plain, "a bool destination is the natural one; it must be free"

    # The rule, stated so it does not depend on the weights profile: the
    # destination widens the rate exactly as an equally wide OPERAND does —
    # "widest participating buffer", not "widest operand".
    for dest, wide_operand, label in (
        (f64_dest, x_f64, "float64"),
        (c128_dest, x_c128, "complex128"),
    ):
        via_dest, _ = _billed(budget, lambda d=dest: call(x, out=d))
        via_operand, _ = _billed(budget, lambda w=wide_operand: call(w))
        assert via_dest == via_operand, (
            f"{op} into a {label} destination billed {via_dest}, but the same "
            f"predicate over {label} operands bills {via_operand}"
        )

    # A strict increase, pinned on the complex destination rather than the
    # float64 one: the complex structure factor comes from the registry, not
    # the weights table, so this bites under any weights profile — including
    # the test profile, where float64 and float32 happen to share a rate and a
    # float64 destination legitimately costs the same as a bool one.
    c128_cost, _ = _billed(budget, lambda: call(x, out=c128_dest))
    assert c128_cost > natural, (
        f"{op} into a complex128 destination billed {c128_cost}, the same as "
        f"the bool destination — the cast into it is free"
    )


@pytest.mark.parametrize("op", ["take", "compress"])
def test_take_and_compress_cannot_reach_a_wider_destination_at_all(budget, op):
    """Why those two get no destination-dtype fold, unlike their siblings.

    numpy requires ``can_cast(out.dtype, a.dtype, "safe")`` for take/compress,
    so a wider destination is not a mispriced call — it is not a call. Only
    same-or-narrower destinations exist, and those never move the rate. This
    pins the premise; if numpy ever relaxes it, the fold becomes necessary and
    this test is what says so.
    """
    a = np.arange(64 * 32, dtype="float32").reshape(64, 32)
    wider = np.zeros((32, 32), dtype="float64")
    with pytest.raises(TypeError, match="Cannot cast"):
        if op == "take":
            np.take(a, np.arange(32), axis=0, out=wider)
        else:
            np.compress(np.array([True, False] * 32), a, axis=0, out=wider)


# ---------------------------------------------------------------------------
# outer: a symmetric destination used to be returned unwritten
# ---------------------------------------------------------------------------


def test_outer_writes_a_symmetric_destination_it_used_to_silently_skip(budget):
    """``fnp.zeros((n, n))`` IS a SymmetricTensor — a square constant fill picks
    up an inferred symmetry tag — so this is the obvious way to build a
    destination, not an exotic one.

    numpy was handed ``out=None`` for it (correct: a direct write would leave
    the tag standing over data numpy never showed us), but nothing then copied
    the result across when the result carried no symmetry of its own. The
    caller got their untouched destination back, having paid the whole
    contraction, with no exception raised.
    """
    dest = fnp.zeros((8, 8))
    # The premise of the whole test, asserted rather than assumed.
    assert isinstance(dest, SymmetricTensor) and dest.symmetry is not None
    a = fnp.asarray(np.arange(8.0))
    b = fnp.asarray(np.arange(8.0) + 1.0)

    result = fnp.outer(a, b, out=dest)

    assert result is dest
    assert np.allclose(np.asarray(dest), np.outer(np.arange(8.0), np.arange(8.0) + 1.0))
    # The inferred tag described the zeros; it must not survive the write.
    assert dest.symmetry is None


def test_outer_still_carries_a_real_symmetry_into_a_symmetric_destination(budget):
    dest = fnp.zeros((8, 8))
    assert isinstance(dest, SymmetricTensor)
    v = fnp.asarray(np.arange(8.0) + 1.0)

    result = fnp.outer(v, v, out=dest)

    assert result is dest
    assert np.allclose(
        np.asarray(dest), np.outer(np.arange(8.0) + 1, np.arange(8.0) + 1)
    )
    assert dest.symmetry is not None


def test_a_one_tuple_out_bills_like_a_bare_out_through_the_positional_slot(budget):
    # cumsum takes out= as its fourth POSITIONAL parameter, which reaches the
    # billing fold by a different channel (out_dtype=) than the keyword path.
    a = _f32(1000)

    def run(make_out):
        dest = fnp.zeros(1000, dtype="float64")
        return _billed(budget, lambda: fnp.cumsum(a, None, None, make_out(dest)))[0]

    assert run(lambda d: (d,)) == run(lambda d: d)


# ---------------------------------------------------------------------------
# The unwrapped destination is what comes back
# ---------------------------------------------------------------------------


def test_a_one_tuple_out_returns_the_destination_not_the_container(budget):
    dest = fnp.zeros(1000, dtype="complex128")
    assert (
        fnp.multiply(_f32(1000), _f32(1000), out=(dest,))  # pyright: ignore[reportArgumentType]
        is dest
    )

    fft_dest = fnp.zeros(64, dtype="complex128")
    assert fnp.fft.fft(_f32(64), out=(fft_dest,)) is fft_dest

    ein_dest = fnp.zeros((64, 32), dtype="complex128")
    result = fnp.einsum("ij,jk->ik", _f32(64, 64), _f32(64, 32), out=(ein_dest,))
    assert result is ein_dest


def test_a_routed_binary_returns_the_destinations_shape_not_the_containers(budget):
    # matvec used to do `result = out` with out still the tuple, so the caller
    # got back something of shape (1, 256) instead of (256,).
    dest = fnp.zeros(256, dtype="complex128")
    result = fnp.matvec(_f32(256, 256), _f32(256), out=(dest,))
    assert result.shape == (256,)
    assert result is dest


def test_a_none_holding_tuple_allocates_instead_of_destroying_the_answer(budget):
    # out=(None,) is numpy's "allocate this slot for me". The routed binaries
    # used to treat the tuple as the destination and hand back a shape-(1,)
    # object array — the answer silently gone, no exception, full price.
    result = fnp.matvec(_f32(256, 256), _f32(256), out=(None,))
    assert result.shape == (256,)
    assert result.dtype != np.dtype(object)


# ---------------------------------------------------------------------------
# Everything else is refused, and refusal is free
# ---------------------------------------------------------------------------

_Pair = collections.namedtuple("_Pair", "first")


class _TupleSubclass(tuple):
    pass


def _bad_forms(dest):
    return {
        "list": [dest],
        "two-tuple": (dest, dest),
        "nested-list": [[0.0, 0.0]],
        "empty-tuple": (),
        "string": "dest",
        "memoryview": memoryview(np.zeros(2).tobytes()),
        # numpy refuses both of these, so `type(out) is tuple` (not isinstance)
        # is what keeps flopscope from being quietly more permissive.
        "namedtuple": _Pair(dest),
        "tuple-subclass": _TupleSubclass((dest,)),
    }


#: numpy distinguishes the two: a tuple of the wrong LENGTH is a ValueError
#: ("The 'out' tuple must have exactly one entry per ufunc output"), anything
#: of the wrong TYPE is a TypeError ("return arrays must be of ArrayType").
_WRONG_LENGTH = {"two-tuple", "empty-tuple"}


@pytest.mark.parametrize("form", sorted(_bad_forms(None)))
def test_a_refused_out_form_costs_nothing_on_every_path(budget, form):
    # Every array is built BEFORE the measurement starts. Constructing one
    # costs FLOPs of its own, and building inside the measured region is how
    # a "refusal is free" assertion quietly measures the wrong thing.
    a2 = fnp.array([1.0, 2.0])
    b2 = fnp.array([3.0, 4.0])
    eye = fnp.array([[1.0, 0.0], [0.0, 1.0]])

    cases = [
        ("multiply", lambda o: fnp.multiply(a2, b2, out=o), fnp.zeros(2)),
        ("matvec", lambda o: fnp.matvec(eye, a2, out=o), fnp.zeros(2)),
        ("einsum", lambda o: fnp.einsum("i,i->i", a2, b2, out=o), fnp.zeros(2)),
        ("cumsum-positional", lambda o: fnp.cumsum(a2, None, None, o), fnp.zeros(2)),
    ]
    for name, call, dest in cases:
        bad = _bad_forms(dest)[form]
        before = budget.flops_used
        expected = ValueError if form in _WRONG_LENGTH else TypeError
        with pytest.raises(expected):
            call(bad)
        assert budget.flops_used == before, (
            f"{name} was billed for refusing out={form}; a refusal must be free"
        )
        assert np.array_equal(np.asarray(dest), np.zeros(2)), (
            f"{name} wrote to the destination while refusing out={form}"
        )


def test_einsum_refuses_the_container_forms_it_used_to_mis_write(budget):
    # All of these were accepted, billed in full, and left dest untouched,
    # because einsum copies into out itself rather than handing it to numpy.
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = fnp.array([[5.0, 6.0], [7.0, 8.0]])
    bad_forms = ([fnp.zeros((2, 2))], [fnp.zeros((2, 2)), fnp.zeros((2, 2))], ())

    for bad in bad_forms:
        dest_before = budget.flops_used
        expected = ValueError if isinstance(bad, tuple) else TypeError
        with pytest.raises(expected):
            fnp.einsum("ij,jk->ik", a, b, out=bad)
        assert budget.flops_used == dest_before


def test_a_list_out_is_refused_and_costs_nothing(budget):
    # numpy accepts a list in none of its out= surfaces, so neither do we.
    # The cost assertion is the part that used to fail: the guard ran inside
    # the deduct block, so multiply billed in full and then raised.
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    dest = fnp.array([0.0, 0.0])
    before = budget.flops_used

    with pytest.raises(TypeError, match="out= must be an array"):
        fnp.multiply(a, b, out=[dest])  # pyright: ignore[reportArgumentType]

    assert budget.flops_used == before
    assert np.array_equal(np.asarray(dest), np.zeros(2))


# ---------------------------------------------------------------------------
# Multi-output out= is a different protocol and must not be unwrapped
# ---------------------------------------------------------------------------


def test_multi_output_out_tuples_are_left_alone(budget):
    frac = fnp.zeros(4)
    whole = fnp.zeros(4)
    result = fnp.modf(fnp.array([1.5, 2.25, -3.75, 0.5]), out=(frac, whole))

    assert result[0] is frac and result[1] is whole
    assert np.asarray(whole).tolist() == [1.0, 2.0, -3.0, 0.0]


def test_a_wrong_length_multi_output_tuple_is_refused_for_free(budget):
    src = fnp.array([1.5, 2.5])
    dest = fnp.zeros(2)
    before = budget.flops_used
    with pytest.raises(ValueError, match="exactly one entry per ufunc output"):
        fnp.modf(src, out=(dest,))  # pyright: ignore[reportArgumentType]
    assert budget.flops_used == before


# ---------------------------------------------------------------------------
# Nothing that already worked stops working
# ---------------------------------------------------------------------------


def test_einsum_with_a_real_out_still_works(budget):
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = fnp.array([[5.0, 6.0], [7.0, 8.0]])
    dest = fnp.array([[0.0, 0.0], [0.0, 0.0]])

    result = fnp.einsum("ij,jk->ik", a, b, out=dest)

    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert np.asarray(result).tolist() == expected
    assert np.asarray(dest).tolist() == expected, "out= must receive the result"
    assert result is dest, "out= must return the destination itself"


def test_out_still_works_on_the_routed_binaries(budget):
    """vecdot/matvec/vecmat expose out= publicly; they must keep working."""
    eye = fnp.array([[1.0, 0.0], [0.0, 1.0]])
    v = fnp.array([3.0, 4.0])

    dest = fnp.array([0.0, 0.0])
    assert np.asarray(fnp.matvec(eye, v, out=dest)).tolist() == [3.0, 4.0]

    scalar_dest = fnp.array(0.0)
    assert float(np.asarray(fnp.vecdot(v, v, out=scalar_dest))) == 25.0


def test_a_flopscope_array_inside_a_tuple_is_accepted(budget):
    # The destination a participant actually holds is a FlopscopeArray, and
    # wrapping one used to trip an internal strip check rather than working.
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    dest = fnp.zeros(2)

    result = fnp.multiply(a, b, out=(dest,))  # pyright: ignore[reportArgumentType]

    assert result is dest
    assert np.asarray(dest).tolist() == [3.0, 8.0]


def test_out_none_and_absent_are_unaffected(budget):
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    assert np.asarray(fnp.multiply(a, b)).tolist() == [3.0, 8.0]
    assert np.asarray(fnp.multiply(a, b, out=None)).tolist() == [3.0, 8.0]
