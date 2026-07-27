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
