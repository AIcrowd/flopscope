"""An index reduction's ``out=`` is refused for free and never sets the rate.

``argmin``/``argmax``/``nanargmin``/``nanargmax`` return POSITIONS. numpy
fixes their result dtype at ``np.intp`` regardless of the input's precision
and constrains ``out=`` by KIND rather than by width: every integer or
boolean buffer is accepted at any width, every float and complex one is
refused outright, before the reduction runs.

Two things followed from charging anyway. ``argmin`` on 10,000 float32 with
a float64 destination billed 19,998 FLOPs and only then raised -- a refused
form is supposed to cost zero, and the guard has to sit ABOVE
``budget.deduct`` for that (there are no refunds; see
``tests/test_budget.py``). And 19,998 is twice the 9,999 the bare call
costs, because an index buffer that cannot be the accumulator was widening
the rate: supplying the ``intp`` destination numpy would have allocated
anyway must be price-neutral.
"""

import inspect

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._pointwise import _INDEX_RETURNING_REDUCTIONS
from flopscope._registry import REGISTRY
from flopscope._weights import load_weights

N = 10_000
OPS = sorted(_INDEX_RETURNING_REDUCTIONS)


def _billed(fn):
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.fixture(autouse=True)
def _weights():
    load_weights()


def _f32():
    return np.random.default_rng(0).standard_normal(N).astype(np.float32)


def _expected_bare_cost(op_name: str) -> int:
    # nanargmin/nanargmax run the #177.4 isnan pass in addition to the
    # N-1 scan their plain argmin/argmax sibling runs.
    base = N - 1
    return base + N if op_name.startswith("nan") else base


@pytest.mark.parametrize("op_name", OPS)
def test_refused_destination_costs_zero(op_name):
    a = _f32()
    fs_func = getattr(fnp, op_name)
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        with pytest.raises(TypeError):
            fs_func(a, out=np.empty((), np.float64))
        assert b.flops_used == 0  # was 19,998, charged then raised


@pytest.mark.parametrize("op_name", OPS)
def test_index_destination_prices_exactly_like_the_bare_call(op_name):
    a = _f32()
    fs_func = getattr(fnp, op_name)
    bare = _billed(lambda: fs_func(a))
    with_out = _billed(lambda: fs_func(a, out=np.empty((), np.intp)))
    expected = _expected_bare_cost(op_name)
    assert bare == with_out == expected  # was 9,999 vs 19,998


@pytest.mark.parametrize("op_name", OPS)
def test_narrow_index_destination_does_not_discount_either(op_name):
    a = _f32()
    fs_func = getattr(fnp, op_name)
    assert _billed(lambda: fs_func(a, out=np.empty((), np.int8))) == (
        _expected_bare_cost(op_name)
    )


@pytest.mark.parametrize("op_name", OPS)
def test_one_tuple_out_bills_what_the_bare_destination_bills(op_name):
    a = _f32()
    fs_func = getattr(fnp, op_name)
    dest = np.empty((), np.intp)
    assert _billed(lambda: fs_func(a, out=(dest,))) == _billed(
        lambda: fs_func(a, out=dest)
    )


@pytest.mark.parametrize("op_name", OPS)
def test_refusal_is_free_through_the_positional_out_slot_too(op_name):
    """``argmin(a, None, dest)`` reaches ``out`` by position, not keyword."""
    a = _f32()
    fs_func = getattr(fnp, op_name)
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        with pytest.raises(TypeError):
            fs_func(a, None, np.empty((), np.float64))
        assert b.flops_used == 0


@pytest.mark.parametrize("op_name", OPS)
def test_refusal_is_free_through_the_method_surface_too(op_name):
    a = fnp.asarray(_f32())
    method = getattr(a, op_name, None)
    if method is None:  # nanarg* have no ndarray method
        pytest.skip(f"ndarray has no .{op_name}()")
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        with pytest.raises(TypeError):
            method(out=np.empty((), np.float64))
        assert b.flops_used == 0


@pytest.mark.parametrize("op_name", OPS)
def test_per_axis_form_matches_the_bare_per_axis_form(op_name):
    a = np.random.default_rng(1).standard_normal((100, 100)).astype(np.float32)
    fs_func = getattr(fnp, op_name)
    bare = _billed(lambda: fs_func(a, axis=0))
    with_out = _billed(lambda: fs_func(a, axis=0, out=np.empty(100, np.intp)))
    assert bare == with_out


@pytest.mark.parametrize("op_name", OPS)
def test_destination_still_receives_the_index(op_name):
    """Free refusal must not have cost the accepted path its write."""
    a = _f32()
    fs_func = getattr(fnp, op_name)
    dest = np.empty((), np.intp)
    with f.BudgetContext(flop_budget=10**15, quiet=True):
        result = fs_func(a, out=dest)
    assert int(dest) == int(getattr(np, op_name)(a))
    assert int(result) == int(dest)


# ---------------------------------------------------------------------------
# Accept / refuse parity with numpy, and the membership derivation.
# ---------------------------------------------------------------------------

_PROBE_DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "O",
    "U5",
    "datetime64[s]",
    "timedelta64[s]",
]


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype_name", _PROBE_DTYPES)
@pytest.mark.parametrize("axis", [None, 0])
def test_guard_accepts_and_refuses_exactly_what_numpy_does(op_name, dtype_name, axis):
    """A pre-dispatch guard that is stricter than numpy is a regression.

    The rule numpy actually applies is ``can_cast(out.dtype, intp, "safe")``:
    bool and every integer narrower than ``intp`` go through (numpy casts the
    index down on store), ``uint64`` and everything inexact or non-numeric do
    not. This sweep is what pins the guard to it.
    """
    a = np.random.default_rng(2).standard_normal((4, 5)).astype(np.float32)
    shape = () if axis is None else (5,)
    kwargs = {} if axis is None else {"axis": axis}
    dtype = np.dtype(dtype_name)

    try:
        getattr(np, op_name)(a, out=np.empty(shape, dtype), **kwargs)
        numpy_accepts = True
    except Exception:
        numpy_accepts = False

    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        try:
            getattr(fnp, op_name)(a, out=np.empty(shape, dtype), **kwargs)
            flopscope_accepts = True
        except Exception:
            flopscope_accepts = False
        used = b.flops_used

    assert flopscope_accepts == numpy_accepts, (
        f"{op_name} out={dtype_name} axis={axis}: "
        f"numpy {'accepts' if numpy_accepts else 'refuses'}, "
        f"flopscope {'accepts' if flopscope_accepts else 'refuses'}"
    )
    if not numpy_accepts:
        assert used == 0


def _derive_index_returning_reductions():
    """Re-derive the family from the registry by asking numpy, not by memory.

    An op qualifies when, on a float input, numpy hands back an INTEGER
    result (an index, not a value) and constrains ``out=`` by kind: a
    strictly wider float destination refused, an integer one accepted. That
    last clause is what keeps ``compress`` -- which also refuses a wider
    destination, but returns values of the input's own dtype -- out of the
    set.
    """
    probe = np.random.default_rng(3).standard_normal(6).astype(np.float32)
    derived = set()
    for name, entry in REGISTRY.items():
        if entry.get("module") != "numpy":
            continue
        np_func = getattr(np, name, None)
        if np_func is None or not callable(np_func):
            continue
        if not isinstance(np_func, np.ufunc):
            try:
                if "out" not in inspect.signature(np_func).parameters:
                    continue
            except (TypeError, ValueError):
                continue
        natural = None
        accepted_args: tuple = ()
        for args in ((probe,), (probe, probe)):
            try:
                with np.errstate(all="ignore"):
                    natural = np.asarray(np_func(*args))
                accepted_args = args
                break
            except Exception:
                continue
        if natural is None or natural.dtype.kind not in "iu":
            continue

        def _accepts(dtype, _np_func=np_func, _args=accepted_args, _natural=natural):
            try:
                with np.errstate(all="ignore"):
                    _np_func(*_args, out=np.empty(_natural.shape, dtype))
                return True
            except Exception:
                return False

        if not _accepts(np.intp) or _accepts(np.float64):
            continue
        derived.add(name)
    return derived


def test_index_returning_family_is_complete():
    """The frozenset in ``_pointwise`` must be what numpy says it is."""
    derived = _derive_index_returning_reductions()
    assert derived == set(_INDEX_RETURNING_REDUCTIONS)
    # The probe demonstrably reaches the four -- an empty derivation would
    # otherwise pass vacuously if the set were ever emptied.
    assert derived == {"argmin", "argmax", "nanargmin", "nanargmax"}
