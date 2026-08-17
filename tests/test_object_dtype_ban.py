"""The dtype ban: flopscope refuses any non-numeric dtype.

The predicate is a NUMERIC ALLOWLIST (``dtype.kind in "biufc"``: bool,
signed/unsigned integer, float, complex), not a denylist of the non-numeric
kinds. Two families fall outside it, refused for different reasons: object
(``'O'``) carries unbounded per-element Python cost that no rate can price;
str/bytes/structured-void/datetime64/timedelta64 (``'USVMm'``) are bounded
but their real per-element cost is not the fixed unit a flat rate assumes.
Both are refused outright rather than mis-priced.
"""

import datetime
import multiprocessing

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope import BudgetContext
from flopscope._dtype_billing import refuse_non_numeric_dtype
from flopscope.errors import UnsupportedDtypeError

requires_vecmat = pytest.mark.skipif(
    not hasattr(np, "vecmat"), reason="requires numpy >= 2.2"
)


def test_plain_object_dtype_is_refused():
    with pytest.raises(UnsupportedDtypeError, match="is not billable"):
        refuse_non_numeric_dtype("multiply", np.dtype(object))


def test_structured_dtype_embedding_object_is_refused():
    """kind is 'V', not 'O' -- a kind=='O' check would miss this entirely,
    but 'V' is outside the numeric allowlist regardless, so this is refused
    the same way as any other structured dtype."""
    dt = np.dtype([("a", object), ("b", "f8")])
    assert dt.kind == "V" and dt.hasobject
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("sort", dt)


def test_subarray_dtype_embedding_object_is_refused():
    dt = np.dtype([("a", object, (2,))])
    assert dt.hasobject
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("sort", dt)


def test_object_free_record_dtype_is_also_refused():
    """Structured/void ('V') is outside the numeric allowlist unconditionally
    -- unlike the old hasobject-only ban, an object-free record dtype is no
    longer a carve-out. Its real per-element cost (comparing/copying a wide
    record) is not the flat rate the ban would otherwise have to assume."""
    dt = np.dtype([("mu", "f8"), ("var", "f8")])
    assert not dt.hasobject
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("sort", dt)


@pytest.mark.parametrize(
    "dt", ["bool", "int32", "uint8", "float64", "float32", "complex128"]
)
def test_numeric_dtypes_are_allowed(dt):
    refuse_non_numeric_dtype("multiply", np.dtype(dt))  # must not raise


@pytest.mark.parametrize("dt", ["<U8", "S4", "M8[s]", "m8[s]", "V8"])
def test_non_numeric_dtypes_are_refused(dt):
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("multiply", np.dtype(dt))


def test_python_scalars_are_ignored():
    """billing_operand passes Python scalars through for NEP 50 weak promotion;
    np.dtype(1.0) raises TypeError, so they must be skipped, not crash."""
    refuse_non_numeric_dtype("multiply", 1.0, True, 3, None)  # must not raise


@pytest.mark.parametrize("dt", [np.dtype([]), np.dtype("V0"), np.dtype("V")])
def test_zero_itemsize_non_numeric_dtypes_are_allowed(dt):
    """Zero bytes per element cannot embed an object field or any
    itemsize-dependent cost, so a zero-itemsize dtype is let through
    regardless of kind -- NumPy's own ``broadcast_shapes`` allocates
    ``np.empty(shape, dtype=np.dtype([]))`` internally as a zero-byte
    shape-computation placeholder, so this dtype must not be refused as if
    a caller had asked to compute something in it.

    The carve-out is measured against what numpy MATERIALISES, not against
    what was requested -- see
    ``test_zero_length_string_dtypes_are_refused_because_numpy_promotes_them``
    below for the family that requests zero bytes and gets more. Void is the
    kind that genuinely keeps its zero itemsize through construction."""
    assert dt.itemsize == 0
    assert np.empty(0, dtype=dt).dtype.itemsize == 0  # numpy keeps it at zero
    refuse_non_numeric_dtype("empty", dt)  # must not raise


@pytest.mark.parametrize("spec", ["U0", "S0", "U", "S", np.str_, np.bytes_])
def test_zero_length_string_dtypes_are_refused_because_numpy_promotes_them(spec):
    """The zero-itemsize carve-out must read the MATERIALISED dtype.

    ``np.dtype('U0').itemsize`` is 0, so a guard written on the requested
    dtype let it through -- but numpy promotes an unsized/zero-length string
    dtype to a one-character one the moment it actually allocates, so
    ``fnp.zeros(1000, dtype='U0')`` handed back a real 4000-byte ``<U1``
    array for free. Zero bytes per element was the whole justification for
    the carve-out, and this family never gets zero bytes per element.

    ``np.dtype(np.str_)``/``np.dtype(np.bytes_)`` are the same unsized
    spellings, and are what ``refuse_non_numeric_source`` and the operand
    scan's leaf check hand in as the representative dtype for a bare
    ``str``/``bytes`` payload -- so this is also what makes those refusals
    fire."""
    dt = np.dtype(spec)
    assert dt.itemsize == 0  # requested: zero bytes
    assert np.empty(0, dtype=dt).dtype.itemsize != 0  # materialised: not zero
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("empty", dt)


@pytest.mark.parametrize("spec", ["U0", "S0"])
def test_zero_length_string_creation_is_refused_end_to_end(spec):
    """The reported repro: a free, 0-FLOP creation op manufacturing a real
    string array. ``fnp.zeros(1000, dtype='U0')`` billed 0 FLOPs and returned
    4000 bytes of ``<U1``."""
    with BudgetContext(BUDGET, quiet=True) as bc:
        with pytest.raises(UnsupportedDtypeError):
            fnp.zeros(1000, dtype=spec)
        assert bc.flops_used == 0


def test_zero_itemsize_void_creation_still_works():
    """The documented carve-out must keep working end to end: numpy's own
    zero-byte placeholder stays zero-byte through construction, so it is not
    a way to get free string storage."""
    with BudgetContext(BUDGET, quiet=True) as bc:
        result = fnp.zeros(1000, dtype="V0")
        assert result.nbytes == 0
        assert bc.flops_used == 0


def test_nonzero_itemsize_structured_dtype_is_still_refused():
    """The zero-itemsize exception must not widen into a blanket structured
    carve-out: a structured dtype with a real (non-zero-itemsize) field is
    refused exactly as before."""
    nonzero = np.dtype("V8")
    assert nonzero.itemsize == 8
    with pytest.raises(UnsupportedDtypeError):
        refuse_non_numeric_dtype("empty", nonzero)


def test_error_message_names_op_and_gives_the_numpy_remedy():
    with pytest.raises(UnsupportedDtypeError) as exc:
        refuse_non_numeric_dtype("multiply", np.dtype(object))
    msg = str(exc.value)
    assert "multiply" in msg
    assert "np.array(x, dtype=np.float64)" in msg  # remedy is RAW numpy
    # fnp conversion ops also refuse non-numeric input -- the message must
    # warn of that, not stay silent and let a reader reach for fnp.array next.
    assert "fnp.array" in msg and "refuse non-numeric input too" in msg


BUDGET = int(1e12)


def _obj_cells(n=1, size=64):
    o = np.empty(n, dtype=object)
    for i in range(n):
        o[i] = np.ones((size, size))
    return o


def test_the_reported_repro_now_raises():
    """The originally reported repro must now be refused."""
    c = np.empty(1, dtype=object)
    c[0] = np.ones((800, 800))
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(fnp.asarray(c), fnp.asarray(c))


def test_object_out_destination_raises():
    a = np.ones(64)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(a, a, out=np.empty(64, dtype=object))  # type: ignore[arg-type]


def test_object_compute_dtype_raises():
    """The remotely-reachable variant: an explicit dtype='object' request,
    not an object-dtype operand."""
    a = np.ones((32, 32))
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.sum(a, dtype="object")


def test_conversion_ops_refuse_too():
    """object->numeric casting calls float() per cell, which is arbitrary
    participant code that must never run before the ban's check."""
    calls = {"n": 0}

    class Payload:
        def __float__(self):
            calls["n"] += 1
            return 1.0

    o = np.empty(4, dtype=object)
    for i in range(4):
        o[i] = Payload()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.array(o, dtype=np.float64)
    assert calls["n"] == 0, "payload ran before the refusal"


class _CountingPayload:
    """Object whose ``__float__`` counts invocations -- lets a test confirm
    the conversion-op gap below never actually ran participant code."""

    def __init__(self):
        self.float_calls = 0

    def __float__(self):
        self.float_calls += 1
        return 1.0


def _object_list(n=4):
    """A plain Python list of objects -- the LIST spelling."""
    return [_CountingPayload() for _ in range(n)]


def _object_ndarray(n=4):
    """A dtype=object ndarray holding the same objects -- the NDARRAY
    spelling. Returns (array, payloads) so callers can inspect the payloads
    directly regardless of which spelling was used."""
    payloads = [_CountingPayload() for _ in range(n)]
    o = np.empty(n, dtype=object)
    for i, p in enumerate(payloads):
        o[i] = p
    return o, payloads


def _object_source(source_kind):
    """Return (source, payloads) for the given spelling ("list"/"ndarray")."""
    if source_kind == "list":
        payloads = _object_list()
        return payloads, payloads
    source, payloads = _object_ndarray()
    return source, payloads


@pytest.mark.parametrize("source_kind", ["list", "ndarray"])
def test_array_refuses_object_source_both_spellings(source_kind):
    """Regression: ``fnp.array(<LIST of objects>, dtype=<numeric>)`` was not
    refused -- the cost probe performed the object->numeric cast itself
    (``np.asarray(object, dtype=dtype)``), calling ``__float__`` on every
    cell, so by the time the ban's check ran, the probe's own (already
    coerced) dtype had nothing object-shaped left to catch. The NDARRAY
    spelling was already refused by a different code path (the
    ``_refuse_non_numeric_operands`` backstop) -- the pre-existing regression
    test (``test_conversion_ops_refuse_too`` above) only covered that
    spelling, which is why the LIST spelling needed its own coverage."""
    source, payloads = _object_source(source_kind)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.array(source, dtype=np.float64)  # type: ignore[arg-type]
    assert sum(p.float_calls for p in payloads) == 0, "payload ran before the refusal"


@pytest.mark.parametrize("source_kind", ["list", "ndarray"])
def test_asarray_refuses_object_source_both_spellings(source_kind):
    """Same gap as above, for ``fnp.asarray``."""
    source, payloads = _object_source(source_kind)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.asarray(source, dtype=np.float64)  # type: ignore[arg-type]
    assert sum(p.float_calls for p in payloads) == 0, "payload ran before the refusal"


@pytest.mark.parametrize("source_kind", ["list", "ndarray"])
def test_astype_refuses_object_source_both_spellings(source_kind):
    """Same gap as above, for ``fnp.astype``."""
    source, payloads = _object_source(source_kind)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.astype(source, np.float64)  # type: ignore[arg-type]
    assert sum(p.float_calls for p in payloads) == 0, "payload ran before the refusal"


def test_ragged_object_array_raises_the_ban_error_not_a_value_error():
    """fnp.array on ragged, explicitly object-dtyped input must surface the
    ban's UnsupportedDtypeError, not numpy's raw "inhomogeneous shape"
    ValueError.

    With the common `dtype=object` spelling used here, this does not
    exercise the array()-internal probe fix at _array_ops.py: the
    `_counted_wrapper` operand backstop (`_refuse_non_numeric_operands` in
    _budget.py) already resolves `dtype=object` via `_plain_dtype_like`
    and raises before array()'s body -- and therefore before the probe --
    ever runs.

    That does not mean the probe fix is unreachable through the public API
    -- see test_duck_typed_dtype_reaches_the_probe_and_is_refused below for
    the scenario that actually pins it: a dtype-LIKE object whose `.dtype`
    is a property rather than one of _plain_dtype_like's cheap forms.
    """
    ragged = [np.ones(2), np.ones(3)]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.array(ragged, dtype=object)


def test_duck_typed_dtype_reaches_the_probe_and_is_refused():
    """The scenario that pins the array()-internal probe fix at
    _array_ops.py: a dtype-LIKE object exposing a `.dtype` PROPERTY rather
    than being one of `_plain_dtype_like`'s cheap forms (an `np.dtype`
    instance, str/bytes, a `type`, `None`, or a list/tuple structured spec
    built entirely from such inert leaves).

    `_refuse_non_numeric_operands`'s dtype-kwarg check
    (`_resolve_dtype_kwarg_value` + `_plain_dtype_like`) only resolves
    those cheap forms, deliberately -- it must never touch an
    arbitrary object's `.dtype` PROPERTY, since that would run whatever
    code answers it as a side effect of the ban's own check (see
    test_operand_scan_never_executes_a_dtype_property above). The
    backstop's separate operand-scanning loop (`check()`) also explicitly
    skips the `"dtype"` kwarg. So an object like the one below reaches
    `array()`'s body unblocked by the backstop.

    numpy itself accepts such an object: `np.dtype()` documents a
    duck-typing protocol ("Any type object with a dtype attribute") and
    genuinely reads `.dtype` to resolve it -- this is numpy doing its own,
    real work, not flopscope speculatively probing an untrusted property.
    """

    class _DtypeLikeObject:
        @property
        def dtype(self):
            return np.dtype(object)

    ragged = [np.ones(2), np.ones(3)]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.array(ragged, dtype=_DtypeLikeObject())


def test_nothing_is_charged_when_refused():
    """Fail closed means fail free: no FLOPs recorded for a refused op."""
    o = _obj_cells()
    with BudgetContext(BUDGET, quiet=True) as bc:
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(fnp.asarray(o), fnp.asarray(o))
        assert bc.flops_used == 0


def test_ban_fires_in_unit_mode():
    """Guards the get_dtype_rate trap: with an empty rate table the ban must
    still fire. conftest resets weights for every test, so if the ban were
    table-expressed this whole suite would be testing nothing."""
    from flopscope._weights import reset_weights

    reset_weights()
    o = _obj_cells()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(fnp.asarray(o), fnp.asarray(o))


def test_numeric_ops_are_unaffected():
    """1 FLOP/element at rate 1.0 -- the autouse conftest fixture resets
    weights to unit mode for every test (see test_ban_fires_in_unit_mode),
    so float64 does not carry its real-weights 2.0 dtype rate here."""
    a = np.ones(1000)
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.multiply(a, a)
        assert bc.flops_used == 1000


def test_raw_numpy_remediation_works():
    """The documented fix: convert outside the meter, then hand it in."""
    o = np.array([1.0, None, 3.0], dtype=object)
    clean = np.array(o, dtype=np.float64)
    assert np.isnan(clean[1])
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.multiply(fnp.asarray(clean), fnp.asarray(clean))
        assert bc.flops_used == 3


def test_astype_refuses_object_source_under_real_weights():
    """astype bills the heavier of source/destination rate as a SINGLE
    winning dtype (_heavier_billing_dtype), which would silently drop an
    object source whenever the destination's real-weights rate beats
    object's floor rate of 1.0 -- float64 rates 2.0. Unit mode (rate 1.0
    everywhere) hides this: the tie always resolves to the first-listed
    (object) dtype, so the ban suite's ambient unit-mode reset (see
    test_ban_fires_in_unit_mode) would never exercise this path."""
    from flopscope._weights import load_weights

    load_weights()
    o = np.empty(3, dtype=object)
    o[:] = [1.0, 2.0, 3.0]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.astype(o, np.float64)


def test_astype_method_refuses_object_source_under_real_weights():
    """Same gap, the ndarray .astype() method's counted backend."""
    from flopscope._array_ops import _astype_counted
    from flopscope._weights import load_weights

    load_weights()
    o = np.empty(3, dtype=object)
    o[:] = [1.0, 2.0, 3.0]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            _astype_counted(o, np.float64)


# --- reduction family (sum/prod/mean/cumsum/...): reduction_billing_dtype ---
#
# reduction_billing_dtype() had the same "fold two dtypes into one winner by
# rate, silently drop the loser" defect as astype(), in two places: the
# out_dtype fold and the explicit_dtype branch (which returns the requested
# dtype outright, never consulting the operand). Unlike astype's gap, both
# are structural rather than rate-dependent, so the parametrized tests below
# need no load_weights() to catch them.


def _obj_row(n=4):
    o = np.empty(n, dtype=object)
    o[:] = [1.0, 2.0, 3.0, 4.0][:n]
    return o


@pytest.mark.parametrize("op", ["sum", "mean", "cumsum"])
def test_reduction_out_destination_raises(op):
    """out_dtype's fold into the accumulator (heavier_billing_dtype(floor,
    out_dtype)) put the numeric floor first, so a tie (or a numeric floor
    outranking object outright) drops the object destination from the
    billing tuple before it ever reaches deduct() -- discount without a
    raise, e.g. fnp.sum(float64_arr, out=object_arr) billed as if no
    destination were supplied at all."""
    a = np.ones(9)
    call = {
        "sum": lambda: fnp.sum(a, out=np.empty((), dtype=object)),
        "mean": lambda: fnp.mean(a, out=np.empty((), dtype=object)),  # type: ignore[arg-type]
        "cumsum": lambda: fnp.cumsum(a, out=np.empty(9, dtype=object)),
    }[op]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call()


@pytest.mark.parametrize("op", ["sum", "mean", "cumsum"])
def test_reduction_explicit_dtype_does_not_launder_object_operand(op):
    """reduction_billing_dtype's explicit_dtype branch returned
    np.dtype(explicit_dtype) outright and never looked at the operand at
    all -- the same gap as array()'s conversion-op probe. An object operand
    with a numeric dtype= (e.g. fnp.sum(object_arr, dtype=np.float64)) ran
    real per-cell Python (float() on every element) while billing as though
    the input were already the requested numeric dtype."""
    call = {
        "sum": lambda: fnp.sum(_obj_row(), dtype=np.float64),
        "mean": lambda: fnp.mean(_obj_row(), dtype=np.float64),
        "cumsum": lambda: fnp.cumsum(_obj_row(), dtype=np.float64),
    }[op]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call()


def test_reduction_gaps_also_fail_under_real_weights():
    """Both reduction gaps above are structural (a first-argument tie-break
    and an unconditional early return), not rate-dependent, so they are
    unaffected by which dtype-rate table is active -- unlike the astype gap
    earlier in this file, which unit mode actively hides. Checked here under
    real production weights so a future reader does not assume every gap in
    this family behaves like astype's and simplify this coverage down to
    unit-mode only."""
    from flopscope._weights import load_weights

    load_weights()
    a = np.ones(9)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.sum(a, out=np.empty((), dtype=object))
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.sum(_obj_row(), dtype=np.float64)


# --- dtype-neutral ops (dtypes=() at the deduct()/deduct_after() call site):
# the operand-side backstop in _counted_wrapper (_budget.py) --------------
#
# deduct()'s own check only sees dtypes a wrapper chose to DECLARE, and a
# whole family of ops -- pure data movement whose cost genuinely does not
# depend on dtype -- declares dtypes=(). That spans array-shape movement
# (reshape/transpose/.../diagonal family), array creation via a dtype=
# argument (zeros/empty/asarray/.../stack), and the random choice/
# permutation/shuffle movement family (module-level and the Generator/
# RandomState class-method factory in _counted_classes.py, which sets
# billing_dtypes=() unconditionally for its whole `_movement_methods` set).
# Two more (matrix_transpose, permute_dims) don't even go through
# @_counted_wrapper at all, so they needed a direct in-function check
# instead of the wrapper backstop.


@pytest.mark.parametrize(
    "call",
    [
        lambda o: fnp.reshape(o, (2, 2)),
        lambda o: fnp.copy(o),
        lambda o: fnp.concatenate([o, o]),
        lambda o: fnp.take(o, [0, 1]),
        lambda o: fnp.transpose(o),
        lambda o: fnp.squeeze(o),
        lambda o: fnp.flip(o),
        lambda o: fnp.atleast_2d(o),
        lambda o: fnp.diagonal(o),
        lambda o: fnp.expand_dims(o, axis=0),
        lambda o: fnp.matrix_transpose(o),
        lambda o: fnp.linalg.matrix_transpose(o),
        lambda o: fnp.permute_dims(o),
    ],
)
def test_dtype_neutral_ops_also_refuse_object(call):
    """Pure data-movement ops relocate values without touching them, and
    their cost model is genuinely dtype-independent -- but an object array
    reaching them still carries unbounded per-cell work through whatever
    numpy operation the CALLER runs on the result next, and every one of
    them silently returned an object-dtype result before this fix."""
    o = _obj_cells(n=4, size=8).reshape(2, 2)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call(o)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fnp.zeros((3, 3), dtype=object),
        lambda: fnp.zeros((3, 3), object),  # positional dtype slot
        lambda: fnp.empty((3, 3), dtype=object),
        lambda: fnp.empty((3, 3), "O"),  # positional, dtype-code string
        lambda: fnp.zeros_like(np.ones((3, 3)), dtype=object),
        lambda: fnp.empty_like(np.ones((3, 3)), dtype=object),
        lambda: fnp.asarray(np.ones((3, 3)), dtype=object),
        lambda: fnp.stack([np.ones(3), np.ones(3)], dtype=object),
        lambda: fnp.concatenate([np.ones(3), np.ones(3)], dtype=object),
        pytest.param(
            lambda: fnp.vecmat(np.ones(3), np.ones(3), dtype=object),
            marks=requires_vecmat,
        ),
    ],
)
def test_creation_ops_refuse_a_dtype_object_request(call):
    """A dtype= request for object output is a second, distinct way to
    reach an object array through flopscope: no object operand is involved
    at all, the caller just asks a creation/combination op to manufacture
    one directly. ``zeros``/``empty`` are 0-FLOP "free" ops that never
    consult dtype= for billing in the first place, so they had no dtype
    check to bypass; both the keyword and positional dtype= forms must be
    refused."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: fnp.zeros(2, dtype=[("a", object)]),
        lambda: fnp.zeros(2, dtype=[("a", [("b", object)])]),
        lambda: fnp.zeros(2, dtype=np.dtype([("a", object)])),
        lambda: fnp.zeros(2, dtype="O"),
    ],
)
def test_structured_dtype_spec_spellings_all_refused(call):
    """The same object-embedding structured dtype, requested through four
    different dtype= spellings, must be refused identically: a list/tuple
    field spec, one nested a level deep, the equivalent np.dtype(...), and
    a bare dtype-code string. `_plain_dtype_like` previously resolved only
    the np.dtype/string form; the list/tuple forms bypassed it entirely."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call()


def test_object_free_structured_dtype_spec_is_also_refused():
    """Widening `_plain_dtype_like` to resolve list/tuple dtype specs closes
    a gap for every structured request, object-free ones included: void
    ('V') is outside the numeric allowlist unconditionally, so this dtype=
    spec is refused the same way the object-embedding spellings are."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.zeros(2, dtype=[("a", "f8")])


def test_float_power_out_is_refused_under_real_weights():
    """Regression for an argument-order tie-break trap: _pointwise.py's
    _BINARY_FLOAT64_MIN_OPS branch folds the resolved dtype with a float64
    floor via heavier_billing_dtype(resolved, minimum), which -- before the
    operand-side backstop -- could silently drop an object out= whenever
    float64 outranked object's floor rate (masked in unit mode by the tie
    landing on the first-listed argument). The operand-side backstop in
    _counted_wrapper now catches the object out= array before
    _pointwise.py's fold logic ever runs, checked here under real weights
    since that is the mode the gap actually showed up in."""
    from flopscope._weights import load_weights

    load_weights()
    a = np.ones((2, 3), dtype=np.float32)
    out = np.empty((2, 3), dtype=object)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.float_power(a, a, out=out)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "call",
    [
        lambda: fnp.random.permutation(_obj_row()),
        lambda: fnp.random.choice(_obj_row()),
        lambda: fnp.random.default_rng(0).permutation(_obj_row()),
        lambda: fnp.random.default_rng(0).choice(_obj_row()),
        lambda: fnp.random.default_rng(0).permuted(_obj_row()),
        lambda: fnp.random.RandomState(0).permutation(_obj_row()),
    ],
)
def test_random_movement_family_refuses_object_pools(call):
    """random.permutation/choice/shuffle (module-level) and the identical
    Generator/RandomState class-method family (_make_counted_method's
    ``_movement_methods`` set in _counted_classes.py) relocate caller-
    supplied values without touching them, so they bill dtype-neutral by
    design -- and unconditionally set billing_dtypes=() regardless of the
    pool's actual dtype, so an object-dtype pool never reached a dtype
    check at all. This also supersedes the old permit-and-preserve-identity
    behavior pinned by test_api_branch_paths_cov.py (see
    test_choice_object_dtype_pool_is_refused there) -- the ban is
    unconditional, with no carve-out for movement ops."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call()


def test_random_shuffle_family_refuses_object_pools():
    """shuffle mutates in place and returns None, so it needs its own test
    (the other movement ops above return the object array directly)."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.shuffle(_obj_row())
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.default_rng(0).shuffle(_obj_row())
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.RandomState(0).shuffle(_obj_row())


def test_random_movement_family_refuses_non_object_non_numeric_pools():
    """The same ``_movement_methods`` gap above, for a non-numeric pool that
    is not object -- a string array. It relocates values without touching
    them the same way an object pool does, so the fix must not be scoped to
    `hasobject`: the operand-side backstop's dtype check is a numeric
    allowlist, and the pool's array argument reaches it either way."""
    pool = np.array(["a", "b", "c", "d"], dtype="U8")
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.permutation(pool)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.default_rng(0).choice(pool)


def test_bartlett_refuses_an_object_window_length():
    """bartlett's own dtypes= declaration is hardcoded to float64 ("numpy
    window functions always return float64"), which is false the moment
    the window length M itself is an object-dtype 0-d array: np.bartlett
    computes via compare/divide/add/select (no transcendental), all of
    which object-dtype Python floats support, so it silently returned an
    object-dtype window before this fix instead of the float64 the
    wrapper's own dtypes= tuple claimed."""
    m = np.array(4.0, dtype=object)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.bartlett(m)  # type: ignore[arg-type]


def test_stats_distribution_functions_refuse_non_numeric_operands():
    """flopscope.stats's pdf/cdf/ppf callables are outside the flopscope.numpy
    namespace but go through the same @_counted_wrapper machinery."""
    import flopscope.stats as fst

    x = _obj_row()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fst.norm.pdf(x)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fst.norm.ppf(x)


def test_dtype_neutral_ops_also_refuse_non_object_non_numeric_dtypes():
    """The operand-side backstop fires for U/S/M/m and an object-free
    structured (V) dtype too, not just `hasobject` -- the ban is a numeric
    allowlist, and a dtype-neutral movement op offers no exemption from it."""
    u = np.array(["a", "bb", "ccc"], dtype="U8")
    s = np.array([b"a", b"bb", b"ccc"], dtype="S8")
    m8 = np.array(["2020-01-01", "2020-01-02"], dtype="M8[D]")
    delta = np.array([1, 2, 3], dtype="m8[D]")
    v = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=[("a", "f8"), ("b", "f8")])
    with BudgetContext(BUDGET, quiet=True):
        for arr in (u, s, m8, delta, v):
            with pytest.raises(UnsupportedDtypeError):
                fnp.reshape(arr, arr.shape)
            with pytest.raises(UnsupportedDtypeError):
                fnp.copy(arr)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.zeros((3, 3), dtype="U8")
        with pytest.raises(UnsupportedDtypeError):
            fnp.zeros((3, 3), dtype="M8[D]")


# --- the operand scan must never execute participant code ---
#
# A naive `getattr(value, "dtype", None)` on every positional arg and every
# non-"dtype" kwarg resolves through ordinary attribute lookup, which runs
# whatever code answers a `.dtype` PROPERTY -- not just a genuine array's
# C-level attribute. Any op accepting such an object (as a plain argument,
# as `out=`, or any other kwarg -- e.g. `linalg.norm`'s `ord=`, which
# `norm_cost` never even looks at) would run that property's body as a side
# effect of the ban's own refusal check, whether or not the check
# ultimately raises -- exactly the kind of hidden, unbilled work the ban
# exists to close, relocated into the guard itself. The scan is restricted
# to genuine `np.ndarray` instances, read through the base class's own
# `dtype` descriptor, so a subclass that shadows `.dtype` with a Python
# property of its own cannot run its body via this check.


class _SpyDtype:
    """A non-array object whose `.dtype` property records how many times it
    was actually read, standing in for arbitrary hidden participant work."""

    def __init__(self):
        self.calls = 0

    @property
    def dtype(self):
        self.calls += 1
        return np.dtype("float64")


class _SpyNdarraySubclass(np.ndarray):
    """An np.ndarray SUBCLASS that shadows the base `.dtype` descriptor with
    its own property -- confirms a bare `isinstance(value, np.ndarray)`
    gate is not sufficient on its own; the base descriptor must be read
    directly."""

    calls = 0

    @property
    def dtype(self):
        type(self).calls += 1
        return np.dtype("float64")


def test_operand_scan_never_executes_a_dtype_property():
    """A `.dtype` property on `out=` and on an unrelated kwarg (`ord=`,
    which `norm_cost` doesn't even consult) must never run. Neither `out=`
    nor `ord=` accepts a bare non-array/non-ord object, so both calls raise
    downstream in the real op's own validation (`sum` raises "out= must be
    an array", `norm` raises on an invalid `ord`) -- the point under test
    is only that our check never touches `.dtype` on the way there,
    independent of what the real op does with it afterward."""
    spy_out = _SpyDtype()
    with BudgetContext(BUDGET, quiet=True):
        try:
            fnp.sum(np.ones(4), out=spy_out)
        except Exception:
            pass
    assert spy_out.calls == 0

    spy_ord = _SpyDtype()
    with BudgetContext(BUDGET, quiet=True):
        try:
            fnp.linalg.norm(np.ones(8), ord=spy_ord)
        except Exception:
            pass
    assert spy_ord.calls == 0


def test_operand_scan_never_executes_a_dtype_property_across_a_spread_of_ops():
    """Same invariant, swept across a representative spread of ops:
    dtype-neutral movement (reshape), array-creation dtype= (zeros -- passed
    a spy as a plain positional arg, which it will reject downstream, but
    must not read .dtype on the way), a list-of-arrays op (concatenate),
    and a keyword slot that isn't `dtype`/`out` at all (axis=)."""
    spies = [_SpyDtype() for _ in range(4)]
    with BudgetContext(BUDGET, quiet=True):
        for spy in spies:
            try:
                fnp.reshape(spy, (1,))  # type: ignore[arg-type]
            except Exception:
                pass
        try:
            fnp.concatenate([spies[0], spies[1]])  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            fnp.flip(spies[2], axis=spies[3])  # type: ignore[arg-type]
        except Exception:
            pass
    assert all(spy.calls == 0 for spy in spies)


def test_plain_dtype_like_resolves_inert_structured_specs_only():
    """Direct unit coverage for `_plain_dtype_like`'s list/tuple branch: an
    inert structured spec -- including one nested a level deep -- resolves
    to the same np.dtype as its np.dtype(...) spelling, while a spec
    containing a non-inert leaf (a duck-typed `.dtype` proxy) is left
    unresolved: this backstop must never read that property speculatively,
    so the call count must stay 0."""
    from flopscope._budget import _plain_dtype_like

    assert _plain_dtype_like([("a", object)]) == np.dtype([("a", object)])
    assert _plain_dtype_like([("a", [("b", object)])]) == np.dtype(
        [("a", [("b", object)])]
    )
    assert _plain_dtype_like([("a", "f8")]) == np.dtype([("a", "f8")])

    spy = _SpyDtype()
    assert _plain_dtype_like([("a", spy)]) is None  # type: ignore[list-item]
    assert spy.calls == 0


def test_operand_scan_bypasses_a_hostile_ndarray_subclass_dtype_override():
    """A bare `isinstance(value, np.ndarray)` gate is not enough on its own:
    a Python subclass of ndarray can shadow the base `.dtype` descriptor
    with its own property, and ordinary attribute access (even after an
    isinstance check) runs it. The fix reads through the base descriptor
    directly, so the override must never run, while the array's real
    (object) dtype must still be detected and refused."""
    _SpyNdarraySubclass.calls = 0
    o = np.empty(3, dtype=object)
    o[:] = [1.0, 2.0, 3.0]
    evil = o.view(_SpyNdarraySubclass)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.copy(evil)
    assert _SpyNdarraySubclass.calls == 0


def test_copyto_string_destination_is_refused():
    """copyto declared its destination as a plain operand, so a non-numeric
    destination used to drag result_type to rate 1.0 -- exactly half price.
    Every other out= path routes through store_billing_dtypes, and now that
    it does too, a str ('U') destination is refused outright rather than
    priced at either rate -- str is outside the numeric allowlist, the same
    as every other non-numeric out=."""
    src = np.ones(64)
    with BudgetContext(BUDGET, quiet=True) as bc:
        with pytest.raises(UnsupportedDtypeError):
            fnp.copyto(np.empty(64, dtype="<U32"), src)
        assert bc.flops_used == 0


def test_copyto_bytes_destination_is_refused():
    """Same fix, bytes kind ('S') rather than str ('U') -- both are
    non-numeric kinds that `result_type` resolves TO rather than raising
    a DTypePromotionError for, so both used to silently reach the neutral
    rate; both are now refused outright."""
    src = np.ones(64)
    with BudgetContext(BUDGET, quiet=True) as bc:
        with pytest.raises(UnsupportedDtypeError):
            fnp.copyto(np.empty(64, dtype="|S32"), src)
        assert bc.flops_used == 0


def test_copyto_numeric_destination_billing_is_unchanged():
    """The fix must not touch a call that was already correct: a narrower
    numeric destination (float32) still bills at the source's (float64)
    rate, exactly as before."""
    from flopscope._weights import load_weights

    load_weights()
    src = np.ones(64)
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.copyto(np.empty(64, dtype=np.float32), src)
        assert bc.flops_used == 128


def test_copyto_object_destination_still_refused():
    """The object ban is a separate mechanism from this fix and must keep
    refusing an object copyto destination outright, not merely bill it at
    the repaired rate."""
    src = np.ones(64)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.copyto(np.empty(64, dtype=object), src, casting="unsafe")


# ---------------------------------------------------------------------------
# the object-dtype scans must be depth-bounded AND cycle-safe
# ---------------------------------------------------------------------------
# _refuse_non_numeric_operands.check() (_budget.py) and _refuse_non_numeric_dtype_tree
# (_array_ops.py) both recurse into list/tuple operands looking for an
# object-dtype array. Unbounded, a self-referential container turns that
# search itself into a RecursionError -- a stack blowout the ban's own
# machinery introduces, not something the ORIGINAL numpy call would ever hit
# (numpy raises its own clean ValueError on a self-referential sequence).
#
# The depth bound alone stops a RecursionError but not a container that
# references itself more than once at the same level: that branches
# exponentially with depth, so both scans additionally track container ids
# currently on the recursion path and raise promptly on a repeat -- without
# that, control falls through to NumPy's own array construction, whose shape
# discovery has the identical exponential blind spot (bounded only by its own
# ~32-dimension limit, which is far from prompt).


def test_self_referential_list_raises_value_error_not_recursion_error():
    """Plain numpy raises a clean ``ValueError`` for
    ``a = []; a.append(a); fnp.array(a)``. The operand scan in
    ``_refuse_non_numeric_operands`` must not turn that into a
    ``RecursionError`` by recursing into the self-referential list."""
    a: list = []
    a.append(a)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(ValueError):
            fnp.array(a)


def test_refuse_non_numeric_dtype_tree_single_self_reference_raises_value_error():
    """Same invariant as the test above, pinned directly against
    ``_refuse_non_numeric_dtype_tree`` (the backstop ``permute_dims`` uses, since
    it has no ``@_counted_wrapper`` of its own)."""
    from flopscope._array_ops import _refuse_non_numeric_dtype_tree

    a: list = []
    a.append(a)
    with pytest.raises(ValueError):
        _refuse_non_numeric_dtype_tree("permute_dims", a)


# A container that references itself more than once at the same level (not
# just a single a.append(a)) is the case the depth bound alone does not
# cover: each self-reference doubles (or triples) the scan's branching
# factor at every level, so 64 levels of unchecked recursion is ~2**64 (or
# ~3**64) visits, not 64. These run each repro in a subprocess with a hard
# wall-clock timeout, so a regression that reintroduces the exponential scan
# -- or one that falls through into NumPy's own equally-exponential shape
# discovery -- fails the test suite promptly instead of hanging CI. A
# signal-based, in-process timeout is not good enough here: if the fallback
# path is NumPy's own (C-level) array construction, a pending signal is not
# delivered until that call returns control to Python, which is exactly the
# multi-minute case this guard needs to fail fast on.

_CYCLE_GUARD_TIMEOUT_S = 30.0


def _array_two_self_reference_cycle():
    import flopscope.numpy as fnp
    from flopscope import BudgetContext

    x: list = []
    x.extend([x, x])
    with BudgetContext(BUDGET, quiet=True):
        fnp.array(x)


def _array_three_self_reference_cycle():
    import flopscope.numpy as fnp
    from flopscope import BudgetContext

    x: list = []
    x.extend([x, x, x])
    with BudgetContext(BUDGET, quiet=True):
        fnp.array(x)


def _refuse_non_numeric_dtype_tree_two_self_reference_cycle():
    from flopscope._array_ops import _refuse_non_numeric_dtype_tree

    x: list = []
    x.extend([x, x])
    _refuse_non_numeric_dtype_tree("permute_dims", x)


def _refuse_non_numeric_dtype_tree_three_self_reference_cycle():
    from flopscope._array_ops import _refuse_non_numeric_dtype_tree

    x: list = []
    x.extend([x, x, x])
    _refuse_non_numeric_dtype_tree("permute_dims", x)


def _cycle_guard_runner(target, q):
    """Subprocess entry point: run *target* and report the outcome back
    over *q*, rather than letting an exception simply end the process --
    the parent needs the actual exception object to assert its type."""
    try:
        target()
    except BaseException as exc:  # noqa: BLE001 -- reported to the parent, not swallowed
        q.put(("exception", exc))
    else:
        q.put(("ok", None))


def _returns_or_raises_within(target, timeout=_CYCLE_GUARD_TIMEOUT_S):
    """Run *target* (a no-arg, picklable module-level callable) to
    completion in a subprocess, re-raising whatever it raised in this
    process. Fails the test immediately, rather than hanging, if *target*
    has not finished within *timeout* seconds -- ``proc.kill()`` reaps a
    still-running child directly instead of waiting it out, so a regression
    that reintroduces the hang cannot block the rest of the suite either."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_cycle_guard_runner, args=(target, q))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(
            f"{target.__name__} did not return within {timeout}s -- "
            "the cyclic-container guard looks like it regressed"
        )
    kind, payload = q.get()
    if kind == "exception":
        raise payload


@pytest.mark.parametrize(
    "target",
    [
        _array_two_self_reference_cycle,
        _array_three_self_reference_cycle,
        _refuse_non_numeric_dtype_tree_two_self_reference_cycle,
        _refuse_non_numeric_dtype_tree_three_self_reference_cycle,
    ],
)
def test_multi_self_reference_cycle_raises_value_error_promptly(target):
    with pytest.raises(ValueError):
        _returns_or_raises_within(target)


# --- a shared-but-acyclic sublist ("diamond") must still be scanned -------
#
# The fix above skips a container once its id has already been (or is
# currently being) walked. That must not cost correctness for the ordinary,
# non-cyclic case of the same sublist being referenced from two places: it
# is the same object with the same contents each time, so the walk that runs
# the first time it is reached already finds anything inside it -- but a
# skip keyed on the wrong condition (e.g. treating "seen before" as "cannot
# ever be visited again") could equally well drop it as a false cycle.


def test_diamond_shared_sublist_object_dtype_is_still_refused():
    """A non-cyclic sublist reachable twice, carrying an object-dtype array,
    must still be refused -- id-based deduplication must not read as "skip
    this content" when it is really "already fully scanned"."""
    shared = [_obj_cells(n=1, size=4)]
    x = [shared, shared]
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.array(x)


def test_diamond_shared_sublist_is_not_mistaken_for_a_cycle():
    """The mirror case: a purely numeric sublist referenced twice is
    shared but not self-referential, and must construct normally -- neither
    refused as object dtype nor raised as a self-referential sequence."""
    shared = [1.0, 2.0, 3.0]
    x = [shared, shared]
    with BudgetContext(BUDGET, quiet=True):
        fnp.array(x)  # must not raise


# ---------------------------------------------------------------------------
# the operand scan must judge a raw Python sequence by its LEAVES
# ---------------------------------------------------------------------------
#
# `_refuse_non_numeric_operands.check()` walked a list/tuple looking only for
# an `np.ndarray` to read a dtype off. A raw Python sequence of str/bytes/
# datetime holds no array anywhere, so the walk found nothing and the op ran
# -- while the ndarray spelling of the very same call was refused. The ops
# that leaked are exactly the ones with no source check of their own: the
# dtype-neutral movement family (`transpose`, `flip`, `permute_dims`) and the
# random pool ops (`choice`, `permutation`, `shuffle`), all of which declare
# `dtypes=()` and so never reach `deduct`'s refusal either.
#
# The rule refuses the leaf types that are unambiguously DATA and
# unambiguously non-numeric: str, bytes/bytearray, a NumPy scalar whose own
# dtype is not numeric, and a stdlib date/time/timedelta. It stops there --
# `None`, a callable, a set and a range all realise as object dtype, but they
# are also how numpy spells `fftn`'s per-axis sentinel, `piecewise`'s
# funclist, and `tensordot`'s `axes` pair, and this scan sees every argument
# of every counted op with no way to tell those apart from a payload. Two
# exemptions follow from the same reasoning: the rule applies only INSIDE a
# list/tuple (a bare `str` argument is an option string -- `ord='fro'`,
# einsum subscripts, `casting=`), and never to the string-specifier
# parameters (`requirements=["C"]`, `order=["x"]`, a ufunc `signature`).


def _str_pool_list():
    return ["a", "b", "c", "d"]


SEQUENCE_REPROS = [
    ("choice", lambda pool: fnp.random.choice(pool, 2)),
    ("permutation", lambda pool: fnp.random.permutation(pool)),
    ("shuffle", lambda pool: fnp.random.shuffle(pool)),
    ("transpose", lambda pool: fnp.transpose([pool, pool])),
    ("flip", lambda pool: fnp.flip(pool)),
    ("permute_dims", lambda pool: fnp.permute_dims(pool, (0,))),
    ("reshape", lambda pool: fnp.reshape(pool, (4,))),
    ("tuple spelling", lambda pool: fnp.flip(tuple(pool))),
    ("nested", lambda pool: fnp.transpose([[pool[0], pool[1]], [pool[2], pool[3]]])),
]


@pytest.mark.parametrize(
    "label,call", SEQUENCE_REPROS, ids=[label for label, _ in SEQUENCE_REPROS]
)
def test_python_sequence_of_strings_is_refused_and_bills_nothing(label, call):
    """The reported repros. `fnp.random.choice(['a','b','c','d'], 2)` billed
    8 FLOPs and handed back a real `<U1` array; the ndarray spelling of the
    same call already raised."""
    with BudgetContext(BUDGET, quiet=True) as bc:
        with pytest.raises(UnsupportedDtypeError):
            call(_str_pool_list())
        assert bc.flops_used == 0


@pytest.mark.parametrize(
    "leaf",
    [
        "a",
        b"a",
        np.str_("a"),
        np.bytes_(b"a"),
        np.datetime64("2020-01-01"),
        np.timedelta64(1, "D"),
        np.void(b"ab"),
        datetime.date(2020, 1, 1),
        datetime.datetime(2020, 1, 1),
        datetime.time(1, 2),
        datetime.timedelta(days=1),
    ],
    ids=[
        "str",
        "bytes",
        "np.str_",
        "np.bytes_",
        "datetime64",
        "timedelta64",
        "np.void",
        "date",
        "datetime",
        "time",
        "timedelta",
    ],
)
def test_every_non_numeric_leaf_category_is_refused(leaf):
    """One entry per branch of the leaf classifier: str, bytes, a NumPy
    scalar carrying its own non-numeric dtype, and a stdlib temporal value
    (`datetime.datetime` subclasses `datetime.date`)."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.flip([leaf, leaf])


@pytest.mark.parametrize(
    "call",
    [
        lambda: fnp.piecewise(np.linspace(-2, 2, 8), [np.zeros(8, bool)], [0.0, 1.0]),
        lambda: fnp.fft.fftn(np.ones((4, 8)), s=(None, 8)),
        lambda: fnp.tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=({1}, {0})),
        lambda: fnp.tensordot(
            np.ones((2, 3)), np.ones((3, 4)), axes=(range(1, 2), range(0, 1))
        ),
        lambda: fnp.require(np.ones(4), requirements=["C"]),
        lambda: fnp.sort(np.ones(4), order=None),
        lambda: fnp.multiply(np.ones(4), np.ones(4), signature=("d", "d", "d")),
    ],
    ids=["piecewise", "fftn-s", "axes-set", "axes-range", "require", "order", "sig"],
)
def test_specifier_slots_holding_non_array_leaves_still_run(call):
    """The over-rejection guard for the rule's deliberate stopping point.

    Every one of these is a numpy spelling whose SPECIFIER holds leaves that
    realise as object or string dtype: a funclist of callables, `None` as a
    per-axis sentinel, a pair of sets/ranges, a list of flag names, a ufunc
    signature. A leaf rule that fails closed on the whole non-numeric family
    refuses all of them -- calls numpy runs."""
    with BudgetContext(BUDGET, quiet=True):
        call()  # must not raise


def test_einsum_explicit_path_spec_is_not_refused_by_the_ban():
    """`optimize=['einsum_path', (0, 1)]` is a list whose first leaf is a
    STRING -- the one string-specifier slot the leaf rule would otherwise
    catch in a positional-looking operand list.

    flopscope's `einsum` has a separate, pre-existing gap on that spelling
    (opt_einsum rejects the path list with a `TypeError` before numpy sees
    it, on `origin/main` too), so this pins the ban specifically: whatever
    einsum does with the path, it must not be `UnsupportedDtypeError`."""
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(Exception) as exc:  # noqa: B017 - type is not the point
            fnp.einsum(
                "ij,jk->ik",
                np.ones((2, 3)),
                np.ones((3, 4)),
                optimize=["einsum_path", (0, 1)],
            )
    assert not isinstance(exc.value, UnsupportedDtypeError)


@pytest.mark.parametrize(
    "seq",
    [
        [1.0, 2.0],
        [1, 2],
        [True, False],
        [1 + 2j, 3 + 4j],
        [np.float32(1.0), np.float32(2.0)],
        [np.int8(1), np.uint64(2)],
        [np.bool_(True)],
        [[1.0, 2.0], [3.0, 4.0]],
        ([1.0, 2.0], (3.0, 4.0)),
        [],
        [np.ones(2), np.ones(2)],
    ],
    ids=[
        "float",
        "int",
        "bool",
        "complex",
        "np.float32",
        "np.int8/np.uint64",
        "np.bool_",
        "nested",
        "mixed-containers",
        "empty",
        "arrays",
    ],
)
def test_numeric_python_sequences_are_untouched(seq):
    """The over-rejection guard. Every numeric leaf category numpy realises
    into a numeric dtype must still construct, at every nesting depth."""
    with BudgetContext(BUDGET, quiet=True):
        fnp.array(seq)  # must not raise


def test_numeric_sequence_billing_is_unchanged():
    """A leaf check on the operand scan must not move a single billed FLOP
    for a call that was already valid."""
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.multiply([1.0, 2.0], [3.0, 4.0])
        assert bc.flops_used == 2
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.array([1.0, 2.0])
        assert bc.flops_used == 2


@pytest.mark.parametrize(
    "leaf",
    [bytearray(b"abcd"), memoryview(b"abcd")],
    ids=["bytearray", "memoryview"],
)
def test_buffer_leaves_are_numeric_and_still_bill(leaf):
    """A buffer leaf realizes as uint8, so it is INSIDE the allowlist.

    ``bytearray`` looks like ``bytes`` and is not: NumPy reads it through the
    buffer protocol, so ``np.array([bytearray(b"ab")]).dtype`` is uint8 where
    ``np.array([b"ab"]).dtype`` is "S". Refusing it would reject a numeric
    payload, and unlike the ``str`` case the ndarray spelling never refused it
    either -- so refusing the sequence spelling would MANUFACTURE an asymmetry
    rather than close one. Pinned against numpy's own realization so this
    tracks numpy rather than a remembered list.
    """
    assert np.array([leaf]).dtype == np.uint8

    with BudgetContext(BUDGET, quiet=True) as bc:
        result = fnp.sum([leaf])
    assert bc.flops_used > 0, f"a {type(leaf).__name__} leaf must still bill"
    assert np.asarray(result).dtype.kind in "biufc"

    # The mixed spelling is unambiguously a numeric call.
    with BudgetContext(BUDGET, quiet=True) as bc:
        fnp.concatenate([np.ones(4, np.uint8), leaf])
    assert bc.flops_used > 0


def test_top_level_string_arguments_are_not_treated_as_operands():
    """A bare `str`/`bytes` argument is an option, not an array payload.

    The leaf rule fires only inside a list/tuple; refusing a top-level one
    would break every op numpy spells with a string mode -- `ord=`, einsum
    subscripts, `casting=`, `kind=`, an `astype` dtype code."""
    a = np.ones((3, 3))
    with BudgetContext(BUDGET, quiet=True):
        fnp.linalg.norm(a, ord="fro")
        fnp.einsum("ij,jk->ik", a, a)
        fnp.astype(a, "float64")
        fnp.sort(a, kind="stable")
        fnp.sum(a, dtype="float64")
        fnp.pad(a, 1, mode="constant")


def test_leaf_check_never_executes_participant_code():
    """The scan's standing invariant, extended to the leaf branch: a value
    reached inside a list is classified from its TYPE alone, so no attribute
    of it is read on the way past -- refused or not.

    A `str` SUBCLASS is the case that makes this non-trivial: it is refused,
    and the classifier must reach that conclusion without asking the instance
    anything (no `len()`, no `.dtype`, no `__array__`)."""

    class _HostileStr(str):
        def __getattr__(self, name):  # pragma: no cover - must never run
            raise AssertionError(f"operand scan touched .{name}")

        def __len__(self):  # pragma: no cover - must never run
            raise AssertionError("operand scan measured the leaf")

    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.flip([_HostileStr("a"), _HostileStr("b")])


def test_string_spec_positional_slots_cover_the_positional_spelling():
    """`requirements` is exempt however it is passed. The keyword spelling is
    a name lookup; the positional one needs the wrapper's signature, resolved
    once per wrapper rather than once per call."""
    from flopscope._budget import _string_spec_positional_slots

    with BudgetContext(BUDGET, quiet=True):
        fnp.require(np.ones(4), None, ["C"])  # must not raise
    # `require`'s wrapper body is `(*args, **kwargs)`; the numpy signature
    # that maps slot 2 to `requirements` lives on the DECORATED object, which
    # is why the scan resolves slots against that rather than the closure.
    assert _string_spec_positional_slots(fnp.require) == frozenset({2})


def test_refuse_non_numeric_sequence_leaf_unit():
    """Direct unit coverage for the classifier, independent of any op."""
    from flopscope._budget import refuse_non_numeric_sequence_leaf

    passes = (True, 1, 1.5, 1 + 2j, np.float16(1.0), np.int32(1), None, object(), {1})
    for value in passes:
        refuse_non_numeric_sequence_leaf("multiply", value)  # must not raise
    for non_numeric in ("a", b"a", np.datetime64("2020-01-01"), datetime.date.today()):
        with pytest.raises(UnsupportedDtypeError):
            refuse_non_numeric_sequence_leaf("multiply", non_numeric)


# --- sibling sequence-coercion routes: fromiter/require/full/full_like,
# every random sampler, and the stats distribution surface all cast a
# caller-supplied source (or a distribution parameter) to a numeric dtype
# the same way array()/asarray()/astype() do above -- the same no-dtype
# probe closes the same gap on each of them.


def test_fromiter_refuses_object_source():
    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.fromiter(payloads, dtype=np.float64)
    assert sum(p.float_calls for p in payloads) == 0


def test_require_refuses_object_source():
    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.require(payloads, dtype=np.float64)
    assert sum(p.float_calls for p in payloads) == 0


def test_full_refuses_object_fill_value():
    p = _CountingPayload()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.full((4,), p, dtype=np.float64)  # type: ignore[arg-type]
    assert p.float_calls == 0


def test_full_like_refuses_object_fill_value():
    p = _CountingPayload()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.full_like(fnp.ones(4), p)  # type: ignore[arg-type]
    assert p.float_calls == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda payloads: fnp.random.uniform(payloads, 1.0),
        lambda payloads: fnp.random.normal(loc=payloads, scale=1.0),
        lambda payloads: fnp.random.exponential(scale=payloads),
        lambda payloads: fnp.random.default_rng(0).normal(loc=payloads, scale=1.0),
        lambda payloads: fnp.random.default_rng(0).uniform(low=payloads, high=1.0),
        lambda payloads: fnp.random.RandomState(0).normal(loc=payloads, scale=1.0),
    ],
    ids=[
        "module.uniform",
        "module.normal",
        "module.exponential",
        "Generator.normal",
        "Generator.uniform",
        "RandomState.normal",
    ],
)
def test_random_sampler_family_refuses_object_distribution_parameters(call):
    """Every counted sampler -- module-level, Generator, RandomState -- casts
    its distribution parameters to float64 itself; a bare payload sequence
    reaching one of them must be refused before that cast runs."""
    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            call(payloads)
    assert sum(p.float_calls for p in payloads) == 0


def test_random_multivariate_normal_refuses_object_mean():
    payloads = _object_list(3)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.multivariate_normal(payloads, np.eye(3))
    assert sum(p.float_calls for p in payloads) == 0


def test_stats_distribution_refuses_object_x():
    import flopscope.stats as fst

    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fst.norm.pdf(payloads)
    assert sum(p.float_calls for p in payloads) == 0


def test_stats_distribution_refuses_object_distribution_parameter():
    """x itself is a plain numeric array here -- only loc (broadcast against
    it inside the pure-numpy kernel) carries the object payload."""
    import flopscope.stats as fst

    payloads = _object_list(2)
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fst.norm.pdf(np.array([0.0, 1.0]), loc=payloads)  # type: ignore[arg-type]
    assert sum(p.float_calls for p in payloads) == 0


# --- the refusal message must name the real op, not the generic factory
# closure that some of these routes are built from internally ---


@pytest.mark.parametrize(
    "op_name, call",
    [
        ("multiply", lambda o: fnp.multiply(o, o)),
        ("mean", lambda o: fnp.mean(o)),
        ("sum", lambda o: fnp.sum(o)),
        ("std", lambda o: fnp.std(o)),
        ("fromiter", lambda o: fnp.fromiter(o, dtype=np.float64)),
    ],
    ids=[
        "pointwise-ufunc",
        "reduction-factory-a",
        "reduction-factory-b",
        "reduction-factory-c",
        "creation-coercion",
    ],
)
def test_refusal_message_names_the_real_op(op_name, call):
    """A wrapper built by a factory (mean/std/... via an internal closure
    literally named ``wrapper``) must still report ITS OWN name, not the
    closure's definition-time name."""
    o = _obj_row()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError, match=rf"^{op_name}:"):
            call(o)


def test_sibling_routes_also_refuse_under_real_weights():
    """The sibling gaps above are structural (a missing probe, an unchecked
    argument), not rate-dependent -- checked here under real production
    weights since a prior gap in this same area was masked in unit mode by
    an argument-order tie-break that a table of all-1.0 rates cannot expose."""
    from flopscope._weights import load_weights

    load_weights()
    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.fromiter(payloads, dtype=np.float64)
    assert sum(p.float_calls for p in payloads) == 0

    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.require(payloads, dtype=np.float64)
    assert sum(p.float_calls for p in payloads) == 0

    p = _CountingPayload()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.full_like(fnp.ones(4), p)  # type: ignore[arg-type]
    assert p.float_calls == 0

    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.uniform(payloads, 1.0)
    assert sum(p.float_calls for p in payloads) == 0

    payloads = _object_list()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.random.default_rng(0).normal(loc=payloads, scale=1.0)
    assert sum(p.float_calls for p in payloads) == 0

    o = _obj_row()
    with BudgetContext(BUDGET, quiet=True):
        with pytest.raises(UnsupportedDtypeError, match=r"^mean:"):
            fnp.mean(o)
