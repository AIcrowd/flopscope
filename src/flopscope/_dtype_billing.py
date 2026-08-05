"""Dtype-aware billing helpers: resolved calculation dtype and complex factors.

The billed dtype for a call is ``np.result_type`` over the declared operands
(input dtypes, python scalars for NEP 50 weak promotion, plus any explicit
``dtype=``/``out=`` dtype). ``dtypes=()`` declares a dtype-neutral op
(rate 1.0, factor 1.0).

Folding ``out=`` into that resolution prices the widest participating
buffer -- ``max(compute width, store width)`` -- not just the compute width.
A wider ``out=`` forces a real materialization: writing the result into a
wider buffer is itself chargeable, the same as any other copy into a wider
dtype. Billing it at the wider rate holds ``out=``-casting at exact
``astype`` parity in both directions -- a wider ``out=`` costs what the
equivalent ``astype`` costs, and a narrower ``out=`` never discounts the
loop that actually runs. Reductions separately fold ``out=`` into
``reduction_billing_dtype`` for a different, genuine reason: there, a wider
``out=`` changes the accumulator numpy actually runs, not merely the final
store, so that path is unaffected by this note.

Two destinations are outside the doctrine because they are not stores of the
computed value at all, and both are handled by their own helper rather than
by ``store_billing_dtypes``. A multi-output op can have an output whose dtype
is part of the op SIGNATURE (``frexp``'s int32 exponent) --
``multi_store_billing_dtypes``. An index reduction's destination holds
positions rather than values, so its width says nothing about the arithmetic
-- ``_pointwise._INDEX_RETURNING_REDUCTIONS``. In both cases supplying the
buffer numpy would have allocated anyway must be price-neutral against the
bare call; widening past it still widens the rate.
"""

from __future__ import annotations

import numpy as _np

from flopscope._weights import _UFUNC_METHOD_SUFFIXES, get_dtype_rate
from flopscope.errors import UnsupportedDtypeError


def _weak_numeric_subclass_types() -> frozenset[type]:
    """Built-in scalar families whose subclasses NumPy still promotes weakly.

    NumPy 2.0 treats subclasses of Python's numeric scalar types as weak,
    while NumPy 2.1 and later treat them as concrete scalars. Probe the public
    ``result_type`` behavior so resolver operands track the installed NumPy
    instead of encoding that version boundary here.
    """

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class ComplexSubclass(complex):
        pass

    cases = (
        (int, IntSubclass(1), _np.dtype(_np.int8)),
        (float, FloatSubclass(1.0), _np.dtype(_np.float32)),
        (complex, ComplexSubclass(1.0j), _np.dtype(_np.complex64)),
    )
    return frozenset(
        scalar_type
        for scalar_type, probe, narrow_dtype in cases
        if _np.result_type(narrow_dtype, probe) == narrow_dtype
    )


_WEAK_NUMERIC_SUBCLASS_TYPES = _weak_numeric_subclass_types()


def billing_operand(orig, coerced):
    """Result-type operand for billing.

    Python scalars are passed through so numpy's weak-promotion rules apply
    (``f32_array * 2.0`` bills float32); everything else contributes the
    coerced array's dtype.
    """
    if isinstance(orig, (bool, int, float, complex)) and not isinstance(
        orig, _np.generic
    ):
        return orig
    return coerced.dtype


def ufunc_resolver_operand(orig, coerced) -> _np.dtype | type:
    """Operand form accepted by ``ufunc.resolve_dtypes``.

    Exact built-in int, float, and complex scalars contribute their bare type
    so NumPy applies NEP 50 weak promotion. Numeric subclasses follow the
    installed NumPy's promotion behavior. Bool, NumPy scalars, and arrays
    contribute their concrete dtype.
    """
    if type(orig) in (int, float, complex):
        return type(orig)
    if type(orig) is not bool and not isinstance(orig, _np.generic):
        for scalar_type in _WEAK_NUMERIC_SUBCLASS_TYPES:
            if isinstance(orig, scalar_type):
                return scalar_type
    return coerced.dtype


def store_billing_dtypes(out) -> tuple:
    """What an ``out=`` buffer contributes to the billing resolution.

    Its dtype, unconditionally, whenever ``out`` is an ``ndarray``. Every
    non-numeric ``out=`` (object, string, bytes, structured/void, datetime64,
    timedelta64) is refused outright rather than priced -- see
    ``refuse_non_numeric_dtype`` -- and that refusal is keyed on ``dtypes``
    tuples built from this function's return value, at every
    ``deduct``/``deduct_after`` call site that folds this in. Dropping a
    non-numeric dtype here instead of returning it would hide the
    destination from that refusal, the same gap a discount would open.

    A *numeric* ``out=`` is priced at its own rate rather than folded away:
    billing it at the operand rate is deliberate and matches how ``out=`` is
    already treated elsewhere -- see the widest-participating-buffer note at
    the top of this module.
    """
    if not isinstance(out, _np.ndarray):
        return ()
    return (out.dtype,)


def natural_output_dtypes(np_func, resolved_input: _np.dtype | None) -> tuple | None:
    """Dtypes numpy allocates for each output slot when ``out=`` is omitted.

    Read out of the ufunc's own loop table via ``resolve_dtypes``, so it
    tracks whatever numpy in the support matrix is installed rather than a
    hand-maintained mapping that can drift.

    ``resolved_input`` is the ALREADY-PROMOTED operand dtype -- the same
    ``result_type`` the rest of the billing runs on, weak Python scalars
    folded in -- and it is fed to every input slot. Promoting first is what
    keeps a NEP 50 weak scalar from inventing a natural output wider than the
    call really has: ``divmod(f32_array, 2.0)`` computes float32, so its
    natural destinations are float32, and a caller-supplied float64 one is a
    genuine widening that must still reach the rate. Handing the raw operand
    dtypes to ``resolve_dtypes`` (which knows nothing of weak scalars) would
    have reported float64 as natural and handed that widening back for free.

    Returns ``None`` when the loop cannot be resolved at all -- a non-ufunc,
    an operand kind with no loop, a promotion numpy refuses. Callers then
    fall back to folding every destination into the rate unconditionally,
    which over-bills rather than under-bills.
    """
    resolve = getattr(np_func, "resolve_dtypes", None)
    nin = getattr(np_func, "nin", None)
    nout = getattr(np_func, "nout", None)
    if resolve is None or not nin or not nout or resolved_input is None:
        return None
    try:
        signature = resolve((resolved_input,) * nin + (None,) * nout)
    except (TypeError, ValueError, _np.exceptions.DTypePromotionError):
        return None
    if signature is None or len(signature) != nin + nout:
        return None
    return tuple(signature[nin:])


def multi_store_billing_dtypes(out, natural: tuple | None) -> tuple:
    """What the ``out=`` slots of a multi-output op contribute to the rate.

    Same widest-participating-buffer doctrine as :func:`store_billing_dtypes`,
    with the one correction a multi-output signature forces: a slot
    contributes only what it adds ON TOP of the buffer numpy would have
    allocated for that slot anyway.

    A single-output op does not need the distinction, because its natural
    destination is the compute dtype already in the resolution -- folding it
    in again is a no-op. A multi-output op can have an output whose dtype is
    part of the op SIGNATURE rather than of its arithmetic, and there folding
    unconditionally prices a buffer the caller merely supplied instead of
    letting numpy allocate it. ``frexp`` is the case in point: its second
    output is always ``int32``, whatever the mantissa's precision, so
    ``np.result_type(float32, float32, int32)`` promoted to float64 and
    ``frexp(a32, out=(m32, e32))`` billed twice what ``frexp(a32)`` billed --
    the caller paying the float64 rate for arithmetic numpy runs at float32,
    purely for naming destinations numpy would have created itself. Supplying
    the natural destination must be price-neutral.

    Widening is untouched in either axis. A slot pricier than its natural
    counterpart still joins the resolution, so a float64 mantissa on a
    float32 ``frexp`` still bills at the float64 rate, and so does an int64
    exponent buffer where numpy would have made an int32 one. The kind axis
    is guarded separately: a complex destination over a real natural output
    joins even at an equal rate, since complex64 and float32 both rate 1.0
    and the op's complex factor rides on the KIND, not the rate -- the same
    tie that produced a 6x swing in ``prod`` and is pinned in
    :func:`reduction_billing_dtype`.

    ``natural=None`` means the loop could not be resolved; every destination
    then folds in as before, which over-bills rather than under-bills.
    """
    if out is None:
        return ()
    contributed: tuple = ()
    for index, slot_out in enumerate(out):
        slot = store_billing_dtypes(slot_out)
        if not slot:
            continue
        if natural is not None and index < len(natural):
            # EXACT match, not "no pricier by rate". The thing being skipped
            # here is a `result_type` JOIN, and rate ordering is a different
            # relation from promotion: result_type(int32, float32) is float64
            # even though int32 and float32 rate the same. A rate test
            # therefore drops every equal-rate CROSS-KIND destination from the
            # resolution -- measured as a fresh 2x discount on divmod, where a
            # float32 buffer receiving an int32 quotient is a genuine store of
            # the computed value and must be priced. Only the destination
            # numpy would have allocated itself is price-neutral, and that is
            # an identity, so test it as one.
            if slot[0] == _np.dtype(natural[index]):
                continue
        contributed += slot
    return contributed


def resolve_billing_dtype(dtypes: tuple) -> _np.dtype | None:
    """Resolved calculation dtype, or None for a declared dtype-neutral call.

    Some numpy loops accept operand mixes that have NO common promoted dtype
    (logical ufuncs accept anything; ``timedelta64 / float64`` divides without
    promoting), so ``result_type`` can refuse a combination the op itself
    supports. In that case bill conservatively at the heaviest individual
    operand's rate instead of failing a call plain numpy would allow.
    """
    if not dtypes:
        return None
    try:
        return _np.result_type(*dtypes)
    except _np.exceptions.DTypePromotionError:
        resolved = [_np.result_type(d) for d in dtypes]  # each alone resolves
        return max(resolved, key=rate_for)


# Numeric dtype kinds flopscope bills: bool (b), signed int (i), unsigned int
# (u), float (f), complex (c). This is an ALLOWLIST, not a denylist of the
# non-numeric kinds (object 'O', str 'U', bytes 'S', void/structured 'V',
# datetime64 'M', timedelta64 'm') -- a future numpy dtype kind (e.g. the
# variable-width string dtype's 'T') is refused by default instead of being
# silently admitted the way a denylist would admit it.
#
# Every kind outside this allowlist is refused outright by
# ``refuse_non_numeric_dtype`` rather than priced, for two distinct reasons:
# an object cell is a PyObject* whose arithmetic dispatches into arbitrary
# Python of unbounded cost, which no finite rate expresses; the other kinds
# (U/S/V/M/m) are bounded but their real per-element cost is not the fixed
# 32-bit-class unit a flat rate would have to assume -- a wide string or
# structured record does more work per element than a narrow one, and
# datetime64/timedelta64 are integers underneath, at whatever the platform's
# integer rate is, not the flat rate a dtype-blind price would charge them.
# Both reasons converge on the same fix: refuse rather than mis-price.
#
# ``rate_for`` and ``get_dtype_rate`` only ever see a dtype that already
# cleared this allowlist -- every ``dtypes=`` tuple passed to
# ``deduct``/``deduct_after`` is checked by ``refuse_non_numeric_dtype``
# first, and numeric-to-numeric promotion (``resolve_billing_dtype``) cannot
# produce a non-numeric result from non-numeric-free inputs. The kind check
# below is kept anyway as ``rate_for``'s own defensive fallback, not a path
# either function relies on being reachable.
#
# ``get_dtype_rate`` separately fails closed for NUMERIC dtypes absent from
# the supported table (future types numpy or an extension package might
# introduce; every known numpy numeric dtype, including the longdouble
# family, carries an explicit rate).
_NUMERIC_KINDS = frozenset("biufc")


def rate_for(resolved: _np.dtype) -> float:
    if resolved.kind not in _NUMERIC_KINDS:
        return 1.0
    return get_dtype_rate(resolved.name)


def refuse_non_numeric_dtype(op_name: str, *dtypes) -> None:
    """Raise if any participating dtype is not numeric.

    flopscope's meter is ``count x rate``, sound only while a dtype's
    per-element cost is bounded and captured by a single flat rate. The
    predicate is a NUMERIC ALLOWLIST (``dtype.kind in "biufc"``: bool,
    signed/unsigned integer, float, complex) rather than a non-numeric
    denylist, so a dtype kind flopscope has never seen is refused by default
    instead of silently admitted -- see ``_NUMERIC_KINDS`` above for why.

    Two distinct dtype families fall outside the allowlist. An object cell
    is a ``PyObject*`` whose arithmetic dispatches into arbitrary Python of
    unbounded cost, so neither the count axis nor the rate axis observes it.
    str/bytes/structured (void)/datetime64/timedelta64 are bounded, but their
    real per-element cost is not the fixed 32-bit-class unit a flat rate
    assumes -- a wide string or record does more work than a narrow one, and
    datetime64/timedelta64 are integers underneath, at whatever the
    platform's integer rate is, not a dtype-blind flat one. No finite rate
    repairs either case, so every non-numeric dtype fails closed.

    Subsumes ``hasobject``: a structured dtype embedding an object field has
    kind ``'V'``, already outside the allowlist, so it is refused the same
    way as any other structured dtype -- there is no separate object check
    to bypass.

    One exception: a zero-itemsize dtype (an empty structured spec such as
    ``np.dtype([])``, or a zero-length ``'U0'``/``'S0'``) is let through
    regardless of kind. Zero bytes per element cannot embed an object field
    or any itemsize-dependent cost -- it is data-free by construction, the
    same safety property that makes a numeric dtype billable in the first
    place. This is not a hypothetical: NumPy's own ``broadcast_shapes``
    allocates ``np.empty(shape, dtype=np.dtype([]))`` internally as a
    zero-byte shape-computation placeholder, so this exception keeps a
    dtype numpy invents for its own bookkeeping from being refused as if a
    caller had asked to compute something in it.

    Deliberately independent of the dtype-rate table: ``get_dtype_rate``
    returns 1.0 for every name in unit mode, and the test suite resets weights
    for every test, so a table-expressed ban would be silently disabled.

    Python scalars reach the billing tuple via NEP 50 weak promotion and are
    not dtypes; they are skipped rather than raising.
    """
    for candidate in dtypes:
        if candidate is None:
            continue
        try:
            resolved = _np.dtype(candidate)
        except (TypeError, ValueError):
            continue  # a weak Python scalar, not a dtype
        if resolved.kind not in _NUMERIC_KINDS and resolved.itemsize != 0:
            raise UnsupportedDtypeError(
                f"{op_name}: dtype {resolved!r} is not billable -- flopscope "
                "meters only numeric dtypes (bool, integer, float, complex). "
                "object carries unbounded per-element computation; string, "
                "bytes, structured/void, datetime64, and timedelta64 have a "
                "real per-element cost a single flat rate cannot capture. "
                "Refused as an operand, a dtype=, or an out= destination. "
                "Fix: hold mixed/ragged data in a Python list of numeric "
                "arrays, or pre-convert with plain NumPy where it is "
                "available -- clean = np.array(x, dtype=np.float64) -- note "
                "fnp.array/fnp.asarray/fnp.astype refuse non-numeric input "
                "too."
            )


def heavier_billing_dtype(*dtypes: _np.dtype) -> _np.dtype:
    """Return the operand dtype with the highest billing rate (ties -> first).

    Unlike ``np.result_type``, this never promotes to a *third* dtype. Use it
    where the billed cost is the MAX of the operand rates rather than the
    promoted-loop rate -- e.g. ``astype``, which reads the source and writes
    the destination, so it should bill at whichever is pricier. ``result_type``
    would be wrong twice over there: ``result_type(float32, int32) == float64``
    over-charges a narrowing cross-kind cast, while billing the source alone
    under-charges when the destination is pricier (``complex64 -> float64``).
    """
    dts = [_np.dtype(d) for d in dtypes]
    return max(dts, key=rate_for)


_DEFAULT_INT = _np.dtype(_np.int_)
_DEFAULT_UINT = _np.dtype(_np.uint)


def sum_accumulator_dtype(a_dtype: _np.dtype) -> _np.dtype:
    """numpy's accumulator dtype for a ``sum``/``prod``-style reduction with
    ``dtype=None``.

    numpy widens a narrow-integer or boolean reduction to the default platform
    integer to avoid overflow (bool/int8-32 -> int64, uint8-32 -> uint64 on a
    64-bit platform); floats and complex -- including float16 -- keep their own
    dtype. Billing on the input dtype would charge such a reduction at the
    narrow-integer rate even though numpy accumulates at 64-bit, a width
    discount; billing on this accumulator dtype closes it. Keying the threshold
    off the platform's own ``int_`` size keeps it correct where the default
    integer is 32-bit.
    """
    kind, size = a_dtype.kind, a_dtype.itemsize
    if kind == "b" or (kind == "i" and size < _DEFAULT_INT.itemsize):
        return _DEFAULT_INT
    if kind == "u" and size < _DEFAULT_UINT.itemsize:
        return _DEFAULT_UINT
    return a_dtype


def mean_compute_dtype(a_dtype: _np.dtype) -> _np.dtype:
    """numpy's compute dtype for ``mean``/``var``/``std`` with ``dtype=None``.

    numpy evaluates the mean/variance of a boolean or integer array in
    ``float64``; float and complex inputs keep their own dtype (so complex
    variance still carries its complex factor). Billing on the input dtype
    would charge an integer mean/variance at the integer rate even though the
    arithmetic runs in float64.
    """
    if a_dtype.kind in "biu":
        return _np.dtype(_np.float64)
    return a_dtype


def reduction_billing_dtype(
    a_dtype: _np.dtype,
    *,
    explicit_dtype=None,
    out_dtype: _np.dtype | None = None,
    default_dtype: _np.dtype | None = None,
) -> _np.dtype:
    """Billed dtype for a reduction: the accumulator numpy actually runs.

    An explicit ``dtype=`` (positional or keyword) IS numpy's accumulator --
    billed as requested, wider or narrower.

    ``out=`` without ``dtype=`` is NOT an accumulator selector. It can only
    ever *widen*: the family default (integer widening for sum/prod, float64
    for integer mean/var, the ufunc loop dtype for generic methods) is the
    floor, and a narrower ``out`` merely casts the final store. So the
    resolution is ``max(out, family default)`` by rate -- never
    ``out`` alone.

    numpy's own ``ufunc.reduce`` docs are stale on this point. They say
    ``dtype`` "defaults to the data-type of the output array if this is
    provided", which would make ``out=`` replace the default. numpy 2.2.6
    does not behave that way, and two bit-level probes settle it:

    * ``float64`` input ``[1.0, 5e-8, 5e-8, 5e-8]``: bare gives
      ``1.0000001499999998``; ``dtype=float32`` gives ``1.0`` (a true
      float32 accumulation, bits ``1065353216``); ``out=float32`` gives
      ``1.0000001`` (bits ``1065353217``) -- bit-identical to casting the
      float64 accumulation down, i.e. the loop stayed float64.
    * ``int32`` input ``[2**24+1, 1, 1, 1]``: bare accumulates int64 to
      ``16777220``; ``dtype=float32`` gives ``16777216.0`` (the ``+1``s
      vanish into float32 rounding); ``out=float32`` gives ``16777220.0``
      -- again the int64 accumulation, only the store was cast.

    A *wider* ``out=`` does genuinely widen the loop (``float32`` input with
    ``out=float64`` returns the float64-accurate sum, not the float32 sum
    widened), which is why widening is honoured here.

    The final floor at ``a_dtype`` stands on top of all of that: a
    value-testing loop (comparison, logical) reads full-width operands even
    though its output is bool.

    Widening-only also holds on the KIND axis, not just the width axis. A
    real ``out=`` cannot make a complex reduction real:
    ``prod([1+2j, 3+4j, 5+6j], out=float64)`` returns ``-85.0``, the real
    part of the complex product ``(-85+20j)``, NOT the real-only product
    ``15.0`` -- numpy accumulated in complex and the store merely dropped the
    imaginary part. Since the rate axis alone ranks ``float64``/``int64``
    (2.0) above ``complex64`` (1.0), a plain max-by-rate would let such an
    ``out`` carry away the complex factor with it (6.0 -> 1.0 for the prod
    family). The same happens on the ``a_dtype`` floor when the loop dtype is
    bool (``logical_xor.reduce(complex128)``). Both are pinned back below.
    No reduce-capable op has a complex factor under 2.0, so re-imposing a
    complex dtype here can only raise a bill, never lower one.

    A non-numeric dtype anywhere in the call -- the operand, an explicit
    ``dtype=``, ``out=``, or the family default -- must reach ``deduct()``'s
    refusal unchanged, so it is checked and returned before any of the
    folding below runs. Every branch past this point picks a SINGLE winning
    dtype by rate: the ``explicit_dtype`` branch returns it outright
    (dropping ``a_dtype`` entirely), and ``heavier_billing_dtype`` folds two
    into one. Every non-numeric dtype rates 1.0, so a numeric partner with a
    higher rate would silently erase it from the resolution the same way it
    did in ``astype()`` before that was fixed -- e.g. ``sum(object_arr,
    dtype=np.float64)`` returning float64 outright, or ``sum(float64_arr,
    out=m8_arr)`` losing the destination to
    ``heavier_billing_dtype(floor, out_dtype)``. Returning the non-numeric
    dtype here (rather than raising directly) is deliberate: every call site
    threads this return straight into the ``dtypes=`` tuple it hands to
    ``deduct()``/``deduct_after()``, which already refuses it with the
    correct op name -- duplicating the raise here would just race that
    check with a worse error message.
    """
    for candidate in (a_dtype, explicit_dtype, out_dtype, default_dtype):
        if candidate is None:
            continue
        candidate_dtype = _np.dtype(candidate)
        if candidate_dtype.kind not in _NUMERIC_KINDS:
            return candidate_dtype
    if explicit_dtype is not None:
        return _np.dtype(explicit_dtype)
    # The family default is the floor, not something ``out=`` can replace --
    # billing ``out`` alone would hand back half the bill whenever numpy's
    # real accumulator is wider than a narrow ``out`` destination. ``floor``
    # goes first so a rate tie keeps the accumulator rather than the store.
    a_dtype = _np.dtype(a_dtype)
    floor = _np.dtype(default_dtype if default_dtype is not None else a_dtype)
    accumulator = (
        heavier_billing_dtype(floor, out_dtype) if out_dtype is not None else floor
    )
    resolved = heavier_billing_dtype(accumulator, a_dtype)
    # ``out_dtype`` belongs in here alongside the other two. Complex-ness is a
    # property of ANY participating buffer, and the rate axis cannot see it:
    # complex64 and float32 both rate 1.0, so the tie-break above hands the
    # win to whichever came first. Leaving ``out`` out of this list made that
    # tie-break decide the complex factor, and a complex64 destination on a
    # real accumulator lost it -- measured prod at 391,680 -> 65,280, a 6x
    # DROP, which is the same defect this function exists to close, only
    # pointing the other way.
    loop_complex = [d for d in (floor, a_dtype) if d.kind == "c"]
    if loop_complex and resolved.kind != "c":
        # Complex in the LOOP is intrinsic and a real store cannot carry it
        # away: prod of complex64 with out=float64 really does accumulate in
        # complex64 and merely drops the imaginary part on the way out, so the
        # complex factor has been earned.
        #
        # Restoring it must not cost width, though. This function ranks dtypes
        # by RATE, and cannot see the op's complex factor -- so it cannot tell
        # whether a complex64 loop (rate 1.0, factor applied later) outprices a
        # float128 store (rate 4.0, no factor). Swapping the complex dtype in
        # outright is right when it is the wider of the two and wrong when it
        # is not: for sum(complex64) into a float128 destination it would trade
        # rate 4.0 for rate 1.0, and the factor of 2.0 does not make that back.
        # Promoting instead keeps both properties -- complex kind AND the rate
        # already earned. It over-bills the case where the complex loop was
        # genuinely the pricier participant, which is the safe direction for a
        # meter to err, and the honest fix is to thread the op's complex factor
        # in so the comparison can be made on effective cost rather than rate.
        candidate = heavier_billing_dtype(*loop_complex)
        resolved = (
            candidate
            if rate_for(candidate) >= rate_for(resolved)
            else _np.result_type(resolved, candidate)
        )
    elif out_dtype is not None and _np.dtype(out_dtype).kind == "c":
        # Complex arriving only in the STORE is a different claim: the
        # arithmetic was real, and the complex buffer is a participant like
        # any other. So it competes on rate rather than winning outright --
        # but it wins TIES, because complex64 and float32 both rate 1.0 and
        # letting argument order decide the complex factor is what produced a
        # 6x swing in prod. It must not invent width it has not earned: a
        # complex64 store against an int64 accumulator stays int64, which is
        # both the heavier rate and what numpy actually accumulates in.
        resolved = heavier_billing_dtype(_np.dtype(out_dtype), resolved)
    return resolved


def unary_float_loop_dtype(resolved: _np.dtype) -> _np.dtype:
    """Compute dtype of a float-only UNARY ufunc (exp/sin/sqrt family).

    numpy selects the same-size float loop for integer/bool inputs
    (exp(int8) -> float16, exp(int16) -> float32, exp(int32/64) -> float64);
    float and complex inputs keep their own loop. Billing the raw integer
    dtype would charge int32 transcendentals at the 32-bit rate while the
    arithmetic runs in float64.
    """
    if resolved.kind not in "biu":
        return resolved
    if resolved.itemsize <= 1:
        return _np.dtype(_np.float16)
    if resolved.itemsize == 2:
        return _np.dtype(_np.float32)
    return _np.dtype(_np.float64)


def integer_to_float64_min_dtype(resolved: _np.dtype) -> _np.dtype:
    """Floor integer/bool computation at float64; preserve inexact kinds."""
    if resolved.kind in "biu":
        return _np.dtype(_np.float64)
    return resolved


def fft_billing_dtype(input_dtype: _np.dtype) -> _np.dtype:
    """Compute dtype of an FFT: the complex working precision.

    Half-width inputs (float16/float32/complex64) run the complex64 path;
    everything else — float64, complex128, and ALL integer inputs — runs
    complex128. The complex-arithmetic structure is already priced into the
    5N*log2(N) formulas (complex_factor 1.0), so the rate axis carries the
    component width: complex64 bills 1.0, complex128 bills 2.0.
    """
    kind, size = input_dtype.kind, input_dtype.itemsize
    if (kind == "f" and size <= 4) or (kind == "c" and size <= 8):
        return _np.dtype(_np.complex64)
    return _np.dtype(_np.complex128)


def linalg_compute_dtype(resolved: _np.dtype) -> _np.dtype:
    """Compute dtype of a LAPACK-backed linalg op.

    numpy.linalg maps integer/bool inputs to float64 (its _commonType);
    float32/float64/complex inputs keep their own LAPACK driver precision.
    """
    if resolved.kind in "biu":
        return _np.dtype(_np.float64)
    return resolved


def linalg_billing_dtypes(*dtypes) -> tuple:
    """dtypes= tuple for a LAPACK-backed deduct site: one resolved compute dtype.

    Mirrors ``numpy.linalg._commonType``: the moment ANY operand is
    non-inexact (integer/bool), the driver runs in double precision --
    regardless of the other operand's width -- so ``solve(bool_matrix,
    float32_vector)`` computes float64 and ``solve(int8_matrix,
    complex64_vector)`` computes complex128. Only all-inexact operand sets
    keep their promoted single-precision drivers.
    """
    dts = [_np.dtype(d) for d in dtypes]
    if any(dt.kind in "biu" for dt in dts):
        if any(dt.kind == "c" for dt in dts):
            return (_np.dtype(_np.complex128),)
        return (_np.dtype(_np.float64),)
    return (linalg_compute_dtype(_np.result_type(*dts)),)


def complex_factor_for(op_name: str, resolved: _np.dtype) -> float:
    """Complex structure factor for one billed unit of ``op_name``.

    1.0 for real dtypes. For complex dtypes the factor comes from the op's
    registry classification. Ops explicitly marked ``"illegal"`` (numpy raises
    on complex) or ``"exact"`` (contraction family — the call site must supply
    an override) fail closed. Ops with NO classification are free /
    data-movement / blacklisted / unknown: they relocate or allocate whole
    complex values, and a complex value is two real components, so their
    factor is 2.0 (one unit per component). Charged ops are guaranteed an
    explicit factor by ``test_complex_factor_completeness``, so a missing
    classification here is never a charged op silently defaulting.

    A generic ufunc-method name (``"<ufunc>.<method>"``, e.g.
    ``"multiply.reduce"``) that is not itself a registry key falls back to
    its base ufunc's factor: the per-element arithmetic of ``reduce`` /
    ``accumulate`` / ``reduceat`` / ``outer`` / ``at`` IS the base ufunc's.
    Real registry keys that happen to contain dots (``fft.fft``,
    ``linalg.svd``, ``linalg.outer``, ``stats.norm.pdf``) always resolve on
    the direct lookup first, so they are never mistaken for a ufunc method.
    """
    if resolved.kind != "c":
        return 1.0
    from flopscope._registry import REGISTRY

    entry = REGISTRY.get(op_name)
    if entry is None:
        for suffix in _UFUNC_METHOD_SUFFIXES:
            if op_name.endswith(suffix):
                entry = REGISTRY.get(op_name[: -len(suffix)])
                break
    factor = None if entry is None else entry.get("complex_factor")
    if factor == "illegal":
        raise UnsupportedDtypeError(
            f"operation {op_name!r} is not defined for complex dtypes "
            f"(resolved dtype {resolved.name!r})"
        )
    if factor == "exact":
        raise RuntimeError(
            f"operation {op_name!r} computes its complex cost exactly; the call "
            "site must pass complex_factor_override to deduct()/deduct_after()"
        )
    if factor is None:
        # Free / data-movement / blacklisted / unknown op: relocates or
        # allocates whole complex values -- two real components per value.
        return 2.0
    return float(factor)
