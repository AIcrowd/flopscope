"""Participant-shaped expressions that generated corpora cannot produce.

Hand-maintained on purpose: small, highest-signal, and the regression ratchet.
Every future participant-reported failure gets a case here BEFORE it gets a fix.
"""

from __future__ import annotations

from tests.parity.case import Case

_FAST = "tier:fast"

CASES: tuple[Case, ...] = (
    # Family 1 - scalar semantics
    Case(
        id="idiom/scalar-float32-add",
        source="V[0] + V[1]",
        tags=frozenset({"family:1", _FAST}),
    ),
    Case(
        id="idiom/scalar-int-array-dtype",
        source="fnp.array(10) * I",
        tags=frozenset({"family:1"}),
    ),
    # Family 2 - index coercion
    Case(
        id="idiom/float-index",
        source="V[2.5]",
        tags=frozenset({"family:2", _FAST}),
    ),
    Case(
        id="idiom/float-slice-bound",
        source="V[: 7 / 2]",
        tags=frozenset({"family:2"}),
    ),
    # Family 3 - content sniffing
    Case(
        id="idiom/handle-lookalike-string",
        source="fnp.sum('a0')",
        tags=frozenset({"family:3", _FAST}),
    ),
    # Family 4 - partial encoding
    Case(
        id="idiom/complex-scalar-mul",
        source="fnp.astype(V, 'complex64') * 1j",
        tags=frozenset({"family:4", _FAST}),
    ),
    Case(
        id="idiom/slice-bound-remote",
        source="V[: fnp.argmax(V)]",
        tags=frozenset({"family:4"}),
    ),
    Case(
        id="idiom/asarray-complex-list",
        source="fnp.asarray([1 + 2j, 3 - 1j])",
        tags=frozenset({"family:4"}),
    ),
    # Family 5 - response packing
    Case(
        id="idiom/complex-element-read",
        source="fnp.full((2,), 1.0, dtype='complex128')[0]",
        tags=frozenset({"family:5", _FAST}),
    ),
    # Family 6 - container identity
    Case(
        id="idiom/tuple-axis-sum",
        source="fnp.sum(A, axis=(0, 1))",
        tags=frozenset({"family:6", _FAST}),
    ),
    Case(
        id="idiom/split-returns-list",
        source="fnp.split(V, 2) + [V]",
        tags=frozenset({"family:6"}),
    ),
    # Family 7 - exception identity
    Case(
        id="idiom/out-of-bounds-index",
        source="V[99]",
        tags=frozenset({"family:7", _FAST}),
    ),
    # Family 8 - undecodable dtypes
    Case(
        id="idiom/string-array-read",
        source="fnp.asarray(['foo', 'bar']).tolist()",
        tags=frozenset({"family:8", _FAST}),
    ),
    # Family 9 - client surface
    Case(
        id="idiom/fft-rfft",
        source="fnp.fft.rfft(V)",
        tags=frozenset({"family:9", _FAST}),
    ),
    # Family 10 - error wrapping
    Case(
        id="idiom/huge-int-operand",
        source="V * 2**70",
        tags=frozenset({"family:10", _FAST}),
    ),
    # Family 12 - predicate return types
    Case(
        id="idiom/ndim-returns-int",
        source="fnp.ndim(A)",
        tags=frozenset({"family:12", _FAST}),
    ),
    # Family 12 - metered-wrapper return kind, and what downstream arithmetic
    # on it costs. #193: an ndarray result reaches the client as a RemoteArray
    # whose arithmetic is dispatched and billed, a numpy scalar reaches it as a
    # RemoteScalar whose arithmetic is local and free. Both kinds are covered
    # here with a downstream operation attached, so the two backends must agree
    # on the FLOP delta and not only on the value.
    #
    # The scalar cases coerce with float() on purpose. Without it they also
    # trip a SEPARATE, pre-existing divergence that has nothing to do with
    # billing: arithmetic on a numpy scalar stays a numpy scalar in-process
    # (pytype "float32") while RemoteScalar arithmetic re-wraps as a
    # RemoteScalar. Verified present on origin/main, so it is not this batch's
    # to fix; folding it in here would only bury it inside a billing case.
    Case(
        id="idiom/vdot-scalar-downstream",
        source="float(fnp.vdot(V, V) * 2.0)",
        tags=frozenset({"family:12", _FAST}),
    ),
    Case(
        id="idiom/trapezoid-scalar-downstream",
        source="float(fnp.trapezoid(V) + fnp.trapezoid(V))",
        tags=frozenset({"family:12", _FAST}),
    ),
    Case(
        id="idiom/interp-scalar-downstream",
        source="float(fnp.interp(2.5, V, V) * 2.0)",
        tags=frozenset({"family:12"}),
    ),
    Case(
        id="idiom/tensordot-array-downstream",
        source="fnp.tensordot(A, B, axes=1) + fnp.tensordot(A, B, axes=1)",
        tags=frozenset({"family:12", _FAST}),
    ),
    Case(
        id="idiom/diff-array-downstream",
        source="fnp.diff(V) * 2.0",
        tags=frozenset({"family:12", _FAST}),
    ),
)
