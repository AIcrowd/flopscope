"""A metered wrapper's ARRAY result must be a flopscope type; its SCALAR result
must not be.

Both halves of that sentence are load-bearing, and the dividing line is the
server's own ``RequestHandler._pack_result``, which tests
``isinstance(result, np.ndarray)`` BEFORE ``isinstance(result, np.generic)``:

* an **ndarray** result is stored as a handle and reaches the participant as a
  ``RemoteArray``. Arithmetic on a ``RemoteArray`` is dispatched to the server
  and billed. ``fnp.tensordot(...)`` and its siblings used to hand back the raw
  ``numpy.ndarray`` ``_call_numpy`` produced, so the same arithmetic billed 0
  in-process. The local estimate was the wrong one, and wrapping fixes it
  (#193).
* a **numpy scalar** result is packed by value and reaches them as a
  ``RemoteScalar``. Arithmetic on a ``RemoteScalar`` runs locally in Python and
  is billed nothing -- exactly what a numpy scalar does in-process. The two
  already agree at 0. A 0-d ``FlopscopeArray`` is an ``ndarray``, so wrapping a
  scalar result would flip its wire form to a handle and start charging
  downstream arithmetic that is free today. That is a grader repricing, not a
  fix, so ``vdot``/``trapezoid``/``trapz`` and the scalar-returning shapes of
  ``interp``/``corrcoef`` are deliberately left alone.

Guards:

* :func:`test_named_offenders_return_flopscope_types` -- the array-returning
  ops #193 names, asserted strictly. The behavioural proof of the fix.
* :func:`test_scalar_results_stay_numpy_scalars` and
  :func:`test_scalar_results_still_pack_by_value` -- the other half, pinned at
  the wire through the real ``_pack_result``. These fail if a scalar result
  ever gets wrapped, and also if ``_pack_result`` reorders its branches.
* :func:`test_tensordot_result_arithmetic_is_billed` -- the consequence,
  pinned directly: arithmetic on a tensordot result must move ``flops_used``.
* :func:`test_raw_numpy_returning_ops_only_ever_shrink` -- the registry-driven
  sweep. It probes every metered op in the registry, in every probe form the op
  accepts, and compares the set that still returns a raw ``numpy.ndarray``
  against :data:`KNOWN_RAW_RETURN_OPS`. The comparison is exact in both
  directions, so it is a ratchet rather than an exclusion list: a newly added
  wrapper that returns raw turns it red, and a fixed op left in the inventory
  also turns it red. Only ops the sweep actually exercised are judged, so an op
  gated off on the running numpy (``matvec``/``vecmat`` below 2.2, ``vecdot``
  below 2.1) does not make the ratchet version-dependent.

``KNOWN_RAW_RETURN_OPS`` is deliberately not empty. The sweep showed the defect
reaches well past the ops #193 filed: 36 more, concentrated in ``linalg`` and
``fft``, returned raw ndarrays. The 19 ``linalg.*`` entries are now closed by
the module-wide ``wrap_module_returns`` call at the foot of
``flopscope/numpy/linalg/__init__.py``, leaving 17. Those 17 include structured
returns, plain tuples (``histogramdd``) and a ``numpy.matrix`` (``bmat``) that
wrapping would have to handle case by case. Closing them raises local FLOP
estimates further, which needs its own measurement and its own release note; it
is not this change. The inventory records them so the gap is visible and cannot
grow.

Three things the sweep structurally cannot judge, and which therefore carry
their own guards below:

* an op that returns its ``out=`` buffer by identity -- ``_probe_op`` discards
  any result identical to an argument, so the ratchet is blind to exactly the
  breakage ``skip_names={"multi_dot"}`` prevents
  (:func:`test_multi_dot_out_hands_back_the_callers_buffer`). The same
  exclusion leaves ``multi_dot``'s ordinary array-returning shape raw, and the
  inventory cannot record that either, because the only probe form fitting
  ``multi_dot`` yields a scalar and so classifies it clean
  (:func:`test_multi_dot_without_out_is_a_recorded_residual_gap`). The honest
  count is 19 closed, 17 inventoried, 1 closed-off-by-exclusion.
* the *type* of a structured return. The sweep inspects the elements inside a
  tuple, so a wrapper that flattened ``EigResult`` to a bare tuple would sweep
  clean while destroying ``.eigenvalues``/``.U``/``.sign`` attribute access
  (:func:`test_linalg_structured_returns_keep_their_named_type`).
* whether a scalar-returning op is still returning a scalar.
  ``_flopscope_typed`` answers True for a ``FlopscopeArray``, so a 0-d-wrapped
  ``linalg.det`` would read clean in the sweep while flipping the grader's wire
  form from a by-value ``RemoteScalar`` to an array handle. That is what
  ``_SCALAR_RETURNING_PROBES`` and the two tests it feeds are for, and why the
  linalg scalar surface is enumerated there.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray
from flopscope._registry import REGISTRY
from flopscope.errors import UnsupportedFunctionError

_SERVER_SRC = str(Path(__file__).parent.parent / "flopscope-server" / "src")
if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)

from flopscope_server._request_handler import (  # pyright: ignore[reportMissingImports]
    RequestHandler,  # noqa: E402
)
from flopscope_server._session import (  # pyright: ignore[reportMissingImports]
    Session,  # noqa: E402
)

_METERED_CATEGORIES = {
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
}

# Metered ops that still hand back a raw numpy.ndarray. SHRINK ONLY -- never
# add. Each entry is a local/grader billing disagreement that has not been
# closed: the grader stores the result as a handle and bills arithmetic on it,
# the in-process path hands back a raw ndarray and bills 0.
KNOWN_RAW_RETURN_OPS = frozenset(
    {
        "bmat",
        "cross",
        "dot",
        "fft.fftfreq",
        "fft.fftshift",
        "fft.ifftshift",
        "fft.rfftfreq",
        "histogramdd",
        "inner",
        "isfinite",
        "isinf",
        "isnan",
        "matmul",
        "matvec",
        "outer",
        "vecdot",
        "vecmat",
    }
)

# The ops #193 filed, plus vdot and trapz, which the sweep caught alongside
# them. Each must stay clean in the live sweep -- see
# test_fixed_ops_return_flopscope_types_in_the_live_sweep, which checks them
# against the swept result rather than against the inventory literal above.
FIXED_OPS = frozenset(
    {
        "convolve",
        "corrcoef",
        "correlate",
        "cov",
        "diff",
        "ediff1d",
        "gradient",
        "interp",
        "kron",
        "sort_complex",
        "tensordot",
        "trapezoid",
        "trapz",
        "vdot",
    }
)


# ---------------------------------------------------------------------------
# Probe cascade
# ---------------------------------------------------------------------------

_M = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
_N = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
_V = np.array([1.0, 2.0, 3.0, 4.0])
_W = np.array([0.5, 1.5, 2.5, 3.5])
_IV = np.array([1, 2, 3, 4])
_IM = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
_P = np.array([0.2, 0.5, 0.8])

# Every probe form the sweep tries. Written as thunks, and rebuilt for each
# call, because the sweep would otherwise be order-dependent: an op that takes
# a spare positional as ``out=``, and ``copyto``, write THROUGH their arguments,
# so shared module-level probe arrays get clobbered part-way through the sweep
# and later ops are silently judged against corrupted input (a clobbered _M
# stops being positive-definite, and every Cholesky-shaped op after it drops out
# of the sweep instead of being classified).
_PROBE_FORMS: list = [
    lambda: ((_V.copy(),), {}),
    lambda: ((_M.copy(),), {}),
    lambda: ((_V.copy(), _W.copy()), {}),
    lambda: ((_M.copy(), _N.copy()), {}),
    lambda: ((_IM.copy(),), {}),
    lambda: ((_IV.copy(), _IV.copy()), {}),
    lambda: ((_IM.copy(), _IM.copy()), {}),
    lambda: ((_P.copy(),), {}),
    lambda: ((_V.copy(), _V.copy(), _W.copy()), {}),
    lambda: ((4,), {}),
    lambda: (([4, 4],), {}),
    lambda: ((_M.copy(), 1), {}),
    lambda: ((_V.copy(), 1), {}),
    # Forms that reach ops whose ARRAY-returning shape none of the above hits:
    # take needs in-range indices, random.choice needs size=, lexsort needs a
    # sequence of keys. Without these the sweep sees only their scalar form and
    # would report them clean.
    lambda: ((_V.copy(), [0, 1]), {}),
    lambda: ((_V.copy(),), {"size": 2}),
    lambda: (((_V.copy(), _W.copy()),), {}),
    # matrix_power at n=1 hands back its input by identity; n=2 allocates.
    lambda: ((_M.copy(), 2), {}),
]


def _resolve_in(root: object, name: str) -> Any:
    """Walk a dotted op name from ``root``. Returns None if any part is absent.

    Dotted because the scalar probes below name ``linalg.det`` and friends, and
    a plain ``getattr(fnp, "linalg.det")`` silently yields None -- which would
    skip the probe instead of running it.
    """
    obj = root
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _resolve(name: str):
    return _resolve_in(fnp, name)


def _flopscope_typed(value) -> bool:
    """True when every ndarray inside ``value`` is a flopscope type.

    A numpy SCALAR return is deliberately not a defect. The server packs one by
    value, the client turns it into a ``RemoteScalar``, and arithmetic on a
    ``RemoteScalar`` runs locally in Python and is billed nothing -- which is
    exactly what the in-process path does with a numpy scalar. The two agree at
    0, and wrapping the scalar would break that agreement rather than fix it
    (see :func:`test_scalar_results_still_pack_by_value`).

    Only a raw ``numpy.ndarray`` diverges: the server stores it as a handle, the
    client gets a ``RemoteArray``, and the grader bills arithmetic the
    in-process path bills 0 for. That is #193.
    """
    if isinstance(value, list | tuple):
        return all(_flopscope_typed(item) for item in value)
    if isinstance(value, FlopscopeArray):
        return True
    if isinstance(value, np.ndarray):
        # An ndarray subclass that is not FlopscopeArray is acceptable only if
        # it is one of flopscope's own (SymmetricTensor); numpy.matrix is not.
        return type(value) is not np.ndarray and not isinstance(value, np.matrix)
    return True


def _probe_op(fn) -> list:
    """Return a result for EVERY probe form the op accepts.

    Every form, not just the first that fits: several ops return an array for
    one argument shape and a numpy scalar for another (``np.take(m, 1)`` is a
    scalar, ``np.take(v, idx)`` is an array), and stopping at the first fit
    would let a scalar-returning form vouch for an op whose array-returning form
    still hands back a raw ndarray.

    An op gated off on the running numpy raises ``UnsupportedFunctionError`` for
    every form and so yields an empty list, the same as an op no probe fits.
    Both are reported as unexercised rather than as clean -- ``matvec`` and
    ``vecmat`` are gated below numpy 2.2 and ``vecdot`` below 2.1, and treating
    them as clean is what would red the numpy 2.0 and 2.1 CI cells.
    """
    results = []
    for build in _PROBE_FORMS:
        args, kwargs = build()
        try:
            with flops.BudgetContext(flop_budget=10**14, quiet=True):
                result = fn(*args, **kwargs)
        except UnsupportedFunctionError:
            return []
        except Exception:  # noqa: BLE001 -- a probe that does not fit is not a failure
            continue
        # A ufunc reads a trailing positional as ``out=`` and returns that
        # buffer by identity, so probe forms with a spare argument hand back
        # the caller's own raw ndarray. Preserving the caller's object is the
        # documented ``out=`` contract, not an unwrapped return -- wrapping it
        # would break identity. Judge only results the op actually allocated.
        if any(result is arg for arg in args):
            continue
        results.append(result)
    return results


def _sweep_raw_return_ops() -> tuple[set[str], set[str], int]:
    """Return ``(raw, unexercised, exercised_count)`` over every metered op."""
    metered = sorted(
        name
        for name, meta in REGISTRY.items()
        if meta["category"] in _METERED_CATEGORIES
    )
    assert metered, "registry exposes no metered ops -- the sweep would be vacuous"

    raw: set[str] = set()
    unexercised: set[str] = set()
    exercised = 0
    for name in metered:
        fn = _resolve(name)
        if fn is None or not callable(fn):
            unexercised.add(name)
            continue
        results = _probe_op(fn)
        if not results:
            unexercised.add(name)
            continue
        exercised += 1
        if not all(_flopscope_typed(result) for result in results):
            raw.add(name)
    return raw, unexercised, exercised


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_raw_numpy_returning_ops_only_ever_shrink():
    raw, unexercised, exercised = _sweep_raw_return_ops()

    newly_raw = sorted(raw - KNOWN_RAW_RETURN_OPS)
    assert not newly_raw, (
        "Metered op(s) newly return a raw numpy.ndarray. Downstream arithmetic "
        "on these bills 0 in-process while the grader bills it through the "
        "client's RemoteArray (#193). Return _wrap_metered_result(result):\n"
        + "\n".join(newly_raw)
    )

    # Only ops this run actually exercised can be judged. An op gated off on the
    # running numpy (matvec/vecmat below 2.2, vecdot below 2.1) or one no probe
    # form fits produced no result to classify, so leaving it in the inventory
    # is not staleness -- asserting on it here is what reddens the numpy 2.0 and
    # 2.1 CI cells while passing on 2.2+.
    stale = sorted(KNOWN_RAW_RETURN_OPS - raw - unexercised)
    assert not stale, (
        "KNOWN_RAW_RETURN_OPS lists op(s) that now return flopscope types. The "
        "inventory is a ratchet: delete these entries so it keeps shrinking.\n"
        + "\n".join(stale)
    )

    # A floor, not an exclusion list: it stops the sweep silently degrading to
    # no coverage if the probe cascade ever stops fitting the ops.
    assert exercised >= 200, (
        f"probe cascade only exercised {exercised} metered ops; the sweep has "
        "lost coverage and would pass vacuously"
    )


def test_fixed_ops_return_flopscope_types_in_the_live_sweep():
    """The ops this change fixes must be clean in the sweep, not excused by it.

    Deliberately asserted against the swept result rather than against
    ``KNOWN_RAW_RETURN_OPS``: comparing two literals in this file would pass no
    matter what the source does.
    """
    raw, unexercised, _ = _sweep_raw_return_ops()

    still_raw = sorted(FIXED_OPS & raw)
    assert not still_raw, (
        "op(s) fixed for #193 return a raw numpy.ndarray again: " + ", ".join(still_raw)
    )

    missed = sorted(FIXED_OPS & unexercised)
    assert not missed, (
        "the sweep never exercised these fixed op(s), so it cannot vouch for "
        "them; widen the probe cascade: " + ", ".join(missed)
    )


@pytest.mark.parametrize(
    "name, args",
    [
        ("tensordot", (np.ones((4, 4)), np.ones((4, 4)))),
        ("tensordot", (np.ones((3, 3)), np.ones((3, 3)))),
        ("kron", (np.ones((2, 2)), np.ones((2, 2)))),
        ("diff", (_V,)),
        ("gradient", (_V,)),
        ("gradient", (_M,)),
        ("ediff1d", (_V,)),
        ("convolve", (_V, _W)),
        ("correlate", (_V, _W)),
        ("corrcoef", (_M,)),
        ("cov", (_M,)),
        ("interp", (np.array([0.5, 1.5]), _V, _W)),
        ("sort_complex", (np.array([3.0 + 1j, 1.0 + 2j]),)),
        ("trapezoid", (_M,)),
        ("trapz", (_M,)),
    ],
)
def test_named_offenders_return_flopscope_types(name, args):
    """Every probe here is an ARRAY-returning shape -- the case the grader bills.

    ``vdot`` is absent on purpose: it has no array-returning shape, so its
    contract is the scalar one two tests below.
    """
    fn = getattr(fnp, name, None)
    if fn is None:
        pytest.skip(f"fnp.{name} is not available on numpy {np.__version__}")
    expected = getattr(np, name)(*args)
    assert isinstance(expected, np.ndarray | tuple), (
        f"probe for {name} no longer yields an array from numpy itself; pick a "
        "different probe rather than weakening the assertion"
    )
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fn(*args)
    assert _flopscope_typed(result), (
        f"fnp.{name} returned {type(result).__module__}.{type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# The grader contract: a numpy scalar result must STAY a numpy scalar
# ---------------------------------------------------------------------------
#
# ``RequestHandler._pack_result`` tests ``isinstance(result, np.ndarray)``
# BEFORE ``isinstance(result, np.generic)``. An ndarray becomes a stored
# handle and reaches the participant as a ``RemoteArray``, whose arithmetic is
# dispatched to the server and billed. A numpy scalar is packed by value and
# reaches them as a ``RemoteScalar``, whose arithmetic runs locally in Python
# and is billed nothing.
#
# A 0-d ``FlopscopeArray`` IS an ``np.ndarray``. So wrapping a scalar-returning
# metered op would flip its wire form from by-value to a handle, and downstream
# arithmetic on the result would go from 0 grader FLOPs to billed -- a real
# repricing, on a surface where local and grader already agree at 0. These ops
# are therefore deliberately left unwrapped, and these two tests are what keeps
# them that way.

_SCALAR_RETURNING_PROBES: list[tuple[str, tuple]] = [
    ("vdot", (_V, _W)),
    ("trapezoid", (_V,)),
    ("trapz", (_V,)),
    ("interp", (2.5, _V, _W)),
    ("corrcoef", (_V,)),
    # linalg's scalar surface. These are the ops the module-wide
    # ``wrap_module_returns`` call at the foot of ``flopscope/numpy/linalg/
    # __init__.py`` passes over, and the whole change is only grader-neutral
    # because it does. The wrap tests ``isinstance(result, np.ndarray)`` and
    # ``np.generic`` is not an ndarray subclass, so they are untouched today --
    # but nothing asserted that until these entries existed, and the ratchet
    # sweep cannot: ``_flopscope_typed`` returns True for a FlopscopeArray, so
    # a 0-d-wrapped scalar reads clean there.
    ("linalg.det", (_M,)),
    ("linalg.norm", (_V,)),
    ("linalg.cond", (_M,)),
    ("linalg.matrix_rank", (_M,)),
    ("linalg.trace", (_M,)),
    ("linalg.vector_norm", (_V,)),
    ("linalg.matrix_norm", (_M,)),
]


def _scalar_probe_result(name: str, args: tuple):
    fn = _resolve(name)
    if fn is None:
        pytest.skip(f"fnp.{name} is not available on numpy {np.__version__}")
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        try:
            return fn(*args)
        except UnsupportedFunctionError:
            pytest.skip(f"fnp.{name} is gated off on numpy {np.__version__}")


@pytest.mark.parametrize("name, args", _SCALAR_RETURNING_PROBES)
def test_scalar_results_stay_numpy_scalars(name, args):
    """A metered op whose numpy call yields a scalar must hand back a scalar.

    Pinned against numpy's own return kind for the identical call, so this
    tracks numpy rather than a remembered list.
    """
    npfn = _resolve_in(np, name)
    if npfn is None:
        pytest.skip(f"numpy {np.__version__} has no {name}")
    expected = npfn(*args)
    assert isinstance(expected, np.generic), (
        f"probe for {name} no longer yields a numpy scalar from numpy itself; "
        "pick a different probe rather than weakening the assertion"
    )

    result = _scalar_probe_result(name, args)
    assert isinstance(result, np.generic) and not isinstance(result, np.ndarray), (
        f"fnp.{name} returned {type(result).__module__}.{type(result).__name__} "
        f"where numpy returns {type(expected).__name__}. Wrapping a scalar "
        f"result makes the server store it as an array handle instead of "
        f"packing it by value, which changes what the GRADER charges for "
        f"downstream arithmetic (0 -> billed). See #193."
    )


@pytest.mark.parametrize("name, args", _SCALAR_RETURNING_PROBES)
def test_scalar_results_still_pack_by_value(name, args):
    """The consequence, pinned at the wire: by value, not an array handle.

    This is the assertion that actually protects grader billing. It fails if
    the op starts returning an ndarray, and it also fails if ``_pack_result``
    ever reorders its ndarray/generic branches underneath us.
    """
    result = _scalar_probe_result(name, args)

    session = Session(flop_budget=10_000_000)
    handler = RequestHandler(session)
    try:
        packed = handler._pack_result(result)
    finally:
        if session.is_open:
            session.close()

    assert packed["status"] == "ok", packed
    assert "value" in packed["result"], (
        f"fnp.{name} packed as {sorted(packed['result'])} instead of a "
        f"by-value scalar. The client turns a handle into a RemoteArray whose "
        f"arithmetic is billed on the grader; it turns a value into a "
        f"RemoteScalar whose arithmetic is free. This is a grader repricing."
    )


def test_gradient_multi_axis_still_returns_a_sequence():
    """Wrapping must not collapse gradient's multi-axis tuple into one array."""
    expected = np.gradient(_M)
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fnp.gradient(_M)
    assert isinstance(result, tuple)
    assert len(result) == len(expected)
    for got, want in zip(result, expected, strict=True):
        assert np.asarray(got).shape == want.shape
        assert np.allclose(np.asarray(got), want)


def test_tensordot_result_arithmetic_is_billed():
    a = fnp.ones((4, 4))
    b = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget:
        z = fnp.tensordot(a, b, axes=1)
        before = budget.flops_used
        _ = z + z
        assert budget.flops_used > before, (
            "arithmetic on a tensordot result was not billed"
        )


@pytest.mark.parametrize("dest_kind", ["plain", "flopscope"])
def test_multi_dot_out_hands_back_the_callers_buffer(dest_kind):
    """``out=`` returns the caller's own object -- the case the sweep cannot see.

    ``_probe_op`` discards any result identical to an argument, so an
    ``out=``-returning op is invisible to the ratchet. ``linalg.multi_dot`` is
    the one op in ``flopscope.numpy.linalg`` that takes ``out=``, so a blanket
    module-wide wrap would view-cast its return and hand back a new
    ``FlopscopeArray`` instead of the buffer the caller passed in, breaking
    ``out is result`` for a plain-ndarray destination. That is why the wrap
    carries ``skip_names={"multi_dot"}``, and this is the assertion that keeps
    it there.
    """
    dest = np.zeros((3, 3))
    if dest_kind == "flopscope":
        dest = dest.view(FlopscopeArray)
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fnp.linalg.multi_dot([_M.copy(), _N.copy()], out=dest)
    assert result is dest, (
        f"linalg.multi_dot(out=<{dest_kind} ndarray>) returned "
        f"{type(result).__module__}.{type(result).__name__} instead of the "
        "caller's own buffer. numpy's out= contract is identity; wrapping the "
        "return breaks it."
    )
    assert np.allclose(np.asarray(dest), _M @ _N)


def test_multi_dot_without_out_is_a_recorded_residual_gap():
    """The price of ``skip_names={"multi_dot"}``, written down rather than left
    implicit.

    Excluding ``multi_dot`` from the module-wide wrap preserves its ``out=``
    identity contract (the test above), but it also leaves its ordinary
    array-returning shape handing back a raw ``numpy.ndarray`` on plain-numpy
    operands -- the #193 defect, still open for this one op. It was raw before
    this change too, so nothing regressed; what is new is that it can no longer
    be tracked by :data:`KNOWN_RAW_RETURN_OPS`.

    It cannot go in that inventory, because the inventory is exact in both
    directions and the sweep classifies ``multi_dot`` as clean: the only probe
    form that fits it is ``multi_dot([_V, _W])``, whose result is a numpy
    scalar. So the sweep exercises the op, sees a scalar, and is blind to the
    array-returning shape. Listing it would red the ratchet's ``stale`` check.

    Hence this test. The honest count for the change is "19 closed, 17 still
    inventoried, 1 closed-off-by-exclusion and recorded here". When a later
    change closes it -- by making the wrap identity-aware, i.e. skipping the
    wrap only when the result *is* one of the arguments or ``kwargs["out"]``,
    which is the same test ``_probe_op`` already uses -- this test should be
    deleted along with the exclusion.
    """
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fnp.linalg.multi_dot([_M.copy(), _N.copy()])
    assert type(result) is np.ndarray, (
        f"fnp.linalg.multi_dot now returns {type(result).__module__}."
        f"{type(result).__name__} rather than a raw ndarray. If that was "
        "deliberate, delete this test and the skip_names={'multi_dot'} "
        "exclusion together -- but first confirm the out= identity test above "
        "still passes, which is the reason the exclusion exists."
    )


@pytest.mark.parametrize(
    "name, args, fields",
    [
        ("eig", (_M,), ("eigenvalues", "eigenvectors")),
        ("eigh", (_M,), ("eigenvalues", "eigenvectors")),
        ("qr", (_M,), ("Q", "R")),
        ("svd", (_M,), ("U", "S", "Vh")),
        ("slogdet", (_M,), ("sign", "logabsdet")),
    ],
)
def test_linalg_structured_returns_keep_their_named_type(name, args, fields):
    """Wrapping must rebuild the namedtuple, not flatten it to a bare tuple.

    ``numpy.linalg`` hands these back as ``EigResult``/``QRResult``/
    ``SVDResult``/``SlogdetResult``, and participant code reaches for
    ``.eigenvalues``/``.U``/``.sign``. ``wrap_module_returns`` rebuilds the
    type via ``type(result)(*wrapped_elems)``; a tuple-flattening wrapper would
    pass the ratchet sweep (which only inspects element types) while silently
    destroying attribute access.
    """
    expected = getattr(np.linalg, name)(*args)
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = getattr(fnp.linalg, name)(*args)
    assert type(result) is type(expected), (
        f"fnp.linalg.{name} returned {type(result).__name__} where numpy "
        f"returns {type(expected).__name__}"
    )
    for field in fields:
        got = getattr(result, field)
        want = getattr(expected, field)
        # Kind-aware, deliberately: an ``isinstance(got, FlopscopeArray)``
        # escape clause would be satisfied by a 0-d-wrapped ``slogdet.sign``,
        # so the test would stay green through exactly the scalar -> handle
        # flip it exists to catch.
        if isinstance(want, np.generic):
            assert isinstance(got, np.generic) and not isinstance(got, np.ndarray), (
                f"fnp.linalg.{name}.{field} is {type(got).__module__}."
                f"{type(got).__name__} where numpy returns {type(want).__name__}. "
                f"A wrapped scalar reaches the grader as an array handle instead "
                f"of a by-value RemoteScalar, which bills downstream arithmetic "
                f"that is free today. See test_scalar_results_still_pack_by_value."
            )
        else:
            assert isinstance(got, FlopscopeArray), (
                f"fnp.linalg.{name}.{field} is {type(got).__module__}."
                f"{type(got).__name__} where numpy returns {type(want).__name__}"
            )
        assert np.allclose(np.abs(np.asarray(got)), np.abs(np.asarray(want)))


def test_linalg_lstsq_keeps_its_scalar_rank():
    """lstsq's rank element is a numpy integer and must stay one.

    The array elements are wrapped; ``result[2]`` is not an ndarray, so it must
    come back exactly as numpy produced it -- a wrapped 0-d rank would reach the
    grader as an array handle instead of a by-value scalar.
    """
    expected = np.linalg.lstsq(_M, _V[:3], rcond=None)
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fnp.linalg.lstsq(_M, _V[:3], rcond=None)
    assert isinstance(result[2], np.integer) and not isinstance(
        result[2], np.ndarray
    ), (
        f"lstsq rank came back as {type(result[2]).__module__}."
        f"{type(result[2]).__name__}"
    )
    assert result[2].dtype == expected[2].dtype
    assert _flopscope_typed(result)


def test_linalg_tensorsolve_returns_a_flopscope_type():
    """No probe form in the sweep fits tensorsolve, so it needs its own guard."""
    a = np.eye(6).reshape(3, 2, 3, 2)
    b = np.ones((3, 2))
    with flops.BudgetContext(flop_budget=10**14, quiet=True):
        result = fnp.linalg.tensorsolve(a, b)
    assert isinstance(result, FlopscopeArray), (
        f"fnp.linalg.tensorsolve returned {type(result).__module__}."
        f"{type(result).__name__}"
    )


@pytest.mark.parametrize(
    "name, args",
    [
        ("kron", (np.ones((2, 2)), np.ones((2, 2)))),
        ("diff", (_V,)),
        ("cov", (_M,)),
        ("trapezoid", (_M,)),
    ],
)
def test_arithmetic_downstream_of_fixed_ops_is_billed(name, args):
    fn = getattr(fnp, name)
    with flops.BudgetContext(flop_budget=10**14, quiet=True) as budget:
        result = fn(*args)
        before = budget.flops_used
        _ = result + result
        assert budget.flops_used > before, (
            f"arithmetic on an fnp.{name} result was not billed"
        )
