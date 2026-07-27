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

Why the coverage here is DERIVED rather than hand-listed
--------------------------------------------------------
``_normalize_out`` is called from ~48 wrapper sites. A hand-written
parametrize list pins whichever of them the author happened to think of, and
mutation testing says exactly what that is worth: with a hand-listed set of
seven ops, the guard could be deleted from the other 34 sites and the entire
repository suite still passed.

So the op set under test is discovered from the registry and the module
surface, and the arguments are generated: an op that accepts ``out=`` is
enumerated whether or not anyone remembered it, and one added later is
enumerated the day it lands. Ops the generator legitimately cannot drive are
listed in ``_NOT_DRIVEN`` **with a reason**, and that list is asserted against
the discovered set from both directions, so it can neither grow silently nor
keep a stale entry.

The invariants, for every op that takes a destination:

(a) ``flops_used`` never decreases — no refunds, anywhere;
(b) a refused ``out=`` form costs ZERO;
(c) ``out=(d,)`` bills EXACTLY what ``out=d`` bills, and returns ``d`` itself.
"""

from __future__ import annotations

import collections
import functools
import inspect
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import flopscope as fl
import flopscope.numpy as fnp
from flopscope._budget import get_active_budget
from flopscope._registry import REGISTRY
from flopscope._symmetric import SymmetricTensor

SEED = 20260727


@pytest.fixture()
def budget():
    with fl.BudgetContext(flop_budget=10**14, quiet=True) as ctx:
        yield ctx


def _rng():
    return np.random.default_rng(SEED)


def _f32(*shape):
    # Asymmetric random data on purpose: a constant fill on a square shape
    # picks up an inferred symmetry tag, which changes the accumulation cost
    # and would pin symmetry-discounted numbers instead of the real ones.
    return fnp.asarray(_rng().standard_normal(shape).astype("float32"))


def _billed(ctx, fn, *args, **kwargs):
    before = ctx.flops_used
    result = fn(*args, **kwargs)
    return ctx.flops_used - before, result


# ---------------------------------------------------------------------------
# Discovery: which ops take a destination at all
# ---------------------------------------------------------------------------


def _walk(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _resolve(name: str) -> Any:
    """The flopscope callable a registry/surface name refers to."""
    return _walk(fnp, name) or _walk(fl, name)


def _numpy_takes_out(np_fn: Any) -> bool:
    """Does numpy's own callable accept ``out=``?

    Needed only for wrappers that take ``out`` inside ``**kwargs`` rather than
    as a declared parameter — the exact shape that let this whole family of
    ops escape the normalization every sibling gets. ``np.concatenate`` is a C
    builtin with no introspectable signature, so its synthesized docstring
    header is the only machine-readable statement that it takes a destination.
    """
    if np_fn is None:
        return False
    if isinstance(np_fn, np.ufunc):
        return True
    try:
        return "out" in inspect.signature(np_fn).parameters
    except (TypeError, ValueError):
        header = (getattr(np_fn, "__doc__", None) or "").lstrip().split("\n\n", 1)[0]
        return "out=" in header


def _advertised_out_kind(fs_fn: Any) -> str | None:
    try:
        sig = inspect.signature(fs_fn)
    except (TypeError, ValueError):
        return None
    if "out" in sig.parameters:
        return "declared"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return "kwargs"
    return None


def _declares_out_in_its_own_code(fs_fn: Any) -> bool:
    """Does some wrapper in the chain take an ``out`` parameter of its own?

    Every wrapper republishes numpy's signature via ``_apply_numpy_signature``,
    so ``inspect.signature`` reports what numpy takes — which is not always
    what flopscope takes. ``fnp.real`` accepts ``out=`` while advertising
    ``np.real``'s ``(val)``; discovery that trusted the advertised signature
    alone missed it, and mutation testing found the miss by deleting its guard
    with nothing failing. The ``__wrapped__`` chain still has the real
    parameter list, so ask that too.

    Only ``out`` is read from the chain, never ``**kwargs``: every counted
    wrapper is ``(*args, **kwargs)`` plumbing, so a varkw frame says nothing
    about the op — ``fnp.matmul`` has one and takes no destination at all.
    """
    seen: set[int] = set()
    fn = fs_fn
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        code = getattr(fn, "__code__", None)
        if code is not None:
            named = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
            if "out" in named:
                return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _out_kind(fs_fn: Any, np_fn: Any) -> str | None:
    """``"declared"``, ``"kwargs"``, or None if the op takes no destination."""
    advertised = _advertised_out_kind(fs_fn)
    if advertised == "declared" or _declares_out_in_its_own_code(fs_fn):
        return "declared"
    if advertised == "kwargs":
        # A ``**kwargs`` wrapper only forwards a destination if numpy has one
        # to receive; without that check every wrapper in the codebase would
        # look like it took an out=.
        return "kwargs" if _numpy_takes_out(np_fn) else None
    return None


def _discover_out_ops() -> dict[str, str]:
    """Every ``out=``-accepting op, from the registry AND the module surface.

    Both sources, because neither alone is complete: the registry knows about
    methods that live on a class (``random.Generator.random``) which no
    attribute walk over the namespace reaches, and the surface knows about
    anything exposed to a caller that the registry has not caught up with
    (``isnat``, whose registry entry is ``blacklisted``).
    """
    found: dict[str, str] = {}
    for name in REGISTRY:
        fs_fn = _resolve(name)
        if fs_fn is None or not callable(fs_fn):
            continue
        kind = _out_kind(fs_fn, _walk(np, name))
        if kind:
            found[name] = kind

    def scan(fs_mod: Any, np_mod: Any, prefix: str) -> None:
        for attr in dir(fs_mod):
            if attr.startswith("_"):
                continue
            obj = getattr(fs_mod, attr)
            if isinstance(obj, types.ModuleType) or not callable(obj):
                continue
            kind = _out_kind(obj, getattr(np_mod, attr, None) if np_mod else None)
            if kind:
                found.setdefault(prefix + attr, kind)

    scan(fnp, np, "")
    scan(fnp.fft, np.fft, "fft.")
    scan(fnp.linalg, np.linalg, "linalg.")
    return found


#: Ops that accept ``out=`` but that the generic driver cannot exercise, each
#: with the reason. Every entry is asserted against below — a stale one fails
#: :func:`test_the_not_driven_list_is_neither_stale_nor_a_silent_skip`, and an
#: op added later that nobody drives fails
#: :func:`test_every_out_accepting_op_is_actually_driven`. The list can only
#: shrink or be argued for; it cannot grow by accident.
_NOT_DRIVEN: dict[str, str] = {
    # numpy's own out= for these is NOT the ufunc protocol: it takes an array
    # and nothing else, and refuses a 1-tuple itself. Normalizing here would
    # make flopscope MORE permissive than numpy, so there is no tuple parity
    # to assert. The premise is pinned in
    # test_the_ops_numpy_itself_refuses_a_container_for.
    "choose": "np.choose's out= refuses a 1-tuple in numpy itself",
    "random.Generator.random": "Generator.random's out= refuses a container in numpy itself",
    "random.Generator.standard_normal": (
        "Generator.standard_normal's out= refuses a container in numpy itself"
    ),
    "random.Generator.standard_exponential": (
        "Generator.standard_exponential's out= refuses a container in numpy itself"
    ),
    "random.Generator.standard_gamma": (
        "Generator.standard_gamma's out= refuses a container in numpy itself"
    ),
    "random.Generator.permuted": (
        "Generator.permuted's out= refuses a container in numpy itself"
    ),
    # nout=2: the tuple IS the protocol, so "a 1-tuple bills like a bare
    # array" is not the invariant. Driven by test_a_multi_output_op_* instead.
    "modf": "multi-output (nout=2); driven by the multi-output tests",
    "frexp": "multi-output (nout=2); driven by the multi-output tests",
    "divmod": "multi-output (nout=2); driven by the multi-output tests",
}

#: The ops in ``_NOT_DRIVEN`` whose stated reason is "numpy itself refuses a
#: container here", paired with a call that demonstrates it against RAW numpy.
_NUMPY_REFUSES_THE_CONTAINER: dict[str, Callable[[Any], Any]] = {
    "choose": lambda out: np.choose(
        np.array([0, 1] * 4), [np.zeros(8), np.ones(8)], out=out
    ),
    "random.Generator.random": lambda out: np.random.default_rng(0).random(out=out),
    "random.Generator.standard_normal": lambda out: np.random.default_rng(
        0
    ).standard_normal(out=out),
    "random.Generator.standard_exponential": lambda out: np.random.default_rng(
        0
    ).standard_exponential(out=out),
    "random.Generator.standard_gamma": lambda out: np.random.default_rng(
        0
    ).standard_gamma(1.0, out=out),
    "random.Generator.permuted": lambda out: np.random.default_rng(0).permuted(
        np.arange(8.0), out=out
    ),
}

_DISCOVERED = _discover_out_ops()
_DRIVEN_NAMES = sorted(set(_DISCOVERED) - set(_NOT_DRIVEN))


# ---------------------------------------------------------------------------
# Driving: generated arguments, so a new op needs no new hand-written call
# ---------------------------------------------------------------------------


def _arr(shape: tuple[int, ...], kind: str = "float", offset: int = 0):
    rng = np.random.default_rng(SEED + offset)
    if kind == "float":
        # In (0.25, 0.75): inside the domain of every unary op on the surface
        # (log, sqrt, arcsin, arctanh, ...), and never a constant fill, which
        # on a square shape would pick up an inferred symmetry tag.
        values: Any = rng.random(shape) * 0.5 + 0.25
    elif kind == "int":
        values = rng.integers(1, 5, size=shape)
    elif kind == "bool":
        values = rng.random(shape) > 0.5
    elif kind == "complex":
        values = rng.random(shape) + 1j * rng.random(shape)
    elif kind == "datetime":
        values = np.arange(shape[0], dtype="int64").astype("datetime64[ns]")
    else:  # pragma: no cover - typo guard
        raise AssertionError(f"unknown kind {kind}")
    return fnp.asarray(values)


#: Argument shapes tried in order for each op; the first one the op accepts
#: (and answers with an ndarray) becomes its driver. Deliberately a battery
#: rather than a per-category lookup: registry categories are not a reliable
#: guide to a signature (``isclose`` is tagged unary and is binary), and an
#: invalid call is filtered for free.
_RECIPES: list[tuple[str, Callable[[], tuple[tuple, dict]]]] = [
    ("unary-1d", lambda: ((_arr((64,)),), {})),
    ("unary-2d", lambda: ((_arr((6, 5)),), {})),
    ("unary-int", lambda: ((_arr((64,), "int"),), {})),
    ("unary-bool", lambda: ((_arr((64,), "bool"),), {})),
    ("unary-datetime", lambda: ((_arr((8,), "datetime"),), {})),
    ("binary-1d", lambda: ((_arr((64,)), _arr((64,), offset=1)), {})),
    ("binary-int", lambda: ((_arr((64,), "int"), _arr((64,), "int", 1)), {})),
    ("binary-bool", lambda: ((_arr((64,), "bool"), _arr((64,), "bool", 1)), {})),
    ("reduce-axis", lambda: ((_arr((6, 5)),), {"axis": -1})),
    ("reduce-axis-int", lambda: ((_arr((6, 5), "int"),), {"axis": -1})),
    ("sequence", lambda: (([_arr((6, 5)), _arr((6, 5), offset=1)],), {})),
    ("matrix", lambda: ((_arr((6, 6)),), {})),
    ("quantile", lambda: ((_arr((64,)), 0.5), {})),
    ("clip", lambda: ((_arr((64,)), 0.3, 0.6), {})),
    ("take", lambda: ((_arr((64,)), np.arange(8)), {})),
    ("compress", lambda: ((np.array([True, False] * 32), _arr((64,))), {})),
    ("outer", lambda: ((_arr((16,)), _arr((16,), offset=1)), {})),
    ("einsum", lambda: (("ij,jk->ik", _arr((6, 6)), _arr((6, 4), offset=1)), {})),
    ("matvec", lambda: ((_arr((6, 6)), _arr((6,), offset=1)), {})),
    ("vecmat", lambda: ((_arr((6,)), _arr((6, 6), offset=1)), {})),
    (
        "chain",
        lambda: (
            (
                [
                    _arr((6, 6)),
                    _arr((6, 6), offset=1),
                    _arr((6, 6), offset=2),
                ],
            ),
            {},
        ),
    ),
    ("complex-1d", lambda: ((_arr((64,), "complex"),), {})),
]

_RECIPES_BY_LABEL = dict(_RECIPES)

#: Destinations tried in order. The WIDER ones come first on purpose: with a
#: destination of the natural result dtype, dropping the normalization changes
#: nothing measurable — the tuple reaches numpy, numpy unwraps it, and the
#: billing fold that got skipped had nothing to fold. It is the wide buffer
#: that makes the skipped fold visible as a cheaper bill.
_DEST_DTYPES = ("complex128", "float64", None)


def _dest(shape, dtype):
    # Built from a plain numpy zeros: fnp.zeros on a square shape infers a
    # symmetry tag, which is a billing discount, not a neutral destination.
    return fnp.asarray(np.zeros(shape, dtype))


def _widest_accepted_dest(call_with_dest, shape, natural) -> str:
    """The widest destination dtype the op accepts, per ``_DEST_DTYPES``.

    Not every op can be handed a wider buffer — ``argmax`` writes indices and
    refuses anything but an integer one — so the ladder ends at the natural
    result dtype, which every op accepts by construction.
    """
    for dtype in _DEST_DTYPES:
        chosen = dtype or str(np.asarray(natural).dtype)
        try:
            _quietly(call_with_dest, _dest(shape, chosen))
        except Exception:
            continue
        return chosen
    raise AssertionError(f"no destination dtype accepted for shape {shape}")


def _quietly(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run *fn* under a budget, whichever one is available.

    BudgetContexts do not nest, and these calls happen both inside a test
    (where the ``budget`` fixture's context is already open) and during driver
    resolution. Everything routed through here is setup — building an input,
    finding out what shape an op answers with — and every one of those calls
    is made OUTSIDE the measured window, so charging it to the test's own
    budget changes no measurement.
    """
    if get_active_budget() is not None:
        return fn(*args, **kwargs)
    with fl.BudgetContext(flop_budget=10**16, quiet=True):
        return fn(*args, **kwargs)


@functools.cache
def _drivers() -> dict[str, tuple[str, str]]:
    """``{op: (recipe label, destination dtype)}`` for every drivable op.

    Resolved once per process. Only the labels are cached — arrays are rebuilt
    per test, so nothing carries budget or write-epoch state between tests.
    """
    resolved: dict[str, tuple[str, str]] = {}
    for name in _DRIVEN_NAMES:
        fn = _resolve(name)
        if fn is None:
            continue
        for label, make in _RECIPES:
            try:
                args, kwargs = make()
                natural = _quietly(fn, *args, **kwargs)
            except Exception:
                continue
            if not isinstance(natural, np.ndarray):
                continue
            try:
                args, kwargs = make()
                # Defaults bind the loop variables into the closure. The
                # call is immediate, but ruff cannot know that (B023).
                dtype = _widest_accepted_dest(
                    lambda d, fn=fn, args=args, kwargs=kwargs: fn(
                        *args, out=d, **kwargs
                    ),
                    natural.shape,
                    natural,
                )
            except AssertionError:
                continue
            resolved[name] = (label, dtype)
            break
    return resolved


def _driven_call(name: str):
    """``(fn, args, kwargs, shape, dtype)`` for driving *name* with an ``out=``."""
    driver = _drivers().get(name)
    if driver is None:
        pytest.fail(
            f"{name} accepts out= but no recipe in _RECIPES can call it. Add one, "
            f"or add {name!r} to _NOT_DRIVEN with the reason it cannot be driven."
        )
    label, dtype = driver
    fn = _resolve(name)
    args, kwargs = _RECIPES_BY_LABEL[label]()
    natural = _quietly(fn, *args, **kwargs)
    return fn, args, kwargs, natural.shape, dtype


def _same_contents(a, b) -> bool:
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    except TypeError:  # dtypes with no NaN (int, bool, datetime)
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


# ---------------------------------------------------------------------------
# The coverage net itself
# ---------------------------------------------------------------------------


def test_every_out_accepting_op_is_actually_driven():
    """The list of ops under test is derived, and this is what makes it honest.

    A hand-listed parametrize set is only as good as the author's memory of
    the wrapper surface; this asserts the set is the whole surface.
    """
    missing = sorted(set(_DRIVEN_NAMES) - set(_drivers()))
    assert not missing, (
        f"{len(missing)} op(s) accept out= but no recipe drives them: {missing}. "
        f"Add a recipe to _RECIPES, or an entry to _NOT_DRIVEN with a reason."
    )
    assert len(_drivers()) >= 150, (
        f"only {len(_drivers())} ops driven — discovery has silently narrowed "
        f"(it found {len(_DISCOVERED)} out=-accepting ops)"
    )


def test_the_not_driven_list_is_neither_stale_nor_a_silent_skip():
    stale = sorted(set(_NOT_DRIVEN) - set(_DISCOVERED))
    assert not stale, (
        f"_NOT_DRIVEN names ops that no longer accept out= (or no longer "
        f"exist): {stale} — remove them so the list keeps meaning something"
    )
    unexplained = sorted(n for n, why in _NOT_DRIVEN.items() if not why.strip())
    assert not unexplained, f"_NOT_DRIVEN entries without a reason: {unexplained}"
    overlap = sorted(set(_NOT_DRIVEN) & set(_drivers()))
    assert not overlap, (
        f"{overlap} are listed as undrivable but the generic driver drives "
        f"them — delete the excuse"
    )
    # A written reason is a claim; each one has to be backed by a test that
    # would fail if the claim stopped being true. Without this, "cannot be
    # driven" degrades into "skipped, with prose".
    demonstrated = set(_NUMPY_REFUSES_THE_CONTAINER) | set(_MULTI_OUTPUT_OPS)
    undemonstrated = sorted(set(_NOT_DRIVEN) - demonstrated)
    assert not undemonstrated, (
        f"{undemonstrated} are excused from the generic driver with a reason "
        f"that nothing asserts — add them to _NUMPY_REFUSES_THE_CONTAINER, or "
        f"give them a test of their own"
    )


@pytest.mark.parametrize("name", sorted(_NUMPY_REFUSES_THE_CONTAINER))
def test_the_ops_numpy_itself_refuses_a_container_for(name):
    """The premise behind six ``_NOT_DRIVEN`` entries, asserted against numpy.

    These take ``out=`` but not through the ufunc protocol: numpy wants the
    array itself and refuses a tuple. Unwrapping one for them would make
    flopscope more permissive than numpy, so there is no parity to assert —
    and if numpy ever adopts the protocol here, this test says so.
    """
    call = _NUMPY_REFUSES_THE_CONTAINER[name]
    dest = np.zeros(8)
    call(dest)  # the bare array is the accepted spelling
    with pytest.raises((TypeError, ValueError)):
        call((dest,))


@pytest.mark.parametrize("name", _DRIVEN_NAMES)
def test_a_one_tuple_out_bills_exactly_like_a_bare_out(budget, name):
    """Invariant (c), over the whole ``out=`` surface.

    The guard's own tests going green is NOT evidence the under-bill closed: a
    prototype had every wrapped-out test passing while
    ``add(f32, 0.0, out=(f64,))`` still billed half price. What closes it is
    the wide destination — see ``_DEST_DTYPES``.
    """
    fn, args, kwargs, shape, dtype = _driven_call(name)
    bare_dest, tuple_dest = _dest(shape, dtype), _dest(shape, dtype)

    bare_cost, bare_result = _billed(budget, fn, *args, out=bare_dest, **kwargs)
    tuple_cost, tuple_result = _billed(budget, fn, *args, out=(tuple_dest,), **kwargs)

    assert tuple_cost == bare_cost, (
        f"{name}: out=(dest,) billed {tuple_cost} where out=dest billed "
        f"{bare_cost} — the destination's dtype is not reaching the rate"
    )
    assert bare_result is bare_dest, f"{name} did not hand back out=dest"
    assert tuple_result is tuple_dest, f"{name} handed back the container, not dest"
    assert _same_contents(bare_dest, tuple_dest), (
        f"{name}: out=(dest,) left the destination holding something else — "
        f"the result went into a temporary built from the container"
    )


@pytest.mark.parametrize("name", _DRIVEN_NAMES)
def test_a_refused_out_form_costs_nothing(budget, name):
    """Invariants (a) and (b), over the whole ``out=`` surface.

    numpy accepts a list in none of its ``out=`` surfaces, so neither do we —
    and the refusal has to happen before ``budget.deduct``, which is the part
    that used to fail: the guard ran inside the deduct block, so the op billed
    in full and then raised.
    """
    fn, args, kwargs, shape, dtype = _driven_call(name)
    # Built BEFORE the measurement: allocating a destination costs FLOPs of
    # its own, and building it inside is how "refusal is free" quietly
    # measures the wrong thing.
    dest = _dest(shape, dtype)
    before = budget.flops_used

    with pytest.raises(TypeError, match="out= must be an array"):
        fn(*args, out=[dest], **kwargs)

    assert budget.flops_used == before, (
        f"{name} was billed {budget.flops_used - before} for refusing out=[dest]"
    )
    assert budget.flops_used >= before, "flops_used must never decrease"
    assert not np.asarray(dest).any(), f"{name} wrote to dest while refusing it"


# ---------------------------------------------------------------------------
# The ufunc-method surface: np.add.outer / .reduce / .accumulate / .reduceat
# ---------------------------------------------------------------------------
#
# Reached only through ``FlopscopeArray.__array_ufunc__``, so no attribute walk
# over ``flopscope.numpy`` finds them and the module-level scan above is blind
# to them: ``fnp.add`` is a plain function with no ``.reduce``. Four separate
# ``_normalize_out`` sites live behind this surface, and deleting any of them
# left every other test in the repository green.


def _binary_ufuncs() -> list[tuple[str, np.ufunc]]:
    found = []
    for name, entry in REGISTRY.items():
        if entry.get("module") != "numpy":
            continue
        ufunc = getattr(np, name, None)
        if not isinstance(ufunc, np.ufunc) or ufunc.nin != 2 or ufunc.nout != 1:
            continue
        # matmul/vecdot/matvec/vecmat are gufuncs: numpy refuses these methods
        # on any ufunc with a non-trivial core signature, so there is no call
        # to make. Structural, not a judgement call.
        if ufunc.signature is not None:
            continue
        found.append((name, ufunc))
    return sorted(found)


_UFUNC_METHODS = ("outer", "reduce", "accumulate", "reduceat")

#: ``(ufunc, method)`` pairs numpy cannot resolve a loop for, with the reason.
#: Asserted against numpy in test_the_ufunc_method_pairs_numpy_cannot_resolve.
_NO_UFUNC_METHOD_LOOP: dict[tuple[str, str], str] = {
    ("ldexp", "accumulate"): "ldexp is (float, int) -> float: no same-dtype loop",
    ("ldexp", "reduceat"): "ldexp is (float, int) -> float: no same-dtype loop",
}

_UFUNC_METHOD_PAIRS = [
    (name, method)
    for name, _ in _binary_ufuncs()
    for method in _UFUNC_METHODS
    if (name, method) not in _NO_UFUNC_METHOD_LOOP
]


def _ufunc_method_call(ufunc: np.ufunc, method: str, x, y, out):
    if method == "outer":
        return ufunc.outer(x, y, out=out)
    if method == "reduce":
        return ufunc.reduce(x, out=out)
    if method == "accumulate":
        return ufunc.accumulate(x, out=out)
    return ufunc.reduceat(x, [0, 4], out=out)


@functools.cache
def _ufunc_method_drivers() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(ufunc, method): (operand kind, destination dtype)}``."""
    resolved: dict[tuple[str, str], tuple[str, str]] = {}
    for name, ufunc in _binary_ufuncs():
        for method in _UFUNC_METHODS:
            if (name, method) in _NO_UFUNC_METHOD_LOOP:
                continue
            for kind in ("float", "int", "bool"):
                x, y = _arr((8,), kind), _arr((8,), kind, 1)
                try:
                    natural = _quietly(_ufunc_method_call, ufunc, method, x, y, None)
                except Exception:
                    continue
                if not isinstance(natural, np.ndarray):
                    continue
                try:
                    dtype = _widest_accepted_dest(
                        lambda d, u=ufunc, m=method, x=x, y=y: _ufunc_method_call(
                            u, m, x, y, d
                        ),
                        natural.shape,
                        natural,
                    )
                except AssertionError:
                    continue
                resolved[(name, method)] = (kind, dtype)
                break
    return resolved


def _ufunc_method_setup(name: str, method: str):
    driver = _ufunc_method_drivers().get((name, method))
    if driver is None:
        pytest.fail(
            f"np.{name}.{method} takes out= but could not be driven. Either it "
            f"stopped working, or it belongs in _NO_UFUNC_METHOD_LOOP with a reason."
        )
    kind, dtype = driver
    ufunc = getattr(np, name)
    x, y = _arr((8,), kind), _arr((8,), kind, 1)
    natural = _quietly(_ufunc_method_call, ufunc, method, x, y, None)
    return ufunc, x, y, natural.shape, dtype


#: Reaching these methods means calling ``np.add.outer(...)`` on a flopscope
#: array, which is exactly the spelling the auto-route notice exists to
#: discourage. Here it is the surface under test, not a mistake.
_AUTO_ROUTE_NOTICE = "ignore:np\\..* auto-routed to fnp"


@pytest.mark.filterwarnings(_AUTO_ROUTE_NOTICE)
@pytest.mark.parametrize("name,method", _UFUNC_METHOD_PAIRS)
def test_a_ufunc_method_bills_a_one_tuple_like_a_bare_destination(budget, name, method):
    ufunc, x, y, shape, dtype = _ufunc_method_setup(name, method)
    bare_dest, tuple_dest = _dest(shape, dtype), _dest(shape, dtype)

    bare_cost, bare_result = _billed(
        budget, _ufunc_method_call, ufunc, method, x, y, bare_dest
    )
    tuple_cost, tuple_result = _billed(
        budget, _ufunc_method_call, ufunc, method, x, y, (tuple_dest,)
    )

    assert tuple_cost == bare_cost, (
        f"{name}.{method}: out=(dest,) billed {tuple_cost}, out=dest {bare_cost}"
    )
    assert bare_result is bare_dest and tuple_result is tuple_dest
    assert _same_contents(bare_dest, tuple_dest)


@pytest.mark.filterwarnings(_AUTO_ROUTE_NOTICE)
@pytest.mark.parametrize("name,method", _UFUNC_METHOD_PAIRS)
def test_a_ufunc_method_refuses_a_list_for_free(budget, name, method):
    ufunc, x, y, shape, dtype = _ufunc_method_setup(name, method)
    dest = _dest(shape, dtype)
    before = budget.flops_used

    with pytest.raises(TypeError, match="out= must be an array"):
        _ufunc_method_call(ufunc, method, x, y, [dest])

    assert budget.flops_used == before, (
        f"{name}.{method} was billed for refusing out=[dest]"
    )
    assert not np.asarray(dest).any()


@pytest.mark.parametrize(
    "name,method", sorted(_NO_UFUNC_METHOD_LOOP), ids=lambda v: str(v)
)
def test_the_ufunc_method_pairs_numpy_cannot_resolve(name, method):
    """The premise behind ``_NO_UFUNC_METHOD_LOOP``, asserted against numpy."""
    ufunc = getattr(np, name)
    x = np.linspace(0.25, 0.75, 8)
    with pytest.raises(TypeError):
        _ufunc_method_call(ufunc, method, x, x, None)


# ---------------------------------------------------------------------------
# Multi-output out= is a different protocol and must not be unwrapped
# ---------------------------------------------------------------------------

_MULTI_OUTPUT_OPS: dict[str, Callable[[Any], Any]] = {
    "modf": lambda out: fnp.modf(_arr((8,)), out=out),
    "frexp": lambda out: fnp.frexp(_arr((8,)), out=out),
    "divmod": lambda out: fnp.divmod(_arr((8,)), _arr((8,), offset=1), out=out),
}


def _multi_output_dests(name: str):
    natural = _quietly(_MULTI_OUTPUT_OPS[name], None)
    return tuple(_dest(part.shape, str(part.dtype)) for part in natural)


@pytest.mark.parametrize("name", sorted(_MULTI_OUTPUT_OPS))
def test_a_multi_output_op_accepts_its_tuple_and_bills_it_like_no_out(budget, name):
    call = _MULTI_OUTPUT_OPS[name]
    dests = _multi_output_dests(name)

    without_cost, _ = _billed(budget, call, None)
    with_cost, result = _billed(budget, call, dests)

    assert with_cost == without_cost, (
        f"{name}: out=(d1, d2) billed {with_cost} where no destination bills "
        f"{without_cost}"
    )
    assert tuple(result) == dests, f"{name} did not hand back the destinations"


@pytest.mark.parametrize("name", sorted(_MULTI_OUTPUT_OPS))
@pytest.mark.parametrize("form", ["bare", "one-tuple", "list"])
def test_a_multi_output_op_refuses_a_single_destination_for_free(budget, name, form):
    """A bare array names one destination; a two-output ufunc needs one per
    output. numpy deprecated that spelling in 1.10 and made it a hard error in
    gh-14682 — refusing it here is what makes it free rather than charged.
    """
    call = _MULTI_OUTPUT_OPS[name]
    dests = _multi_output_dests(name)
    bad = {"bare": dests[0], "one-tuple": (dests[0],), "list": list(dests)}[form]
    expected = ValueError if form == "one-tuple" else TypeError
    before = budget.flops_used

    with pytest.raises(expected):
        call(bad)

    assert budget.flops_used == before, f"{name} was billed for refusing out={form}"


# ---------------------------------------------------------------------------
# Reductions take out= through TWO channels, and both need the guard
# ---------------------------------------------------------------------------
#
# ``_counted_reduction``'s wrapper signature is ``(a, axis=None, *args,
# **kwargs)``: ``out`` arrives either as a keyword or in a positional slot
# whose index comes from numpy's own parameter list. They are separate
# ``_normalize_out`` calls, and pinning one leaves the other free to rot —
# the keyword one being the common spelling for sum/prod/cumsum/argmax.

#: Reduction ops whose ``out=`` cannot be reached positionally, with the
#: reason. Asserted from both sides below.
_NO_POSITIONAL_OUT: dict[str, str] = {
    "average": "takes no out= at all (weights-only signature)",
    "count_nonzero": "takes no out= at all",
    "cumulative_prod": "axis/dtype before out are keyword-only (array-API spelling)",
    "cumulative_sum": "axis/dtype before out are keyword-only (array-API spelling)",
    "percentile": "q has no default, so no positional prefix reaches out",
    "quantile": "q has no default, so no positional prefix reaches out",
    "nanpercentile": "q has no default, so no positional prefix reaches out",
    "nanquantile": "q has no default, so no positional prefix reaches out",
    "ptp": "flopscope's wrapper takes out= as a keyword only (numpy allows both)",
}


def _reduction_ops() -> list[str]:
    return sorted(
        name
        for name, entry in REGISTRY.items()
        if entry.get("category") == "counted_reduction"
        and entry.get("module") == "numpy"
        and callable(_resolve(name))
    )


def _positional_out_prefix(name: str) -> list[Any] | None:
    """Defaults for the parameters between ``a`` and ``out``, or None."""
    fn = _resolve(name)
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    names = [p.name for p in params]
    if "out" not in names:
        return None
    index = names.index("out")
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if params[index].kind not in positional:
        return None
    prefix = params[1:index]
    if any(p.kind not in positional for p in prefix):
        return None
    if any(p.default is inspect.Parameter.empty for p in prefix):
        return None
    return [p.default for p in prefix]


_POSITIONAL_REDUCTIONS = [n for n in _reduction_ops() if n not in _NO_POSITIONAL_OUT]


@pytest.mark.parametrize("name", _POSITIONAL_REDUCTIONS)
def test_a_reduction_guards_both_its_out_channels(budget, name):
    """All four spellings of the same destination must bill the same number."""
    prefix = _positional_out_prefix(name)
    assert prefix is not None, (
        f"{name} has no positional out= slot — add it to _NO_POSITIONAL_OUT "
        f"with the reason"
    )
    fn = _resolve(name)
    a = _arr((6, 5))
    natural = _quietly(fn, a)
    shape = np.shape(natural)
    dtype = _widest_accepted_dest(lambda d: fn(a, out=d), shape, natural)

    costs = {}
    for label, make_out, call in (
        ("keyword", lambda d: d, lambda o: fn(a, out=o)),
        ("keyword-tuple", lambda d: (d,), lambda o: fn(a, out=o)),
        ("positional", lambda d: d, lambda o: fn(a, *prefix, o)),
        ("positional-tuple", lambda d: (d,), lambda o: fn(a, *prefix, o)),
    ):
        dest = _dest(shape, dtype)
        cost, result = _billed(budget, call, make_out(dest))
        assert result is dest, f"{name}: the {label} channel did not return dest"
        costs[label] = cost

    assert len(set(costs.values())) == 1, (
        f"{name}: the four out= spellings billed differently: {costs}"
    )


@pytest.mark.parametrize("name", _POSITIONAL_REDUCTIONS)
@pytest.mark.parametrize("channel", ["keyword", "positional"])
def test_a_reduction_refuses_a_list_for_free_on_both_channels(budget, name, channel):
    prefix = _positional_out_prefix(name)
    assert prefix is not None
    fn = _resolve(name)
    a = _arr((6, 5))
    natural = _quietly(fn, a)
    shape = np.shape(natural)
    dtype = _widest_accepted_dest(lambda d: fn(a, out=d), shape, natural)
    dest = _dest(shape, dtype)
    before = budget.flops_used

    with pytest.raises(TypeError, match="out= must be an array"):
        if channel == "keyword":
            fn(a, out=[dest])
        else:
            fn(a, *prefix, [dest])

    assert budget.flops_used == before, (
        f"{name} was billed for refusing a list on the {channel} channel"
    )
    assert not np.asarray(dest).any()


def test_the_no_positional_out_list_is_neither_stale_nor_a_silent_skip():
    reductions = set(_reduction_ops())
    stale = sorted(set(_NO_POSITIONAL_OUT) - reductions)
    assert not stale, f"_NO_POSITIONAL_OUT names non-reductions: {stale}"
    unexplained = sorted(n for n, why in _NO_POSITIONAL_OUT.items() if not why.strip())
    assert not unexplained, f"entries without a reason: {unexplained}"
    # Every excuse must still be true: either the signature has no reachable
    # positional slot, or the wrapper refuses one.
    for name in sorted(_NO_POSITIONAL_OUT):
        prefix = _positional_out_prefix(name)
        if prefix is None:
            continue
        # The signature says a positional destination is reachable, so the
        # excuse has to be the wrapper's: ptp advertises numpy's slot and
        # accepts only two positional arguments. If that stops being true,
        # the op belongs in the parametrized test instead of on this list.
        fn = _resolve(name)
        a = _arr((6, 5))
        natural = _quietly(fn, a)
        dest = _dest(np.shape(natural), str(np.asarray(natural).dtype))
        with pytest.raises(TypeError):
            _quietly(fn, a, *prefix, dest)


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


@pytest.mark.parametrize("name", _DRIVEN_NAMES)
def test_a_none_holding_tuple_is_treated_as_no_destination(budget, name):
    fn, args, kwargs, shape, _ = _driven_call(name)
    result = fn(*args, out=(None,), **kwargs)
    assert result is not None and np.shape(result) == shape
    assert np.asarray(result).dtype != np.dtype(object)


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


# ---------------------------------------------------------------------------
# The two destination channels that a derived tuple-parity sweep cannot see
# ---------------------------------------------------------------------------
#
# Both of these move the bare and the 1-tuple form by the SAME amount, so the
# derived parity test above stays green whichever way they go. They need their
# own assertions, or the fixes they pin can be reverted silently.


def test_ptp_folds_its_destination_dtype_like_its_reduction_siblings():
    # ptp takes out= through **kwargs, so it reached neither the normalization
    # nor the destination-dtype fold. A wider destination was free: reverting
    # the fold leaves this at a quarter price while every other test passes.
    rng = np.random.default_rng(4)
    with fl.BudgetContext(flop_budget=10**14, quiet=True) as ctx:
        a = fnp.asarray(rng.standard_normal((200, 200)).astype("float32"))
        wide = np.zeros(200, dtype="complex128")

        before = ctx.flops_used
        fnp.ptp(a, axis=-1)
        bare = ctx.flops_used - before

        before = ctx.flops_used
        fnp.ptp(a, axis=-1, out=wide)
        widened = ctx.flops_used - before

        # The sibling is the real oracle: ptp and amax share a registry
        # category, so whatever ratio a wide destination earns for one it must
        # earn for the other. Comparing ptp against ITSELF would pass on any
        # ratio including 1.0; comparing against amax is the outlier check
        # that found this defect in the first place, and it does not need
        # re-tuning when the dtype weights move.
        sibling_wide = np.zeros(200, dtype="complex128")
        before = ctx.flops_used
        fnp.amax(a, axis=-1)
        sibling_bare = ctx.flops_used - before

        before = ctx.flops_used
        fnp.amax(a, axis=-1, out=sibling_wide)
        sibling_widened = ctx.flops_used - before

    assert widened > bare, (
        f"ptp billed {widened} into a complex128 destination against {bare} "
        f"with none -- the destination's dtype is not reaching the rate"
    )
    assert widened / bare == sibling_widened / sibling_bare, (
        f"ptp widened by {widened / bare}x where its sibling amax widened by "
        f"{sibling_widened / sibling_bare}x for the same destination"
    )


def test_clip_prices_a_positional_destination_like_a_keyword_one():
    # clip declares out keyword-only but republishes numpy's signature, which
    # advertises it as the fourth positional. Without the positional channel
    # the destination lands in *args and is counted as an extra BOUND, so it
    # costs MORE than the keyword spelling and its dtype never reaches the rate.
    rng = np.random.default_rng(4)
    with fl.BudgetContext(flop_budget=10**14, quiet=True) as ctx:
        a = fnp.asarray(rng.standard_normal(1000).astype("float32"))
        kw_dest = np.zeros(1000, dtype="complex128")
        pos_dest = np.zeros(1000, dtype="complex128")

        before = ctx.flops_used
        fnp.clip(a, -1.0, 1.0, out=kw_dest)  # pyright: ignore[reportArgumentType]
        keyword_cost = ctx.flops_used - before

        before = ctx.flops_used
        fnp.clip(a, -1.0, 1.0, pos_dest)
        positional_cost = ctx.flops_used - before

    assert positional_cost == keyword_cost
    assert np.array_equal(np.asarray(kw_dest), np.asarray(pos_dest))


def test_clip_does_not_swallow_positionals_past_the_destination_slot():
    # numpy's clip has four positional slots and rejects a fifth. Consuming
    # "three or more" here would truncate the extras instead -- and because
    # `where` is keyword-only in numpy's clip, a caller passing it positionally
    # would get an UNMASKED clip back rather than numpy's TypeError.
    rng = np.random.default_rng(4)
    with fl.BudgetContext(flop_budget=10**12, quiet=True) as ctx:
        a = fnp.asarray(rng.standard_normal(100).astype("float64"))
        dest = np.zeros(100, dtype="float64")
        mask = np.zeros(100, dtype=bool)

        with pytest.raises(TypeError):
            fnp.clip(a, -1.0, 1.0, dest, mask)  # pyright: ignore[reportArgumentType]
