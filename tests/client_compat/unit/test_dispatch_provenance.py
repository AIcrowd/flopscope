"""Only flopscope's own code contributes to the dispatch accumulator.

The accumulator measures flopscope's own dispatch cost and is subtracted from
wall time to separate framework cost from caller cost. A span opened around
non-flopscope code would therefore corrupt that split, so such spans run
normally but contribute nothing and are counted instead.

Module-level state stays reachable through ordinary imports, so the invariant is
enforced where the span is opened rather than by restricting the API surface.
The tests below cover both halves: every internal call site must still
accumulate, and each way of reaching the entry points from outside the package
must be inert.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest

from flopscope import _dispatch as D  # parent conftest puts the CLIENT on sys.path


def _burn(seconds: float = 0.02) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@pytest.fixture(autouse=True)
def _clean():
    D.reset_dispatch()
    yield
    D.reset_dispatch()


# --------------------------------------------------------------------------
# Internal use must keep accumulating — this is the regression that matters.
# --------------------------------------------------------------------------


def test_internal_module_function_accumulates():
    """The @timed_dispatch shape used in _io.py, flops.py and __init__.py."""
    import flopscope

    ns = flopscope.__dict__  # defined *inside* a flopscope module namespace
    exec(
        compile(
            "def _probe():\n    import time; time.sleep(0.02)\n", "<probe>", "exec"
        ),
        ns,
    )
    try:
        wrapped = D.timed_dispatch(ns["_probe"])
        before = D.total_dispatch_ns()
        wrapped()
        assert D.total_dispatch_ns() > before, "internal callable did not accumulate"
        assert D.participant_span_count() == 0
    finally:
        ns.pop("_probe", None)


def test_every_shipped_internal_call_site_is_recognised_internal():
    """Guard the real decorated callables, not a synthetic stand-in.

    Covers the three shapes actually shipped: module-level functions, methods on
    a class, and the dynamically-built proxy closures returned by
    ``__init__``/``linalg``/``random``/``stats``.
    """
    from flopscope import _io, _remote_array, flops

    candidates = [
        _io._ingest,  # module-level @timed_dispatch
        flops.einsum_cost,
        flops.svd_cost,
        _remote_array.RemoteArray._fetch_data,  # @timed_dispatch on a method
    ]
    for mod_name in ("flopscope.linalg", "flopscope.random", "flopscope.stats"):
        mod = importlib.import_module(mod_name)
        # Proxies are built lazily by __getattr__; pull whichever exists.
        for attr in ("norm", "inv", "normal", "uniform", "zscore", "mean"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                candidates.append(fn)
                break

    for fn in candidates:
        assert D._callable_is_internal(fn), f"{fn!r} was misclassified as external"


def test_reimported_submodule_is_still_recognised_internal():
    """Eviction + re-import replaces a module's dict without changing how many
    modules exist, so provenance must be resolved live rather than from a cache
    keyed on a cheap sentinel — otherwise that module's spans silently stop
    counting after any reload."""
    import flopscope.linalg as la

    before_len = len(sys.modules)
    assert D._callable_is_internal(la.norm)

    del sys.modules["flopscope.linalg"]
    reloaded = importlib.import_module("flopscope.linalg")

    assert len(sys.modules) == before_len, "precondition: module count is unchanged"
    assert D._callable_is_internal(reloaded.norm), (
        "a re-imported flopscope submodule was misclassified as external"
    )


def test_internal_dispatch_span_accumulates():
    """The `with dispatch_span():` shape used in _budget/_connection/_remote_array."""
    from flopscope import _connection

    ns = _connection.__dict__
    src = "def _probe(span):\n    with span():\n        import time; time.sleep(0.02)\n"
    exec(compile(src, "<probe>", "exec"), ns)
    try:
        before = D.total_dispatch_ns()
        ns["_probe"](D.dispatch_span)
        assert D.total_dispatch_ns() > before
        assert D.participant_span_count() == 0
    finally:
        ns.pop("_probe", None)


# --------------------------------------------------------------------------
# External use must be inert, however the entry point is reached.
# --------------------------------------------------------------------------


def _external_fn():
    _burn()


def test_inert_via_package_attribute():
    import flopscope

    before = D.total_dispatch_ns()
    flopscope.timed_dispatch(_external_fn)()
    assert D.total_dispatch_ns() == before, "external wall was absorbed into dispatch"
    assert D.participant_span_count() == 1


def test_inert_via_direct_submodule_import():
    import flopscope._dispatch as direct

    before = direct.total_dispatch_ns()
    direct.timed_dispatch(_external_fn)()
    assert direct.total_dispatch_ns() == before
    assert direct.participant_span_count() == 1


def test_inert_via_importlib_and_sys_modules():
    """The module object is the same however it is obtained."""
    for mod in (
        importlib.import_module("flopscope._dispatch"),
        sys.modules["flopscope._dispatch"],
    ):
        D.reset_dispatch()
        before = mod.total_dispatch_ns()
        mod.timed_dispatch(_external_fn)()
        assert mod.total_dispatch_ns() == before
        assert mod.participant_span_count() == 1


def test_inert_via_the_raw_counted_span_primitive():
    """The primitive that mutates the accumulator carries the check itself.

    Guarding only the public wrappers would put the check somewhere other than
    the thing being protected: a span opened here from outside would both count
    and leave the span tally at zero, so the tally could not be relied on.
    """
    before = D.total_dispatch_ns()
    with D._counted_span():
        _burn()
    assert D.total_dispatch_ns() == before
    assert D.participant_span_count() == 1


def test_inert_via_reexported_dispatch_span():
    """dispatch_span is re-exported from _budget and _connection, which are on
    the hot path, so the check has to live in the callee."""
    from flopscope import _budget, _connection

    for mod in (_budget, _connection):
        D.reset_dispatch()
        before = D.total_dispatch_ns()
        with mod.dispatch_span():
            _burn()
        assert D.total_dispatch_ns() == before
        assert D.participant_span_count() == 1


def test_inert_for_a_bound_method():
    """Bound methods unwrap to their underlying function before the check."""

    class Caller:
        def work(self):
            _burn()

    before = D.total_dispatch_ns()
    D.timed_dispatch(Caller().work)()
    assert D.total_dispatch_ns() == before
    assert D.participant_span_count() == 1


def test_unwrapping_is_by_type_not_by_attribute_name():
    """A `.func` attribute is ordinary metadata on wrapper and decorator objects.

    Following it by name would hand the decision to the object being judged: an
    external callable could point it at a flopscope function and be treated as
    internal. Unwrapping is therefore restricted to real bound methods and
    functools.partial.
    """
    from flopscope import flops

    class Wrapper:
        func = flops.einsum_cost  # metadata pointing at a flopscope function

        def __call__(self):
            _burn()

    obj = Wrapper()
    assert not D._callable_is_internal(obj)
    before = D.total_dispatch_ns()
    D.timed_dispatch(obj)()
    assert D.total_dispatch_ns() == before
    assert D.participant_span_count() == 1


def test_real_methods_and_partials_still_unwrap():
    """The shapes flopscope actually decorates must keep working."""
    import functools

    from flopscope import _remote_array, flops

    assert D._callable_is_internal(_remote_array.RemoteArray._fetch_data)
    assert D._callable_is_internal(functools.partial(flops.einsum_cost, "ij,jk->ik"))

    class Ext:
        def method(self):
            pass

    assert not D._callable_is_internal(Ext().method)


def test_external_callable_still_runs_and_returns():
    """The wrapper must be a no-op, not an error: same behaviour, same result."""
    calls = []

    def body(a, b, *, c):
        calls.append((a, b, c))
        return a + b + c

    assert D.timed_dispatch(body)(1, 2, c=3) == 6
    assert calls == [(1, 2, 3)]


def test_module_and_filename_do_not_determine_provenance():
    """``__module__`` and ``co_filename`` are ordinary writable attributes, which
    is why provenance is decided by ``__globals__`` identity."""

    def impostor():
        _burn()

    impostor.__module__ = "flopscope._dispatch"
    impostor.__qualname__ = "flopscope._dispatch.impostor"
    assert not D._callable_is_internal(impostor)
    before = D.total_dispatch_ns()
    D.timed_dispatch(impostor)()
    assert D.total_dispatch_ns() == before
    assert D.participant_span_count() == 1


def test_external_spans_are_counted():
    """The counter lets an embedding harness assert the invariant held."""
    assert D.participant_span_count() == 0
    D.timed_dispatch(_external_fn)()
    D.timed_dispatch(_external_fn)()
    assert D.participant_span_count() == 2
