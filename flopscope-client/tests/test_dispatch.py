"""Unit tests for the dispatch-timing accumulator (deterministic fake clock).

A span only contributes when opened by flopscope's own code, and a test module is
by definition not that. The ``_internal_caller`` fixture below states "treat this
call as internal" explicitly, so these tests cover the baseline/delta arithmetic
without silently depending on where the test file lives. Which callers actually
count is covered separately in
``tests/client_compat/unit/test_dispatch_provenance.py``.
"""

from __future__ import annotations

import flopscope._dispatch as d
import pytest


def _fake_clock(values):
    it = iter(values)
    return lambda: next(it)


@pytest.fixture
def _internal_caller(monkeypatch):
    """Exercise the counting path from a test module.

    The arithmetic under test is independent of provenance; this fixture makes
    that explicit rather than leaving the tests sensitive to their own location.
    """
    monkeypatch.setattr(d, "_caller_is_internal", lambda depth: True)


def test_single_span_adds_its_wall(monkeypatch, _internal_caller):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([100, 350]))  # t0=100, t1=350
    d.reset_dispatch()
    with d._counted_span():
        pass
    assert d.total_dispatch_ns() == 250


def test_nested_spans_count_wall_once(monkeypatch, _internal_caller):
    # outer t0=0 ; inner t0=100,t1=400 (=300) ; outer t1=500 (=500 wall)
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 100, 400, 500]))
    d.reset_dispatch()
    with d._counted_span():  # outer reads now()->0
        with d._counted_span():  # inner reads now()->100, exit now()->400
            pass
        # outer exit reads now()->500
    # inner added 300; outer adds max(0, 500 - 300) = 200; total = 500 (counted once)
    assert d.total_dispatch_ns() == 500


def test_delta_helpers(monkeypatch, _internal_caller):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 40]))
    d.reset_dispatch()
    base = d.total_dispatch_ns()
    with d._counted_span():
        pass
    assert d.total_dispatch_ns() - base == 40


def test_accumulates_even_on_exception(monkeypatch, _internal_caller):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 70]))
    d.reset_dispatch()
    with pytest.raises(ValueError):
        with d._counted_span():
            raise ValueError("boom")
    assert d.total_dispatch_ns() == 70


def test_timed_dispatch_is_transparent_for_outside_callables():
    """Decorating a callable from outside the package leaves it untouched.

    Same return value, same argument handling, nothing accumulated — the wrapper
    is a no-op rather than an error, so no caller can be broken by it.
    """
    d.reset_dispatch()

    @d.timed_dispatch
    def op(a, b=2):
        return a * b

    assert op(21) == 42
    assert op(3, b=4) == 12
    assert d.total_dispatch_ns() == 0


def test_timed_dispatch_wraps_a_package_callable(monkeypatch):
    """A real flopscope callable is wrapped and its wall accumulated."""
    from flopscope import flops

    monkeypatch.setattr(d, "_now_ns", _fake_clock([10, 60]))
    d.reset_dispatch()
    wrapped = d.timed_dispatch(flops.einsum_cost)
    assert wrapped is not flops.einsum_cost, "package callable should be wrapped"
    assert wrapped.__name__ == flops.einsum_cost.__name__
