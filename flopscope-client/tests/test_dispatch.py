"""Unit tests for the dispatch-timing accumulator (deterministic fake clock)."""

from __future__ import annotations

import flopscope._dispatch as d
import pytest


def _fake_clock(values):
    it = iter(values)
    return lambda: next(it)


def test_single_span_adds_its_wall(monkeypatch):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([100, 350]))  # t0=100, t1=350
    d.reset_dispatch()
    with d.dispatch_span():
        pass
    assert d.total_dispatch_ns() == 250


def test_nested_spans_count_wall_once(monkeypatch):
    # outer t0=0 ; inner t0=100,t1=400 (=300) ; outer t1=500 (=500 wall)
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 100, 400, 500]))
    d.reset_dispatch()
    with d.dispatch_span():  # outer reads now()->0
        with d.dispatch_span():  # inner reads now()->100, exit now()->400
            pass
        # outer exit reads now()->500
    # inner added 300; outer adds max(0, 500 - 300) = 200; total = 500 (counted once)
    assert d.total_dispatch_ns() == 500


def test_timed_dispatch_decorator(monkeypatch):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([10, 60]))
    d.reset_dispatch()

    @d.timed_dispatch
    def op():
        return 42

    assert op() == 42
    assert d.total_dispatch_ns() == 50


def test_delta_helpers(monkeypatch):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 40]))
    d.reset_dispatch()
    base = d.total_dispatch_ns()
    with d.dispatch_span():
        pass
    assert d.total_dispatch_ns() - base == 40


def test_accumulates_even_on_exception(monkeypatch):
    monkeypatch.setattr(d, "_now_ns", _fake_clock([0, 70]))
    d.reset_dispatch()
    with pytest.raises(ValueError):
        with d.dispatch_span():
            raise ValueError("boom")
    assert d.total_dispatch_ns() == 70
