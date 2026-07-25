from __future__ import annotations

from tests.parity.allowlist import apply
from tests.parity.compare import Divergence
from tests.parity.report import render
from tests.parity.runner import RunResult


def test_report_lists_unexplained_divergences_with_both_sides():
    div = Divergence("idiom/x", "dtype", "float32", "float64")
    text = render(RunResult(divergences=[div]), apply([div]), coverage={})
    assert "idiom/x" in text
    assert "dtype" in text
    assert "float32" in text
    assert "float64" in text


def test_report_states_coverage_counts():
    text = render(
        RunResult(),
        apply([]),
        coverage={"ops_enumerated": 617, "ops_driven": 470, "ops_undriven": 147},
    )
    assert "617" in text
    assert "470" in text
    assert "147" in text


def test_report_flags_stale_entries_explicitly():
    from tests.parity.allowlist import Category, Entry

    entry = Entry("a/b", "value", Category.PROXY_INHERENT, "reason")
    text = render(RunResult(), apply([], entries=(entry,)), coverage={})
    assert "stale" in text.lower()
    assert "a/b" in text


def test_report_names_flaky_cases_separately_from_divergences():
    text = render(RunResult(flaky=["grid/random::vector"]), apply([]), coverage={})
    assert "flaky" in text.lower()
    assert "grid/random::vector" in text
