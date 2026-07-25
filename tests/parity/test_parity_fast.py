"""Tier 1: blocks every pull request. Target: under 60 seconds.

This is the fix for the gap this whole project exists to close: a parity
signal already existed in this repository (``make test-client-parity-measure``)
and never blocked anything, because its recipe lines are prefixed with ``-``
and its CI step is labelled non-blocking. Nothing here may use that pattern —
every test in this module is a real gate.
"""

from __future__ import annotations

import pytest

from tests.parity.allowlist import ENTRIES, apply, validate_entries
from tests.parity.compare import DIMENSIONS
from tests.parity.corpus import all_cases, fast_cases, registry_grid
from tests.parity.report import render
from tests.parity.runner import run_corpus


def test_allowlist_schema_is_valid():
    assert validate_entries(ENTRIES) == []


def test_every_dimension_is_spelled_consistently():
    for entry in ENTRIES:
        assert entry.dimension in DIMENSIONS


def test_corpus_has_not_silently_shrunk():
    # Guards against the corpus quietly losing ops, which would look like
    # green because there is simply nothing left to diverge on.
    assert len(registry_grid.op_names()) > 500
    assert len(all_cases()) > len(fast_cases())


@pytest.mark.parity_server
def test_fast_tier_has_no_unexplained_divergences():
    result = run_corpus(fast_cases())
    assert result.infrastructure_failure is None, result.infrastructure_failure
    allow = apply(result.divergences, ENTRIES)
    assert not allow.unexplained, render(result, allow, coverage={})
