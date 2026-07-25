"""End-to-end: both backends, a real server, a known divergence.

This is the CANARY. `idiom/complex-scalar-mul` is the 2026-07-25 seed bug: a
Python complex operand cannot cross the wire. If this test stops reporting a
divergence, either the bug was fixed (delete the canary and its allowlist entry)
or the harness plumbing silently broke (fix the harness).
"""

from __future__ import annotations

import pytest

from tests.parity.case import Case
from tests.parity.runner import run_corpus

pytestmark = pytest.mark.parity_server


def test_identical_expressions_agree_on_both_backends():
    result = run_corpus([Case(id="t/sum", source="fnp.sum(V)")])
    assert result.infrastructure_failure is None
    assert result.divergences == []
    assert result.flaky == []


def test_canary_complex_scalar_diverges():
    result = run_corpus(
        [Case(id="idiom/complex-scalar-mul", source="fnp.astype(V, 'complex64') * 1j")]
    )
    assert result.infrastructure_failure is None
    dims = {d.dimension for d in result.divergences}
    assert "outcome" in dims, (
        "the seed bug no longer diverges: either it was fixed (delete this "
        "canary) or the harness plumbing is broken (fix the harness)"
    )


def test_tuple_axis_diverges():
    result = run_corpus([Case(id="t/tuple-axis", source="fnp.sum(A, axis=(0, 1))")])
    assert "outcome" in {d.dimension for d in result.divergences}


def test_observations_are_recorded_for_both_backends():
    result = run_corpus([Case(id="t/sum", source="fnp.sum(V)")])
    assert set(result.observations) == {"inproc", "client"}
    assert "t/sum" in result.observations["inproc"]
    assert "t/sum" in result.observations["client"]
