"""Tier 2: the entire corpus. Release-blocking, not on the pull-request path.

At ~8,900 cases this takes on the order of five minutes, which is too slow to
block a pull request but appropriate as a release gate. See
``tests/parity/test_parity_fast.py`` for the tier that blocks every PR.
"""

from __future__ import annotations

import pytest

from tests.parity.allowlist import ENTRIES, apply
from tests.parity.corpus import all_cases, registry_grid
from tests.parity.report import render
from tests.parity.runner import run_corpus

#: The full corpus's flaky count (cases the twice-run self-check found
#: nondeterministic and therefore quarantined from comparison - see
#: ``run_corpus``) measured stable at 50 across three consecutive full-
#: corpus runs. The ceiling below leaves headroom above that baseline so
#: ordinary run-to-run variation passes, while a systemic regression (e.g.
#: nondeterminism spreading to more ops, or the server hiccupping between a
#: backend's two passes) still fails. Retune this deliberately, from a fresh
#: measurement, rather than by guesswork, if it ever starts failing on
#: otherwise-healthy runs.
_MEASURED_FLAKY_BASELINE = 50
_FLAKY_CEILING = 60


@pytest.mark.parity_server
def test_full_corpus_has_no_unexplained_divergences_and_no_stale_entries():
    cases = all_cases()
    result = run_corpus(cases)
    assert result.infrastructure_failure is None, result.infrastructure_failure
    allow = apply(result.divergences, ENTRIES)
    coverage = {
        "cases": len(cases),
        "ops_enumerated": len(registry_grid.op_names()),
        "ops_undriven": len(registry_grid.undriven()),
        "flaky": len(result.flaky),
    }
    report = render(result, allow, coverage)
    print(report)
    assert len(result.flaky) <= _FLAKY_CEILING, (
        f"{len(result.flaky)} flaky cases exceeds the ceiling of "
        f"{_FLAKY_CEILING} (measured baseline {_MEASURED_FLAKY_BASELINE}); "
        f"an unbounded flaky count silently shrinks what actually gets "
        f"compared. {report}"
    )
    assert not allow.unexplained, report
    assert not allow.stale, report
