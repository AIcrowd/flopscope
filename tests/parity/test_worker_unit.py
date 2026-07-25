"""Worker unit tests that need no backend: namespace construction and dispatch."""

from __future__ import annotations

from tests.parity.case import Case
from tests.parity.worker import FIXTURE_SOURCE, run_case


class _FakeCtx:
    """Stands in for a BudgetContext; returns a rising FLOP counter."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    @property
    def flops_used(self):
        value = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        return value


def _ns():
    # A tiny stand-in backend: enough for the worker's plumbing.
    return {"fnp": None, "V": [1.0, 2.0, 3.0]}


def test_runs_an_expression_and_records_the_result():
    obs = run_case(_ns(), Case(id="t/ok", source="V[0]"), _FakeCtx([0, 5]))
    assert obs["id"] == "t/ok"
    assert obs["outcome"] == "returned"
    assert obs["flops"] == 5


def test_records_an_exception_instead_of_propagating():
    obs = run_case(_ns(), Case(id="t/boom", source="V[99]"), _FakeCtx([0, 3]))
    assert obs["outcome"] == "raised"
    assert obs["exc_type"] == "IndexError"
    assert obs["flops"] == 3


def test_setup_statements_run_before_the_expression():
    case = Case(id="t/setup", source="k", setup="k = V[1]")
    obs = run_case(_ns(), case, _FakeCtx([0, 0]))
    assert obs["outcome"] == "returned"


def test_a_syntactically_invalid_case_is_recorded_not_crashed():
    obs = run_case(_ns(), Case(id="t/bad", source="V["), _FakeCtx([0, 0]))
    assert obs["outcome"] == "raised"
    assert obs["exc_type"] == "SyntaxError"


def test_fixture_source_builds_every_documented_fixture():
    for name in ("A", "B", "V", "I", "M", "E", "S"):
        assert f"{name} = " in FIXTURE_SOURCE
