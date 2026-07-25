"""Worker unit tests that need no backend: namespace construction and dispatch."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from tests.parity.case import Case
from tests.parity.observe import fingerprint
from tests.parity.worker import FIXTURE_SOURCE, build_namespace, run_case, run_stream


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

    def summary(self):
        # No-op stand-in for the real refresh call `run_case` now makes before
        # every `flops_used` read (needed on the client backend; harmless
        # here since the fake's counter doesn't need refreshing).
        return ""


class _FakeCachedCtx:
    """Stands in for a BudgetContext whose `flops_used` is a CACHE that only
    refreshes on an explicit `summary()` call — this is how the real client
    backend behaves (`flopscope-client/src/flopscope/_budget.py`: the local
    `_flops_used` cache is only updated from `__enter__`, `__exit__`, or a
    `summary()` call, never automatically after an individual op dispatch).

    Regression test double for the bug where `run_case` read `ctx.flops_used`
    immediately after `eval()` without refreshing first: since the worker
    holds one ambient context open for the whole corpus, that read always
    landed on a stale cache, so the client backend reported a `flops` delta
    of 0 for every single case, no matter what actually ran.
    """

    def __init__(self, true_values):
        self._true_values = list(true_values)
        self._true_i = 0
        self._cached = 0

    def summary(self):
        self._cached = self._true_values[min(self._true_i, len(self._true_values) - 1)]
        self._true_i += 1
        return ""

    @property
    def flops_used(self):
        return self._cached


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


def test_run_case_refreshes_the_flops_cache_before_reading_the_delta():
    # Regression test for a per-case FLOP delta that was always 0 on the
    # client backend: `run_case` must call `ctx.summary()` (or an equivalent
    # refresh) before every `ctx.flops_used` read, not just read the
    # property directly, or a cached-until-refreshed context reports every
    # case as costing 0 FLOPs regardless of what actually ran.
    obs = run_case(_ns(), Case(id="t/ok", source="V[0]"), _FakeCachedCtx([0, 5]))
    assert obs["flops"] == 5


def test_run_case_refreshes_the_flops_cache_on_the_exception_path_too():
    obs = run_case(_ns(), Case(id="t/boom", source="V[99]"), _FakeCachedCtx([0, 3]))
    assert obs["outcome"] == "raised"
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


class _FakeRandom:
    """Stands in for `flopscope.numpy.random`: only `seed()` is needed, since
    `FIXTURE_SOURCE` calls it unconditionally as its first statement."""

    @staticmethod
    def seed(value=None):
        pass


class _FakeFnp:
    """Stands in for `flopscope.numpy`: `array()` returns a fresh, plain
    mutable `list` on every call (scalars pass through as-is, since a bare
    float has nothing mutable to protect), so a case that mutates a fixture
    in place can only ever corrupt the namespace it was handed, never a
    later case's."""

    random = _FakeRandom()

    @staticmethod
    def array(values, dtype=None):
        if isinstance(values, (list, tuple)):
            return list(values)
        return values


def test_run_stream_rebuilds_fixtures_so_one_case_cannot_contaminate_the_next():
    # This is the regression test for the exact bug that discarded a prior
    # measurement pass: an in-place mutation in one case leaking into the
    # next via a fixture shared across cases. It only passes if `run_stream`
    # calls `build_namespace(fnp)` fresh inside the loop, per case.
    mutate = Case(id="t/mutate", source="V[0]", setup="V[0] = 999.0")
    read = Case(id="t/read", source="V[0]")
    stdin = io.StringIO(
        json.dumps(mutate.to_json()) + "\n" + json.dumps(read.to_json()) + "\n"
    )
    stdout = io.StringIO()

    with patch("tests.parity.worker.build_namespace", wraps=build_namespace) as spy:
        run_stream(_FakeFnp(), _FakeCtx([0, 0]), stdin, stdout)

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(lines) == 2
    assert lines[0]["id"] == "t/mutate"
    assert lines[0]["value"] == fingerprint(999.0)
    # The second case must see a pristine V, not the first case's mutation.
    assert lines[1]["id"] == "t/read"
    assert lines[1]["value"] == fingerprint(3.0)
    # One fresh namespace per case, not one shared namespace for the run.
    assert spy.call_count == 2


#: A value whose `.shape` attribute RAISES when merely accessed (not just
#: reports a wrong type) defeats `observe_result`'s own type-based defenses,
#: exercising the backstop in `run_case` itself. This is the real-world
#: shape: a registry operation returning a class rather than an instance
#: leaves `.shape` as a descriptor that misbehaves when read.
_HOSTILE_SETUP = """
class Hostile:
    @property
    def shape(self):
        raise RuntimeError("shape blew up")
"""


def test_run_case_records_a_result_that_fails_to_describe_instead_of_raising():
    case = Case(id="t/hostile", source="Hostile()", setup=_HOSTILE_SETUP)
    obs = run_case(_ns(), case, _FakeCtx([0, 0]))
    assert obs["id"] == "t/hostile"
    assert obs["outcome"] == "record_failed"
    assert obs["exc_type"] == "RuntimeError"


def test_run_stream_processes_the_next_case_after_a_record_failure():
    # The important half of this regression test: a case that defeats
    # `observe_result` must not cost the worker every case still queued
    # behind it on stdin. This is the exact failure mode a real corpus run
    # hit: an unhandled exception from recording (not from the expression
    # itself) propagated out of `run_case`, out of `run_stream`, and killed
    # the worker process mid-run.
    bad = Case(id="t/hostile", source="Hostile()", setup=_HOSTILE_SETUP)
    good = Case(id="t/after", source="1 + 1")
    stdin = io.StringIO(
        json.dumps(bad.to_json()) + "\n" + json.dumps(good.to_json()) + "\n"
    )
    stdout = io.StringIO()

    run_stream(_FakeFnp(), _FakeCtx([0, 0]), stdin, stdout)

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(lines) == 2
    assert lines[0]["id"] == "t/hostile"
    assert lines[0]["outcome"] == "record_failed"
    # The case queued behind the bad one still ran and was recorded: the
    # worker did not die and did not skip it.
    assert lines[1]["id"] == "t/after"
    assert lines[1]["outcome"] == "returned"


def test_run_stream_skips_malformed_lines_without_losing_queued_cases(capsys):
    good = Case(id="t/ok2", source="1 + 1")
    stdin = io.StringIO(
        "not json at all\n"
        + json.dumps({"source": "1"})  # well-formed JSON, missing required "id"
        + "\n"
        + json.dumps(good.to_json())
        + "\n"
    )
    stdout = io.StringIO()

    run_stream(_FakeFnp(), _FakeCtx([0, 0]), stdin, stdout)

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(lines) == 1
    assert lines[0]["id"] == "t/ok2"
    assert lines[0]["outcome"] == "returned"
    # Diagnostics go to stderr; stdout carries only clean observation JSON
    # (already implied above by every stdout line parsing as JSON).
    captured = capsys.readouterr()
    assert "skipping malformed case line" in captured.err
