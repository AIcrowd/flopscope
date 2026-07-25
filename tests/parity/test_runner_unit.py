"""Fast unit coverage for `run_corpus`'s failure-handling edges.

`test_runner_integration.py` proves the happy path end to end against real
backends and a real server, but that suite is slow to run repeatedly while
iterating on failure handling. These tests isolate specific properties with
fakes instead: a worker that dies mid-corpus does not lose the rest of the
run, a majority-broken backend is reported as infrastructure rather than
thousands of individual divergences (while a majority *domain* exception,
like probing outside a budget context, is not), worker stderr reaches the
result instead of being dropped, and the server is stopped even when a
backend run raises partway through — none of which need a live server or a
real worker subprocess to verify.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.parity import runner
from tests.parity.case import Case
from tests.parity.observe import observe_worker_died
from tests.parity.runner import _looks_like_infrastructure, _run_backend, run_corpus

_ROOT = runner._ROOT


def test_a_worker_that_dies_mid_corpus_backfills_the_rest_as_worker_died():
    # `setup` runs as an exec'd statement (unlike `source`, which is only an
    # expression), so it is the vehicle for actually killing the worker
    # process, not just raising inside it.
    cases = [
        Case(id="t/kill", source="1", setup="import os; os._exit(1)"),
        Case(id="t/never-reached", source="1"),
    ]
    observations, _stderr = _run_backend("inproc", cases)
    assert set(observations) == {"t/kill", "t/never-reached"}
    assert observations["t/kill"] == observe_worker_died()
    assert observations["t/never-reached"] == observe_worker_died()


def test_a_normal_run_is_unaffected_by_the_worker_died_fallback():
    observations, _stderr = _run_backend(
        "inproc", [Case(id="t/sum", source="fnp.sum(V)")]
    )
    assert observations["t/sum"]["outcome"] == "returned"


def test_run_backend_surfaces_worker_stderr_on_failure():
    # An invalid --backend value never reaches any case-execution code: it is
    # rejected by argparse before the worker touches stdin, so this fails
    # deterministically and fast, without needing a live server or a crafted
    # Case to provoke a real crash.
    cases = [Case(id="t/sum", source="fnp.sum(V)")]
    observations, stderr_text = _run_backend("not-a-real-backend", cases)
    assert observations == {"t/sum": observe_worker_died()}
    assert stderr_text != ""
    assert "not-a-real-backend" in stderr_text


def test_run_backend_handles_a_subprocess_timeout(monkeypatch):
    # `_run_backend`'s TimeoutExpired handler is exercised without a
    # genuinely hanging worker: `subprocess.run` is monkeypatched to raise it
    # directly, carrying partial stdout for one case as it would if the
    # worker had produced output before hanging.
    completed_line = json.dumps({"id": "t/first", "outcome": "returned", "flops": 0})

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["worker"],
            timeout=1,
            output=completed_line + "\n",
            stderr="partial stderr",
        )

    monkeypatch.setattr(runner.subprocess, "run", _raise_timeout)

    cases = [Case(id="t/first", source="1"), Case(id="t/second", source="1")]
    observations, stderr_text = _run_backend("inproc", cases)

    assert observations["t/first"]["outcome"] == "returned"
    assert observations["t/second"] == observe_worker_died()
    assert stderr_text == "partial stderr"


def test_looks_like_infrastructure_on_empty_observations():
    assert _looks_like_infrastructure({}) is True


def test_looks_like_infrastructure_when_most_cases_died():
    observations = {
        "a": observe_worker_died(),
        "b": observe_worker_died(),
        "c": {"outcome": "returned"},
    }
    assert _looks_like_infrastructure(observations) is True


def test_looks_like_infrastructure_when_most_cases_died_even_with_mixed_exc_types():
    # A dead worker produces no exception at all, so worker_died must
    # dominate regardless of what the surviving cases' exception types are.
    observations = {
        "a": observe_worker_died(),
        "b": observe_worker_died(),
        "c": {"outcome": "raised", "exc_type": "ValueError"},
    }
    assert _looks_like_infrastructure(observations) is True


def test_does_not_look_like_infrastructure_when_most_cases_are_fine():
    observations = {
        "a": {"outcome": "returned"},
        "b": {"outcome": "returned"},
        "c": observe_worker_died(),
    }
    assert _looks_like_infrastructure(observations) is False


def test_does_not_look_like_infrastructure_when_majority_raise_a_domain_exception():
    # NoBudgetContextError is a genuine domain exception raised by real code
    # paths on both backends, not a transport failure; a corpus family that
    # deliberately probes operating outside a budget context could make a
    # majority of its cases raise it legitimately.
    observations = {
        "a": {"outcome": "raised", "exc_type": "NoBudgetContextError"},
        "b": {"outcome": "raised", "exc_type": "NoBudgetContextError"},
        "c": {"outcome": "returned"},
    }
    assert _looks_like_infrastructure(observations) is False


def test_looks_like_infrastructure_when_majority_share_one_transport_exception_type():
    observations = {
        "a": {"outcome": "raised", "exc_type": "ConnectionError"},
        "b": {"outcome": "raised", "exc_type": "ConnectionError"},
        "c": {"outcome": "returned"},
    }
    assert _looks_like_infrastructure(observations) is True


def test_does_not_look_like_infrastructure_when_exception_types_are_a_mix():
    # Half the cases raise ConnectionError and the other cases raise
    # different, unrelated exceptions: no single exception type reaches a
    # majority, so this is real per-case signal, not a broken backend.
    observations = {
        "a": {"outcome": "raised", "exc_type": "ConnectionError"},
        "b": {"outcome": "raised", "exc_type": "ZMQError"},
        "c": {"outcome": "raised", "exc_type": "ValueError"},
        "d": {"outcome": "returned"},
    }
    assert _looks_like_infrastructure(observations) is False


def test_run_corpus_stops_the_server_even_when_a_backend_run_raises(monkeypatch):
    stopped = []
    sentinel = object()

    monkeypatch.setattr(runner, "start_server", lambda: sentinel)
    monkeypatch.setattr(runner, "stop_server", lambda proc: stopped.append(proc))

    def _boom(backend, cases):
        raise RuntimeError("simulated worker-launch failure")

    monkeypatch.setattr(runner, "_run_backend", _boom)

    with pytest.raises(RuntimeError, match="simulated worker-launch failure"):
        run_corpus([Case(id="t/sum", source="fnp.sum(V)")])

    assert stopped == [sentinel]


def test_run_corpus_stops_the_server_on_the_infrastructure_early_return(monkeypatch):
    stopped = []
    sentinel = object()

    monkeypatch.setattr(runner, "start_server", lambda: sentinel)
    monkeypatch.setattr(runner, "stop_server", lambda proc: stopped.append(proc))
    monkeypatch.setattr(
        runner,
        "_run_backend",
        lambda backend, cases: ({"t/sum": observe_worker_died()}, ""),
    )

    result = run_corpus([Case(id="t/sum", source="fnp.sum(V)")])

    assert result.infrastructure_failure is not None
    assert result.divergences == []
    assert stopped == [sentinel]


def test_run_corpus_surfaces_failing_backend_stderr_in_the_result(monkeypatch):
    monkeypatch.setattr(runner, "start_server", lambda: object())
    monkeypatch.setattr(runner, "stop_server", lambda proc: None)
    monkeypatch.setattr(
        runner,
        "_run_backend",
        lambda backend, cases: (
            {"t/sum": observe_worker_died()},
            f"{backend}: Traceback (most recent call last): boom",
        ),
    )

    result = run_corpus([Case(id="t/sum", source="fnp.sum(V)")])

    assert result.infrastructure_failure is not None
    assert "boom" in result.infrastructure_failure
    assert result.stderr["inproc"] == "inproc: Traceback (most recent call last): boom"


def test_run_corpus_on_an_empty_case_list_returns_a_clean_empty_result():
    # A later stage that filters cases by tag can legitimately produce an
    # empty list; this must not be reported as an infrastructure failure
    # (`_looks_like_infrastructure({})` is `True` on its own, per the test
    # above, but `run_corpus` short-circuits before ever calling it here).
    # No server or backend fixtures are needed for this to be correct.
    result = run_corpus([])
    assert result.infrastructure_failure is None
    assert result.divergences == []
    assert result.flaky == []
    assert result.observations == {}


def test_worker_module_is_invocable_as_a_subprocess_with_no_cases():
    # Sanity check on the exact invocation `_run_backend` uses: confirms
    # `tests.parity.worker` imports cleanly as a namespace package (no
    # `tests/__init__.py` needed) given cwd and PYTHONPATH set to the repo
    # root, independent of any parity-runner logic.
    proc = subprocess.run(
        [sys.executable, "-m", "tests.parity.worker", "--backend=inproc"],
        input="",
        capture_output=True,
        text=True,
        cwd=_ROOT,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
