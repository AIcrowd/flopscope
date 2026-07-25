"""Fast unit coverage for `run_corpus`'s failure-handling edges.

`test_runner_integration.py` proves the happy path end to end against real
backends and a real server, but that suite is slow to run repeatedly while
iterating on failure handling. These tests isolate three specific properties
with fakes instead: a worker that dies mid-corpus does not lose the rest of
the run, a majority-broken backend is reported as infrastructure rather than
thousands of individual divergences, and the server is stopped even when a
backend run raises partway through — none of which need a live server or a
real worker subprocess to verify.
"""

from __future__ import annotations

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
    observations = _run_backend("inproc", cases)
    assert set(observations) == {"t/kill", "t/never-reached"}
    assert observations["t/kill"] == observe_worker_died()
    assert observations["t/never-reached"] == observe_worker_died()


def test_a_normal_run_is_unaffected_by_the_worker_died_fallback():
    observations = _run_backend("inproc", [Case(id="t/sum", source="fnp.sum(V)")])
    assert observations["t/sum"]["outcome"] == "returned"


def test_looks_like_infrastructure_on_empty_observations():
    assert _looks_like_infrastructure({}) is True


def test_looks_like_infrastructure_when_most_cases_died():
    observations = {
        "a": observe_worker_died(),
        "b": observe_worker_died(),
        "c": {"outcome": "returned"},
    }
    assert _looks_like_infrastructure(observations) is True


def test_does_not_look_like_infrastructure_when_most_cases_are_fine():
    observations = {
        "a": {"outcome": "returned"},
        "b": {"outcome": "returned"},
        "c": observe_worker_died(),
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
        runner, "_run_backend", lambda backend, cases: {"t/sum": observe_worker_died()}
    )

    result = run_corpus([Case(id="t/sum", source="fnp.sum(V)")])

    assert result.infrastructure_failure is not None
    assert result.divergences == []
    assert stopped == [sentinel]


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
