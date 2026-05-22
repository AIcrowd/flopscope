"""Shared pytest configuration and fixtures."""

import os
import threading
import time as _time

import pytest

import flopscope._weights as weights_module
from flopscope._budget import _reset_global_default, budget_reset
from flopscope._weights import reset_weights


@pytest.fixture(autouse=True)
def reset_global_budget():
    """Ensure no global default BudgetContext leaks between tests."""
    _reset_global_default()
    budget_reset()
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()
    yield
    _reset_global_default()
    budget_reset()
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()


_SESSION_STATE = {
    "finished": False,
    "exitstatus": 0,
    "failed_count": 0,
    "errored_count": 0,
    "report_count": 0,
    "last_report_ts": 0.0,
    "failed_nodeids": [],
    "errored_nodeids": [],
}


def pytest_runtest_logreport(report):
    """Track per-test progress so the watchdog can detect when the run idles.

    The xdist controller may hang in ``runtestloop`` before reaching
    ``pytest_sessionfinish`` (where ``exitstatus`` would normally be set).
    We record the timestamp of each logreport so the watchdog can detect
    when no reports have arrived for a while - which is the signature of a
    pytest-xdist worker-pipe-close deadlock after the actual tests have
    completed.
    """
    _SESSION_STATE["report_count"] += 1
    _SESSION_STATE["last_report_ts"] = _time.monotonic()
    if report.when == "call":
        if report.failed:
            _SESSION_STATE["failed_count"] += 1
            _SESSION_STATE["failed_nodeids"].append(report.nodeid)
    elif report.failed:  # setup or teardown error
        _SESSION_STATE["errored_count"] += 1
        _SESSION_STATE["errored_nodeids"].append(report.nodeid)


def pytest_sessionstart(session):
    """Arm an idle-detection watchdog at session start.

    pytest-xdist's worker-shutdown race (see pytest issue #7250 and
    pytest-xdist's worker-pipe-close deadlock on Python 3.10-3.14)
    occasionally leaves the controller's runtestloop blocked on
    ``queue.get`` indefinitely, even after all tests have completed and
    workers have exited as zombies. The ``pytest_sessionfinish`` hook is
    never reached without intervention.

    Strategy: track the timestamp of every per-test logreport. If we've
    seen at least ``MIN_REPORTS_FOR_IDLE_CHECK`` reports and no new report
    has arrived for ``IDLE_TIMEOUT_SECS``, assume the run has actually
    completed and xdist is hung in teardown. Force-exit with the
    synthesized status (1 if any failures, 0 otherwise).

    Hard ceiling: ``MAX_SESSION_SECS`` from session start. Catches the
    pathological case where the whole suite hangs before producing
    enough reports for idle detection.

    Only runs on the controller (xdist workers skip this).
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    IDLE_TIMEOUT_SECS = 20
    MIN_REPORTS_FOR_IDLE_CHECK = 100
    MAX_SESSION_SECS = 600
    POLL_INTERVAL_SECS = 5

    session_start = _time.monotonic()

    def _watchdog():
        import sys

        while True:
            _time.sleep(POLL_INTERVAL_SECS)
            if _SESSION_STATE["finished"]:
                return
            now = _time.monotonic()
            session_elapsed = now - session_start
            last_report_age = (
                now - _SESSION_STATE["last_report_ts"]
                if _SESSION_STATE["last_report_ts"]
                else session_elapsed
            )
            should_exit = False
            reason = ""
            if (
                _SESSION_STATE["report_count"] >= MIN_REPORTS_FOR_IDLE_CHECK
                and last_report_age >= IDLE_TIMEOUT_SECS
            ):
                should_exit = True
                reason = (
                    f"no test reports in {last_report_age:.0f}s after "
                    f"{_SESSION_STATE['report_count']} reports"
                )
            elif session_elapsed >= MAX_SESSION_SECS:
                should_exit = True
                reason = (
                    f"session exceeded hard ceiling of {MAX_SESSION_SECS}s "
                    f"(only {_SESSION_STATE['report_count']} reports)"
                )
            if should_exit:
                lines = [
                    f"\n[xdist-watchdog] {reason}; assuming xdist "
                    f"worker-pipe-close deadlock and force-exiting. "
                    f"Failures during the run: "
                    f"{_SESSION_STATE['failed_count']} failed, "
                    f"{_SESSION_STATE['errored_count']} errored.",
                ]
                for nodeid in _SESSION_STATE["failed_nodeids"]:
                    lines.append(f"  FAILED  {nodeid}")
                for nodeid in _SESSION_STATE["errored_nodeids"]:
                    lines.append(f"  ERRORED {nodeid}")
                sys.stderr.write("\n".join(lines) + "\n")
                sys.stderr.flush()
                rc = (
                    1
                    if (
                        _SESSION_STATE["failed_count"]
                        or _SESSION_STATE["errored_count"]
                    )
                    else 0
                )
                os._exit(rc)

    threading.Thread(target=_watchdog, daemon=True, name="xdist-watchdog").start()


def pytest_sessionfinish(session, exitstatus):
    """Record that the session ended cleanly so the watchdog skips force-exit."""
    _SESSION_STATE["finished"] = True
    if isinstance(exitstatus, int):
        _SESSION_STATE["exitstatus"] = exitstatus
