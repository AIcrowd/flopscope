"""Drive both backends over the corpus and diff the results.

The parent process MUST NOT import either backend: doing so silently pins one
of them for the whole run. Only workers import a backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from tests.client_compat._server_fixture import start_server, stop_server
from tests.parity.case import Case
from tests.parity.compare import Divergence, compare_observations
from tests.parity.observe import observe_worker_died

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKENDS = ("inproc", "client")
_CASE_TIMEOUT_S = 30.0
#: Hard ceiling on one backend's whole-corpus run, regardless of case count.
_RUN_TIMEOUT_CAP_S = 1800.0
_CONNECTION_MARKERS = ("ConnectionError", "Again", "ZMQError", "NoBudgetContext")


@dataclass
class RunResult:
    divergences: list[Divergence] = field(default_factory=list)
    flaky: list[str] = field(default_factory=list)
    observations: dict[str, dict[str, dict]] = field(default_factory=dict)
    infrastructure_failure: str | None = None


def _run_backend(backend: str, cases: list[Case]) -> dict[str, dict]:
    """Run the whole corpus on one backend; return observations by case id."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    payload = "".join(json.dumps(case.to_json()) + "\n" for case in cases)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "tests.parity.worker", f"--backend={backend}"],
            input=payload,
            capture_output=True,
            text=True,
            cwd=_ROOT,
            env=env,
            timeout=min(_CASE_TIMEOUT_S * max(len(cases), 1), _RUN_TIMEOUT_CAP_S),
            check=False,
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        # A hung worker never gets to print anything more; treat whatever it
        # emitted before the hang as final and let the setdefault loop below
        # fill in the rest as worker_died so a stuck backend can't wedge the
        # whole run.
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    out: dict[str, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        out[record.pop("id")] = record
    # A worker that died mid-corpus leaves later cases unrecorded.
    for case in cases:
        out.setdefault(case.id, observe_worker_died())
    return out


def _looks_like_infrastructure(observations: dict[str, dict]) -> bool:
    if not observations:
        return True
    bad = sum(
        1
        for obs in observations.values()
        if obs.get("outcome") == "worker_died"
        or any(marker in str(obs.get("exc_type", "")) for marker in _CONNECTION_MARKERS)
    )
    return bad > len(observations) // 2


def run_corpus(cases: list[Case]) -> RunResult:
    """Run *cases* on both backends twice each and diff the results.

    The client backend needs a live flopscope-server. ``start_server`` sets
    ``FLOPSCOPE_SERVER_URL`` in this process's environment, which
    ``_run_backend`` inherits into the worker; it also allocates a per-xdist
    -worker port, so concurrent pytest workers do not collide.
    """
    result = RunResult()
    first: dict[str, dict[str, dict]] = {}
    second: dict[str, dict[str, dict]] = {}

    server = start_server()
    try:
        for backend in _BACKENDS:
            first[backend] = _run_backend(backend, cases)
            if _looks_like_infrastructure(first[backend]):
                result.infrastructure_failure = (
                    f"{backend} backend failed to run the corpus; this is an "
                    f"infrastructure failure, not a parity failure"
                )
                return result
            second[backend] = _run_backend(backend, cases)
    finally:
        stop_server(server)

    result.observations = first

    for case in cases:
        # Self-check: a backend that disagrees with itself is nondeterministic.
        flaky = any(
            first[backend].get(case.id) != second[backend].get(case.id)
            for backend in _BACKENDS
        )
        if flaky:
            result.flaky.append(case.id)
            continue
        result.divergences.extend(
            compare_observations(
                case.id, first["inproc"][case.id], first["client"][case.id]
            )
        )
    return result
