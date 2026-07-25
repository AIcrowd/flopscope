"""Drive both backends over the corpus and diff the results.

The parent process MUST NOT import either backend: doing so silently pins one
of them for the whole run. Only workers import a backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
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
#: Genuine transport failures only. A domain exception (e.g.
#: ``NoBudgetContextError``) that a corpus deliberately provokes on both
#: backends must NOT appear here: it would make a majority of a legitimate
#: corpus family look like a dead backend and silently swallow every
#: divergence.
_CONNECTION_MARKERS = ("ConnectionError", "ZMQError", "Again")
#: How much of a failing worker's stderr to fold into the infrastructure
#: failure message; the full text still lives on ``RunResult.stderr``.
_STDERR_EXCERPT_LEN = 500


@dataclass
class RunResult:
    divergences: list[Divergence] = field(default_factory=list)
    flaky: list[str] = field(default_factory=list)
    observations: dict[str, dict[str, dict]] = field(default_factory=dict)
    #: Captured worker stderr from each backend's first run, keyed by backend
    #: name. Empty string when a backend produced no stderr output.
    stderr: dict[str, str] = field(default_factory=dict)
    infrastructure_failure: str | None = None


def _run_backend(backend: str, cases: list[Case]) -> tuple[dict[str, dict], str]:
    """Run the whole corpus on one backend.

    Returns ``(observations by case id, captured stderr text)``. ``stderr`` is
    ``""`` when the worker produced none.
    """
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
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        # A hung worker never gets to print anything more; treat whatever it
        # emitted before the hang as final and let the setdefault loop below
        # fill in the rest as worker_died so a stuck backend can't wedge the
        # whole run.
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
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
    return out, stderr or ""


def _looks_like_infrastructure(observations: dict[str, dict]) -> bool:
    """A backend counts as having failed to run the corpus — as opposed to
    merely producing individual failing cases — when either:

    - a majority of cases never got the chance to raise anything (the worker
      died outright, so there is no exception to compare), or
    - a majority raised the exact same transport-failure exception type.

    The "exact same type" requirement is deliberate: a corpus where different
    cases fail for different reasons (including a domain exception a corpus
    family provokes on purpose, e.g. ``NoBudgetContextError``) is real
    per-case signal, not a broken backend, and must not be swallowed here.
    """
    if not observations:
        return True
    total = len(observations)
    died = sum(
        1 for obs in observations.values() if obs.get("outcome") == "worker_died"
    )
    if died > total // 2:
        return True
    transport_exc_types = [
        str(obs.get("exc_type", ""))
        for obs in observations.values()
        if any(marker in str(obs.get("exc_type", "")) for marker in _CONNECTION_MARKERS)
    ]
    if not transport_exc_types:
        return False
    _most_common_type, most_common_count = Counter(transport_exc_types).most_common(1)[
        0
    ]
    return most_common_count > total // 2


def run_corpus(cases: list[Case]) -> RunResult:
    """Run *cases* on both backends twice each and diff the results.

    The client backend needs a live flopscope-server. ``start_server`` sets
    ``FLOPSCOPE_SERVER_URL`` in this process's environment, which
    ``_run_backend`` inherits into the worker; it also allocates a per-xdist
    -worker port, so concurrent pytest workers do not collide.
    """
    if not cases:
        # A later stage that filters cases by tag can legitimately end up
        # with nothing to run; that is a trivially clean result, not an
        # infrastructure failure.
        return RunResult()

    result = RunResult()
    first: dict[str, dict[str, dict]] = {}
    second: dict[str, dict[str, dict]] = {}

    server = start_server()
    try:
        for backend in _BACKENDS:
            observations, stderr_text = _run_backend(backend, cases)
            first[backend] = observations
            result.stderr[backend] = stderr_text
            if _looks_like_infrastructure(first[backend]):
                excerpt = stderr_text.strip()[:_STDERR_EXCERPT_LEN]
                detail = f" worker stderr: {excerpt!r}" if excerpt else ""
                result.infrastructure_failure = (
                    f"{backend} backend failed to run the corpus; this is an "
                    f"infrastructure failure, not a parity failure.{detail}"
                )
                return result
            second[backend], _ = _run_backend(backend, cases)
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
