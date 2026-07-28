"""Drive both backends over the corpus and diff the results.

The parent process MUST NOT import either backend: doing so silently pins one
of them for the whole run. Only workers import a backend.

A worker can die mid-corpus (a segfault in native code, for instance, which no
Python ``try``/``except`` can catch). That must cost only the one in-flight
case: the parent records that single case as ``worker_died`` and restarts a
fresh worker on the cases after it, rather than backfilling every case the
dead worker never got to as fake failures.
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
#: Own port range for this harness's server, well clear of
#: ``tests/client_compat``'s (15571 + one slot per xdist worker, so up to the
#: high 15500s on a many-core CI box). A leaked or TIME_WAIT-lingering server
#: from that suite must never be mistakable for this harness's own — the two
#: harnesses can otherwise run back-to-back in the same CI job.
_PARITY_BASE_PORT = 15671
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
_STDERR_EXCERPT_LEN = 2500
#: Hard ceiling on how many times ``_run_backend`` will restart a dying
#: worker for one backend's run. Without this, a corpus that kills the
#: worker on literally every case (e.g. a broken import) would restart
#: forever, one case at a time, and never finish.
_MAX_WORKER_RESTARTS = 50
#: A worker that needed at least this many restarts is not "unlucky on a
#: couple of cases" any more: it is trending toward burning through the
#: whole restart budget above, which itself always counts as
#: infrastructure. Set well under the cap (a fifth of it) so a run that is
#: clearly heading for exhaustion is reported honestly rather than only
#: once it actually hits the ceiling.
_SYSTEMIC_RESTART_THRESHOLD = 10


@dataclass
class RunResult:
    divergences: list[Divergence] = field(default_factory=list)
    flaky: list[str] = field(default_factory=list)
    observations: dict[str, dict[str, dict]] = field(default_factory=dict)
    #: Captured worker stderr from each backend's first run, keyed by backend
    #: name. Empty string when a backend produced no stderr output. When a
    #: worker was restarted mid-run, this is every restart's stderr for that
    #: run, concatenated in order.
    stderr: dict[str, str] = field(default_factory=dict)
    #: How many times each backend's worker was restarted after dying
    #: mid-corpus, keyed by backend name, from each backend's first run. 0
    #: for a clean run; a nonzero count is normal (one segfaulting case
    #: costs exactly one restart) but a large one is a red flag on its own -
    #: see ``_looks_like_infrastructure``.
    restarts: dict[str, int] = field(default_factory=dict)
    infrastructure_failure: str | None = None


def _as_text(value) -> str:
    """Decode subprocess output that may arrive as bytes.

    ``TimeoutExpired`` carries bytes on ``.stdout``/``.stderr`` even when the
    call requested text mode, so callers must not assume ``str``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return ""


def _run_worker(backend: str, cases: list[Case]) -> tuple[dict[str, dict], str]:
    """Run *cases* through exactly ONE worker process; no restart on death.

    Returns ``(observations by case id for every case the worker actually
    emitted a record for, captured stderr text)``. ``stderr`` is ``""`` when
    the worker produced none. If the worker dies partway through *cases*, the
    returned dict simply has fewer entries than ``cases`` - the caller
    (``_run_backend``) is what turns "fewer entries than expected" into a
    ``worker_died`` record and a restart; this function only ever reports
    what the worker actually said.

    Empirically verified (not just assumed): the worker flushes stdout after
    every case (see ``worker.run_stream``), and a killed child's already
    -flushed writes are sitting in the OS pipe buffer, which survives the
    process's death - closing the write end just signals EOF to the reader.
    ``subprocess.run`` drains stdout concurrently while the child runs, so
    nothing flushed before a segfault is lost. Confirmed by feeding a worker
    two cheap cases followed by the real segfaulting
    ``grid/random.Generator.spawn::scalar-operand`` case: the worker exits
    139 with no traceback, and stdout still contains the two prior cases'
    records in full. Because of this, cases are fed to the worker in one
    shot per restart (the whole remaining slice), not in small chunks: there
    is nothing further to lose to buffering, and chunking would only add
    process-spawn overhead.
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
        # emitted before the hang as final. `_run_backend` treats a timeout
        # exactly like a death: the next unrecorded case is marked
        # worker_died and a fresh worker resumes after it, so a stuck
        # backend can't wedge the whole run.
        # `TimeoutExpired` carries BYTES on .stdout/.stderr even when the call
        # passed text=True, so treating a non-str as "no output" would discard
        # every record the worker flushed before it hung. `_run_backend` would
        # then blame the first case of the remaining slice instead of the one
        # that actually hung, misattributing the death and re-running cases the
        # worker had already completed.
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
    out: dict[str, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        out[record.pop("id")] = record
    return out, stderr or ""


def _run_backend(backend: str, cases: list[Case]) -> tuple[dict[str, dict], str, int]:
    """Run the whole corpus on one backend, restarting the worker on death.

    Feeds *cases* to a worker via ``_run_worker``. If the worker exits before
    every case in the slice it was given got a record, exactly ONE case pays
    for that death - the next case after the ones it did emit records for,
    per the worker's in-order-per-line protocol - and a fresh worker resumes
    from the case after that. This repeats until every case has a record or
    ``_MAX_WORKER_RESTARTS`` is exhausted, at which point every remaining
    case is marked ``worker_died`` and the loop stops (so a pathological
    corpus that kills the worker on every case cannot loop forever).

    Returns ``(observations by case id, captured stderr text, restart
    count)``. ``stderr`` is every worker invocation's stderr for this run,
    concatenated in order (empty ones dropped); ``""`` when none produced
    any.
    """
    out: dict[str, dict] = {}
    stderr_chunks: list[str] = []
    restarts = 0
    start = 0
    while start < len(cases):
        remaining = cases[start:]
        emitted, stderr_text = _run_worker(backend, remaining)
        if stderr_text:
            stderr_chunks.append(stderr_text)
        out.update(emitted)
        if len(emitted) >= len(remaining):
            # The worker produced a record for every case it was given: it
            # made it through the rest of the corpus.
            break
        # The worker died (or hung and was killed on timeout) before
        # finishing. It emits one record per case, in order, so the count of
        # records it did emit is exactly how far it got; the next case in
        # the slice it was fed is the one that killed it.
        died_case = cases[start + len(emitted)]
        out[died_case.id] = observe_worker_died()
        start = start + len(emitted) + 1
        if start >= len(cases):
            break  # that was the last case; nothing left to resume.
        if restarts >= _MAX_WORKER_RESTARTS:
            # Restart budget exhausted: stop trying to resume and mark
            # everything still unrun as dead, same as a single run used to
            # do for the whole corpus.
            for case in cases[start:]:
                out.setdefault(case.id, observe_worker_died())
            break
        restarts += 1
    return out, "\n".join(stderr_chunks), restarts


def _looks_like_infrastructure(
    observations: dict[str, dict], restarts: int = 0
) -> bool:
    """A backend counts as having failed to run the corpus — as opposed to
    merely producing individual failing cases — when any of:

    - the worker needed ``_SYSTEMIC_RESTART_THRESHOLD`` or more restarts to
      get through the corpus (this subsumes hitting ``_MAX_WORKER_RESTARTS``
      outright, since the cap is well above the threshold). Now that a dying
      worker costs only the one case that killed it, a handful of restarts
      (one segfaulting case among thousands, say) is normal and expected;
      this many is the worker dying over and over, not bad luck on a couple
      of cases,
    - a majority of cases never got the chance to raise anything (the worker
      died outright, so there is no exception to compare) — kept as a
      fallback for callers that pass observations without a restart count,
      and for the pathological case of a corpus so small the restart cap
      fills it with ``worker_died`` before ``_SYSTEMIC_RESTART_THRESHOLD`` is
      reached, or
    - a majority raised the exact same transport-failure exception type.

    The "exact same type" requirement on the last one is deliberate: a corpus
    where different cases fail for different reasons (including a domain
    exception a corpus family provokes on purpose, e.g.
    ``NoBudgetContextError``) is real per-case signal, not a broken backend,
    and must not be swallowed here.
    """
    if not observations:
        return True
    if restarts >= _SYSTEMIC_RESTART_THRESHOLD:
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

    server = start_server(_PARITY_BASE_PORT)
    try:
        for backend in _BACKENDS:
            observations, stderr_text, restarts = _run_backend(backend, cases)
            first[backend] = observations
            result.stderr[backend] = stderr_text
            result.restarts[backend] = restarts
            if _looks_like_infrastructure(first[backend], restarts):
                # Keep the TAIL, not the head: a Python traceback puts the
                # exception type and message last, so truncating from the front
                # reliably discards the only part worth reading.
                excerpt = stderr_text.strip()[-_STDERR_EXCERPT_LEN:]
                detail = f" worker stderr: {excerpt!r}" if excerpt else ""
                restart_note = (
                    f" ({restarts} worker restart{'s' if restarts != 1 else ''})"
                    if restarts
                    else ""
                )
                result.infrastructure_failure = (
                    f"{backend} backend failed to run the corpus; this is an "
                    f"infrastructure failure, not a parity failure."
                    f"{restart_note}{detail}"
                )
                return result
            second[backend], _, _ = _run_backend(backend, cases)
    finally:
        stop_server(server)

    result.observations = first

    for case in cases:
        # Self-check: a backend that disagrees with itself is nondeterministic.
        # Compare only the modeled dimensions, via the same comparator used for
        # the real diff. Comparing whole records would also compare `exc_msg`,
        # which the comparator deliberately ignores because messages carry
        # process-specific detail like memory addresses -- a case whose message
        # merely differs between two worker processes would be quarantined and
        # then skipped entirely, silently shrinking what the gate compares.
        flaky = any(
            compare_observations(
                case.id,
                first[backend].get(case.id, {}),
                second[backend].get(case.id, {}),
            )
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
