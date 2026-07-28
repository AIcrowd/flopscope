"""Execute the parity corpus against exactly ONE backend.

Run as::

    python -m tests.parity.worker --backend=inproc
    python -m tests.parity.worker --backend=client

Reads one JSON-encoded Case per line on stdin; writes one JSON observation per
line on stdout. The two backends install under the same import name
(``flopscope``) and cannot coexist in one interpreter, which is why this is a
separate process rather than a function call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tests.parity.case import Case
from tests.parity.observe import (
    observe_exception,
    observe_record_failure,
    observe_result,
)

#: Fixtures are rebuilt per case: an audit pass was discarded to in-place-op
#: contamination, so shared mutable fixtures are a known failure mode.
#:
#: The fixed seed below (an arbitrary but memorable constant, not derived
#: from anything) reseeds the global RNG identically on both backends before
#: every case. Without it, a case that draws from `fnp.random.*` compares two
#: independently-seeded streams, so the two backends only agree by chance
#: (e.g. 1-in-6 for a 6-element `random.choice`) - a coin flip, not a
#: comparison. Confirmed by direct measurement (not just assumed) that a
#: seeded draw is bit-identical between the two backends: `fnp.random.seed
#: (1234); fnp.random.choice(V)` returns `1.0` on both, and `fnp.random.seed
#: (1234); fnp.random.rand(3)` returns the same three float64 values to the
#: last bit. This call costs 0 FLOPs on both backends (`random.seed` is a
#: free/configuration op, not a sampler), and even if it billed something,
#: this line runs as part of fixture construction in `build_namespace`,
#: which executes (and is fully accounted for) before `run_case` takes its
#: `before` FLOP snapshot - so it can never leak into a case's measured
#: delta either way.
FIXTURE_SOURCE = """
fnp.random.seed(1234)
A = fnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
B = fnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
V = fnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype="float32")
I = fnp.array([3, 1, 4, 1, 5, 9], dtype="int64")
M = fnp.array([True, False, True, False, True, False], dtype="bool")
E = fnp.array([], dtype="float32")
S = fnp.array(2.0, dtype="float32")
"""

_BUDGET = 10**15


def build_namespace(fnp) -> dict:
    """Return a fresh namespace with `fnp` bound and the fixtures rebuilt."""
    namespace: dict = {"fnp": fnp}
    exec(FIXTURE_SOURCE, namespace)  # noqa: S102 - fixtures are ours, not input
    return namespace


def run_case(namespace: dict, case: Case, ctx) -> dict:
    """Execute one case in *namespace*, recording the outcome and FLOP delta.

    ``ctx.summary()`` is called before every ``flops_used`` read, including
    ``before``. The client backend's ``flops_used`` is a local cache that is
    only refreshed on context enter/exit or an explicit ``summary()`` call —
    never automatically after an op dispatch — and this worker holds one
    ambient context open for the whole corpus rather than one per case, so
    without this the client's per-case delta would read as 0 for every case.
    ``summary()`` is a read-only status query on both backends (confirmed: it
    prints nothing and bills no FLOPs itself), so this is safe to call
    unconditionally, including on the in-process backend where it is a no-op
    refresh.
    """
    local = dict(namespace)
    ctx.summary()
    before = ctx.flops_used
    try:
        if case.setup:
            exec(case.setup, local)  # noqa: S102 - corpus is ours, not user input
        value = eval(case.source, local)  # noqa: S307 - corpus is ours
    except Exception as exc:  # noqa: BLE001 - recording, not handling
        ctx.summary()
        return {"id": case.id, **observe_exception(exc, ctx.flops_used - before)}
    ctx.summary()
    delta = ctx.flops_used - before
    try:
        result = observe_result(value, delta)
    except Exception as exc:  # noqa: BLE001 - recording, not handling
        # observe_result describing an arbitrary returned value is itself
        # fallible (e.g. a registry op handing back a class instead of an
        # instance, whose attributes lie about their own shape). A failure
        # here must be recorded like any other outcome, not propagate and
        # kill the worker mid-stream, discarding every case still queued.
        result = observe_record_failure(exc, delta)
    return {"id": case.id, **result}


def _import_backend(backend: str):
    """Import exactly one backend and return its numpy-shaped module."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend == "client":
        sys.path.insert(0, os.path.join(root, "flopscope-client", "src"))
    else:
        sys.path.insert(0, os.path.join(root, "src"))
    import flopscope
    import flopscope.numpy as fnp

    return flopscope, fnp


def run_stream(fnp, ctx, stdin, stdout) -> None:
    """Read cases from *stdin*, write one observation line per case to *stdout*.

    Fixtures are rebuilt fresh for every case (`build_namespace` is called
    inside this loop, not hoisted above it): a prior measurement pass had to
    be discarded because an in-place operation in one case contaminated a
    later one via a shared fixture object.
    """
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            case = Case.from_json(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - a bad line must not kill the worker
            # Never write diagnostics to stdout: it carries the observation
            # records and the parent parses it line-by-line as JSON.
            print(f"worker: skipping malformed case line: {exc}", file=sys.stderr)
            continue
        observation = run_case(build_namespace(fnp), case, ctx)
        stdout.write(json.dumps(observation) + "\n")
        stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("inproc", "client"), required=True)
    args = parser.parse_args(argv)

    flopscope, fnp = _import_backend(args.backend)

    # ONE ambient context for the whole run: the client rejects nested contexts.
    ctx = flopscope.BudgetContext(flop_budget=_BUDGET, quiet=True)
    ctx.__enter__()
    try:
        run_stream(fnp, ctx, sys.stdin, sys.stdout)
    finally:
        ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
