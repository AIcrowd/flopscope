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
from tests.parity.observe import observe_exception, observe_result

#: Fixtures are rebuilt per case: an audit pass was discarded to in-place-op
#: contamination, so shared mutable fixtures are a known failure mode.
FIXTURE_SOURCE = """
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
    """Execute one case in *namespace*, recording the outcome and FLOP delta."""
    local = dict(namespace)
    before = ctx.flops_used
    try:
        if case.setup:
            exec(case.setup, local)  # noqa: S102 - corpus is ours, not user input
        value = eval(case.source, local)  # noqa: S307 - corpus is ours
    except BaseException as exc:  # noqa: BLE001 - recording, not handling
        return {"id": case.id, **observe_exception(exc, ctx.flops_used - before)}
    return {"id": case.id, **observe_result(value, ctx.flops_used - before)}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("inproc", "client"), required=True)
    args = parser.parse_args(argv)

    flopscope, fnp = _import_backend(args.backend)

    # ONE ambient context for the whole run: the client rejects nested contexts.
    ctx = flopscope.BudgetContext(flop_budget=_BUDGET, quiet=True)
    ctx.__enter__()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            case = Case.from_json(json.loads(line))
            observation = run_case(build_namespace(fnp), case, ctx)
            sys.stdout.write(json.dumps(observation) + "\n")
            sys.stdout.flush()
    finally:
        ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
