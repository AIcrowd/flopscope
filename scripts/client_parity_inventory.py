"""Run the client-parity harness and emit a categorized failure inventory.

Runs the local wrapper test files (NumPy's own test classes subclassed as local
files so the autouse server/budget fixture fires and ops route to the flopscope
CLIENT) and parses the JUnit XML it emits. Running NumPy's suites via ``--pyargs``
instead would collect them from site-packages, outside the conftest scope, so
the fixture never fires and the measurement degrades to a native-vs-native no-op.
JUnit is used (not ``--report-log``) because it ships with pytest — no extra
dependency / lockfile churn for a measurement-only tool.

Output: a markdown report with the run totals and every failing test grouped by
exception signature, split into candidate **fix-now** (a real participant-usable
API gap) vs candidate **xfail** (by-design: immutability / budget / size-cap;
proxy-internal: memory-layout / strides). This feeds the Phase-1 decision gate;
it does NOT fix anything.

Note on coverage: the numpy suite exercises the patched numpy *functions* against
native ndarrays, so it does NOT surface RemoteArray operator/method/namespace
gaps (bitwise &, .argsort, flopscope.numpy package). Those are catalogued
directly in tests/client_compat/test_audit_gaps.py and
test_native_feature_smoke.py; this report points at them.
"""

from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# Local wrapper test files that subclass NumPy's own test classes so the autouse
# server/budget fixture actually fires. Running NumPy's suites via ``--pyargs``
# instead collects them from site-packages — outside the conftest's scope — so
# the budget fixture never activates, the patch wrapper sees no active
# BudgetContext, and every op delegates back to native NumPy (a native-vs-native
# no-op). These wrapper files are what genuinely exercises the CLIENT.
SUITE_FILES = [
    "tests/client_compat/test_numpy_function_classes.py",
    "tests/client_compat/methods/test_numpy_classes.py",
]

_JUNIT = "/tmp/client_parity_junit.xml"
_BY_DESIGN = re.compile(
    r"immutable|budget|too large|exceeds .* limit|contiguous|strides|server-side",
    re.IGNORECASE,
)
_EXC = re.compile(r"([A-Za-z_][\w.]*(?:Error|Exception|Warning)): ([^\n]*)")


def run() -> None:
    cmd = [
        "uv",
        "run",
        "pytest",
        *SUITE_FILES,
        "-p",
        "no:cacheprovider",
        "-n",
        "auto",
        "-q",
        f"--junit-xml={_JUNIT}",
    ]
    # check=False: a non-zero exit (failures exist) is the normal measurement case.
    subprocess.run(cmd, check=False)


def _signature(message: str, text: str) -> str:
    blob = f"{message}\n{text}"
    m = _EXC.search(blob)
    if not m:
        return "UNKNOWN"
    exc, msg = m.group(1), m.group(2).strip()
    # Normalize volatile bits so similar failures group together.
    msg = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", msg)
    msg = re.sub(r"\d[\d,]*", "N", msg)
    return f"{exc}: {msg[:90]}"


def parse() -> dict:
    tree = ET.parse(_JUNIT)
    root = tree.getroot()
    suites = root.findall(".//testsuite") or [root]
    totals = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    fails: dict[str, list[str]] = defaultdict(list)
    for ts in suites:
        for k in totals:
            totals[k] += int(ts.get(k, 0))
        for tc in ts.findall("testcase"):
            node = f"{tc.get('classname', '')}::{tc.get('name', '')}"
            for tag in ("failure", "error"):
                el = tc.find(tag)
                if el is not None:
                    sig = _signature(el.get("message", ""), el.text or "")
                    fails[sig].append(node)
    return {"totals": totals, "fails": fails}


def report(data: dict) -> None:
    t = data["totals"]
    fails = data["fails"]
    n_fail = sum(len(v) for v in fails.values())
    passed = t["tests"] - t["failures"] - t["errors"] - t["skipped"]
    print("# Client-parity failure inventory\n")
    print(
        f"Run totals: {t['tests']} collected — ~{passed} passed, "
        f"{t['failures']} failed, {t['errors']} errored, {t['skipped']} skipped.\n"
    )
    print(
        f"Distinct failure signatures: {len(fails)}; total failing tests: {n_fail}.\n"
    )
    fix_now, xfail = [], []
    for sig, nodes in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        (xfail if _BY_DESIGN.search(sig) else fix_now).append((sig, nodes))
    for title, group in [("CANDIDATE fix-now", fix_now), ("CANDIDATE xfail", xfail)]:
        print(f"\n## {title}\n")
        if not group:
            print("_(none)_")
        for sig, nodes in group:
            print(f"- [{len(nodes):4d}] {sig}")
            print(f"      e.g. {nodes[:3]}")
    print("\n## Operator / method / namespace gaps (not surfaced by the numpy suite)\n")
    print(
        "The numpy suite operates on native ndarrays, so RemoteArray dunders/"
        "methods and the flopscope.numpy package structure are NOT exercised "
        "here. See tests/client_compat/test_audit_gaps.py (bitwise & | ^ ~ << >>, "
        ".argsort/.diagonal, dtype=type-object, flopscope.numpy submodule import) "
        "and test_native_feature_smoke.py (flopscope.accounting) for those gaps."
    )


def main() -> int:
    run()
    report(parse())
    return 0


if __name__ == "__main__":
    sys.exit(main())
