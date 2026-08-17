"""A structured result must reach the client as the same namedtuple it is
in-process.

``fnp.linalg.svd(a).U`` worked in-process and raised ``AttributeError`` against
a server: the wire's multi-result form carried no slot for the container type,
and a namedtuple IS a tuple, so every structured result flowed down the path
built for ``nonzero`` (a homogeneous list of arrays) and arrived stripped.

The expectation here is measured, not written down: the in-process backend runs
the same calls in a subprocess and its field names ARE the expectation. A numpy
release that renames a field, or adds a structured result, moves both sides
together — a hard-coded ``("U", "S", "Vh")`` would instead go quietly stale and
keep passing while the real contract drifted.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import textwrap

import pytest

import flopscope.numpy as fnp  # the CLIENT; the conftest orders sys.path

from ._namedtuple_cases import (
    PLAIN_CASES,
    STRUCTURED_CASES,
    collect_fields,
    container_fields,
)
from ._server_fixture import _REAL_SRC, _ROOT, _VENV_PYTHON


@functools.cache
def _inproc_fields() -> dict[str, dict[str, list[str] | None]]:
    """Field names the IN-PROCESS backend produces, measured in a subprocess.

    A subprocess because this session's ``sys.path`` (and its patched numpy) is
    owned by the client harness; the in-process backend has to be observed
    somewhere that has not been rearranged around the client.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {_REAL_SRC!r})
        sys.path.insert(0, {os.path.join(_ROOT, "tests", "client_compat")!r})

        from _namedtuple_cases import (
            PLAIN_CASES, STRUCTURED_CASES, collect_fields,
        )

        import flopscope as fc
        import flopscope.numpy as fnp

        with fc.BudgetContext(flop_budget=10**12, quiet=True):
            out = {{
                "structured": collect_fields(fnp, STRUCTURED_CASES),
                "plain": collect_fields(fnp, PLAIN_CASES),
            }}
        print(json.dumps(out))
        """
    )
    proc = subprocess.run(
        [_VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_in_process_backend_really_does_return_namedtuples():
    # Without this the parity assertion below could pass vacuously, with both
    # backends agreeing that every structured result has no fields at all.
    structured = _inproc_fields()["structured"]
    assert structured, "no structured cases were driven"
    missing = [name for name, fields in structured.items() if not fields]
    assert not missing, f"in-process results are not namedtuples: {missing}"


def test_client_field_names_match_the_in_process_backend():
    assert collect_fields(fnp, STRUCTURED_CASES) == _inproc_fields()["structured"]


def test_plain_tuple_results_match_the_in_process_backend():
    # The other direction of the same contract: nothing may acquire a container
    # it does not have in-process.
    assert collect_fields(fnp, PLAIN_CASES) == _inproc_fields()["plain"]
    assert all(fields is None for fields in _inproc_fields()["plain"].values())


@pytest.mark.parametrize("case_name", sorted(STRUCTURED_CASES))
def test_every_field_is_reachable_by_name_and_matches_its_position(case_name):
    result = STRUCTURED_CASES[case_name](fnp)
    fields = container_fields(result)
    assert fields is not None, f"{case_name} lost its container type"
    for position, field in enumerate(fields):
        assert getattr(result, field) is result[position]


@pytest.mark.parametrize("case_name", sorted(STRUCTURED_CASES))
def test_a_structured_result_still_unpacks_and_indexes(case_name):
    # examples/04_svd_usage.py unpacks these positionally; rebuilding the
    # container must not disturb that.
    result = STRUCTURED_CASES[case_name](fnp)
    assert isinstance(result, tuple)
    unpacked = [*result]
    assert len(unpacked) == len(result)
    assert all(a is b for a, b in zip(unpacked, result, strict=True))
