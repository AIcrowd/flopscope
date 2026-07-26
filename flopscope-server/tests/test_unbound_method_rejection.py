"""Fully-qualified ``random.RandomState.*`` / ``random.Generator.*`` op names
resolve to numpy methods that require an instance. numpy's C implementations do
not type-check ``self``, so calling one unbound with an arbitrary array
segfaults the interpreter (verified against plain numpy 2.2.x — an upstream
defect). The server dispatches such an op by walking the dotted attribute chain
and calling the resolved object with the request's arguments, so an unguarded
dispatch crashes the whole server process, not just the request.

The server must refuse these names with a typed error instead. The legitimate
way to call a generator method — the client's un-prefixed ``Generator.<method>``
form, routed onto a real instance — is unaffected and covered elsewhere.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from flopscope_server._request_handler import _get_flopscope_func

# Names that resolve THROUGH a class object, i.e. unbound methods. The rule is
# structural (traversal crosses a ``type``), so this list is illustrative, not
# the enforcement mechanism.
_UNBOUND = [
    "random.RandomState.seed",
    "random.RandomState.get_state",
    "random.Generator.spawn",
    "random.Generator.random",
]

# Genuine module-level functions and constructors that must keep resolving.
_LEGITIMATE = [
    "random.random",
    "random.default_rng",
    "linalg.svd",
    "sum",
    "stats.norm.pdf",
]


@pytest.mark.parametrize("op", _UNBOUND)
def test_unbound_method_names_are_rejected_not_resolved(op):
    from flopscope import UnsupportedFunctionError

    with pytest.raises(UnsupportedFunctionError):
        _get_flopscope_func(op)


@pytest.mark.parametrize("op", _LEGITIMATE)
def test_legitimate_ops_still_resolve(op):
    assert callable(_get_flopscope_func(op))


def test_server_survives_an_unbound_method_request():
    """End to end, in a subprocess, because a SIGSEGV cannot be caught in-process.

    Drives a real in-process ``RequestHandler`` exactly as the server does, sends
    the crashing op, and asserts the process exits cleanly with a typed error
    rather than dying on signal 11. Before the fix this child exits -11.
    """
    child = """
import sys
import numpy as np
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session

session = Session(flop_budget=10**15)
handler = RequestHandler(session)
handle = session.store_array(np.array([1.0, 2.0, 3.0], dtype="float32"))
resp = handler.handle(
    {"op": "random.RandomState.seed", "args": [{"__handle__": handle}], "kwargs": {}}
)
assert resp["status"] == "error", resp
assert resp["error_type"] == "UnsupportedFunctionError", resp
session.close()
print("SURVIVED")
"""
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"server process exited {proc.returncode} "
        f"(-11/139 == SIGSEGV); stderr:\n{proc.stderr[-500:]}"
    )
    assert "SURVIVED" in proc.stdout
