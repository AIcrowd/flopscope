"""Source 1: every registered op crossed with a fixed set of argument patterns.

An op that works in-process and raises AttributeError on the client shows up as
an `outcome` divergence, which is how this source catches a whole missing
surface without anyone hand-listing it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache

from tests.parity.case import Case

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

#: (pattern name, source template). ``{op}`` is the dotted op name.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("array", "fnp.{op}(A)"),
    ("array-pair", "fnp.{op}(A, B)"),
    ("vector", "fnp.{op}(V)"),
    ("axis-int", "fnp.{op}(A, axis=0)"),
    ("axis-tuple", "fnp.{op}(A, axis=(0, 1))"),
    ("axis-negative", "fnp.{op}(A, axis=-1)"),
    ("keepdims", "fnp.{op}(A, axis=0, keepdims=True)"),
    ("dtype", "fnp.{op}(A, dtype='float64')"),
    ("scalar-operand", "fnp.{op}(V, 2.0)"),
    ("integer", "fnp.{op}(I)"),
    ("boolean", "fnp.{op}(M)"),
    ("empty", "fnp.{op}(E)"),
    ("zero-d", "fnp.{op}(S)"),
    ("no-arg", "fnp.{op}()"),
)

_DUMP_NAMES = """
import json, sys
sys.path.insert(0, {root!r})
from flopscope._registry import REGISTRY
print(json.dumps(sorted(REGISTRY)))
"""


@lru_cache(maxsize=1)
def op_names() -> list[str]:
    """Return every op name from the CORE registry (the source of truth)."""
    proc = subprocess.run(
        [sys.executable, "-c", _DUMP_NAMES.format(root=os.path.join(_ROOT, "src"))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def undriven() -> dict[str, str]:
    """Ops no generic pattern can drive, with the reason. Counted, not dropped.

    Populated from the first full run: an op whose every pattern raises
    identically on BOTH backends is not being exercised, and saying so is more
    honest than reporting it as agreement.
    """
    return {}


def build() -> tuple[Case, ...]:
    return tuple(
        Case(
            id=f"grid/{op}::{pattern_name}",
            source=template.format(op=op),
            tags=frozenset({"src:grid", f"pattern:{pattern_name}"}),
        )
        for op in op_names()
        for pattern_name, template in PATTERNS
    )
