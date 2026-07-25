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


#: SEGFAULT EXCLUSIONS - do not remove without re-measuring.
#:
#: These three ops crash the in-process worker with SIGSEGV (exit 139) for
#: the patterns listed below, confirmed by running each case in its own
#: subprocess and observing the return code directly. A segfault is
#: uncatchable by any Python `try`/`except` (`tests/parity/worker.py` /
#: `tests/parity/runner.py`'s restart machinery exists for exactly this
#: class of crash), so these cases must never be fed to a worker at all -
#: they are excluded here, at corpus-assembly time, rather than relying on
#: the runner to survive them.
#:
#: All three are unbound-method calls (`fnp.random.RandomState.get_state(A)`
#: etc.) where a plain array fixture stands in for `self`; the patterns that
#: crash are exactly the ones that pass a fixture as the first positional
#: argument with no extra keyword (`array`, `array-pair`, `vector`,
#: `scalar-operand`, `integer`, `boolean`, `empty`, `zero-d`) - the patterns
#: that add a keyword argument (`axis-int`, `axis-tuple`, `axis-negative`,
#: `keepdims`, `dtype`) raise a normal, catchable `TypeError` instead
#: (rejected before the C code ever dereferences `self`), and `no-arg`
#: never supplies a `self` at all.
_SEGFAULT_PATTERNS_BY_OP: dict[str, tuple[str, ...]] = {
    "random.RandomState.get_state": (
        "array",
        "array-pair",
        "vector",
        "scalar-operand",
        "integer",
        "boolean",
        "empty",
        "zero-d",
    ),
    "random.RandomState.seed": (
        "array",
        "array-pair",
        "vector",
        "scalar-operand",
        "integer",
        "boolean",
        "empty",
        "zero-d",
    ),
    "random.Generator.spawn": ("scalar-operand",),
}

#: The 17 exact `grid/<op>::<pattern>` case ids that `_SEGFAULT_PATTERNS_BY_OP`
#: names, precomputed once so both `build()` and `undriven()` read the same
#: set instead of recomputing it.
SEGFAULT_EXCLUDED_CASE_IDS: frozenset[str] = frozenset(
    f"grid/{op}::{pattern}"
    for op, patterns in _SEGFAULT_PATTERNS_BY_OP.items()
    for pattern in patterns
)

#: UNINITIALIZED-MEMORY EXCLUSIONS - do not remove without re-measuring.
#:
#: `empty` and `empty_like` return arbitrary, uninitialized memory by
#: contract (see their docstrings in `src/flopscope/_array_ops.py`); their
#: *contents* cannot ever be meaningfully compared between two independent
#: allocations, in-process or remote, even in principle - unlike ordinary
#: nondeterminism (e.g. an unseeded random draw), no amount of seeding fixes
#: this, because numpy deliberately does not initialize the buffer. These
#: are the only two ops in the core registry with this contract (checked:
#: `grep -rn uninitial src` turns up nothing else).
#:
#: The registry search above found the *ops*; this maps each one to the
#: exact patterns that actually produce a "returned" outcome carrying such a
#: value (confirmed by running every `PATTERNS` entry for both ops through
#: the in-process backend directly). Every other pattern for these two ops
#: raises a normal, catchable, deterministic exception (mismatched arity or
#: an invalid `shape`/`dtype` argument) and stays in the corpus - only the
#: patterns whose value is genuinely incomparable are excluded.
_UNINITIALIZED_VALUE_PATTERNS_BY_OP: dict[str, tuple[str, ...]] = {
    # `fnp.empty(I)` (`I` is an int64 array) is accepted as a `shape` tuple;
    # `fnp.empty(E)` (`E` is an empty float32 array) is accepted as an empty
    # `shape` tuple, i.e. a 0-d output. Every other pattern passes a value
    # `numpy.empty`'s `shape` parameter rejects outright (a float array, a
    # bool array, an unexpected keyword, ...), which raises deterministically
    # and is left in the corpus.
    "empty": ("integer", "empty"),
    # `fnp.empty_like(x)` succeeds for any single positional array-like
    # argument, copying its shape (and, by default, its dtype). Every
    # pattern that instead supplies a second positional argument, an `axis`/
    # `keepdims` keyword, or no argument at all raises deterministically
    # (`empty_like` has no such parameters) and is left in the corpus.
    "empty_like": ("array", "vector", "dtype", "integer", "boolean", "empty", "zero-d"),
}

#: The 9 exact `grid/<op>::<pattern>` case ids that
#: `_UNINITIALIZED_VALUE_PATTERNS_BY_OP` names, precomputed once so both
#: `build()` and `undriven()` read the same set instead of recomputing it.
UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS: frozenset[str] = frozenset(
    f"grid/{op}::{pattern}"
    for op, patterns in _UNINITIALIZED_VALUE_PATTERNS_BY_OP.items()
    for pattern in patterns
)

#: Every case id `build()` refuses to drive, from any cause, unioned once so
#: `build()`'s filter and `undriven()`'s reporting always agree.
_ALL_EXCLUDED_CASE_IDS: frozenset[str] = (
    SEGFAULT_EXCLUDED_CASE_IDS | UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS
)


def undriven() -> dict[str, str]:
    """Grid cases no run can safely include, with the reason. Counted, not
    dropped.

    Three kinds of entry can appear here:

    * a case excluded because it segfaults the worker (see
      `SEGFAULT_EXCLUDED_CASE_IDS` above) - populated below from real
      measurement;
    * a case excluded because its return value is uninitialized memory (see
      `UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS` above) - also populated below
      from real measurement;
    * (populated from a first full run, currently none) an op whose every
      remaining pattern raises identically on BOTH backends, which is not
      being exercised, and saying so is more honest than reporting it as
      agreement.

    Either way, an entry here means "not in `build()`'s output, and here is
    why", so coverage counts stay honest instead of silently shrinking.
    """
    segfault_reason = (
        "SIGSEGV (exit 139) in the in-process backend: an unbound-method "
        "call with an array fixture standing in for `self`; uncatchable, "
        "so excluded from the corpus rather than fed to a worker"
    )
    uninitialized_value_reason = (
        "returns uninitialized memory by contract (`empty`/`empty_like`); its"
        " value can never be meaningfully compared between two independent"
        " allocations, so excluded from the corpus rather than compared"
    )
    reasons = dict.fromkeys(sorted(SEGFAULT_EXCLUDED_CASE_IDS), segfault_reason)
    reasons.update(
        dict.fromkeys(
            sorted(UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS), uninitialized_value_reason
        )
    )
    return reasons


def build() -> tuple[Case, ...]:
    return tuple(
        Case(
            id=f"grid/{op}::{pattern_name}",
            source=template.format(op=op),
            tags=frozenset({"src:grid", f"pattern:{pattern_name}"}),
        )
        for op in op_names()
        for pattern_name, template in PATTERNS
        if f"grid/{op}::{pattern_name}" not in _ALL_EXCLUDED_CASE_IDS
    )
