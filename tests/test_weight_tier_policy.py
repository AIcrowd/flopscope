"""Weight tier policy (spec 2026-06-10, Section A).

flop_cost carries ALL shape-dependent cost; weight is only a per-element,
shape-independent tier multiplier. Legitimate tiers: view/free=0.0,
arithmetic=1.0, gather=4.0, transcendental=16.0 (8.0 retired). An algorithm
constant hiding in a weight (the old linalg 4.0) is a policy violation.
"""

from __future__ import annotations

import json
from importlib import resources

TIER_VALUES = {0.0, 1.0, 4.0, 16.0}

# 8.0 was a transcendental tier that no shipped op uses any more. Kept named
# (not merely absent) so a regression to it reports as a retired tier rather
# than as an unknown value.
RETIRED_TIER_VALUES = {8.0}

# Ops whose elementary operation is plain arithmetic: any non-{0,1} weight on
# these is an algorithm constant in the wrong layer. linalg.* is arithmetic
# by prefix (its 0.0 views are allowed by the {0,1} membership).
ARITHMETIC_PREFIXES = ("linalg.",)
ARITHMETIC_OPS = {
    "matmul",
    "dot",
    "vdot",
    "inner",
    "outer",
    "tensordot",
    "einsum",
    "kron",
    "sum",
    "prod",
    "mean",
    "average",
    "var",
    "std",
    "nanvar",
    "nanstd",
    "trapezoid",
    "trapz",
    "convolve",
    "correlate",
    "cov",
    "corrcoef",
    "cross",
    "polyval",
    "polyadd",
    "polysub",
    "polyder",
    "polyint",
    "polymul",
    "polydiv",
    "polyfit",
    "poly",
    "roots",
    "random.multivariate_normal",
}


def load_packaged_weights() -> dict[str, float]:
    data = resources.files("flopscope").joinpath("data/default_weights.json")
    return json.loads(data.read_text())["weights"]


def policy_violations(table: dict[str, float]) -> list[tuple[str, float, str]]:
    bad = []
    for op, w in table.items():
        if w in RETIRED_TIER_VALUES:
            bad.append((op, w, "weight uses a retired tier value"))
        elif w not in TIER_VALUES:
            bad.append((op, w, "weight is not a known tier value"))
        if (op.startswith(ARITHMETIC_PREFIXES) or op in ARITHMETIC_OPS) and w not in (
            0.0,
            1.0,
        ):
            bad.append((op, w, "arithmetic-tier op must have weight 0.0 or 1.0"))
    return bad


def test_packaged_table_satisfies_tier_policy():
    violations = policy_violations(load_packaged_weights())
    assert violations == [], f"weight tier policy violations: {violations}"


def test_guard_catches_algorithm_constant_weight():
    table = dict(load_packaged_weights())
    table["linalg.solve"] = 4.0
    assert any(op == "linalg.solve" for op, _, _ in policy_violations(table))


def test_guard_catches_novel_tier_value():
    table = dict(load_packaged_weights())
    table["var"] = 2.0
    assert any(op == "var" for op, _, _ in policy_violations(table))


def test_retired_tier_is_rejected():
    """8.0 was retired; no op may regress to it."""
    table = dict(load_packaged_weights())
    table["exp"] = 8.0
    violations = policy_violations(table)
    assert any(op == "exp" and "retired" in reason for op, _, reason in violations), (
        f"8.0 must be reported as a retired tier, got: {violations}"
    )


def test_no_shipped_op_uses_the_retired_tier():
    retired = {op: w for op, w in load_packaged_weights().items() if w == 8.0}
    assert retired == {}, f"ops on the retired 8.0 tier: {retired}"
