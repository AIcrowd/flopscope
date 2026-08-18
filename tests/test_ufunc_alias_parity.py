"""NumPy 2.x ufunc aliases must bill identically to their canonical twins.

``np.acos`` IS ``np.arccos`` (the same ufunc object); billing the alias at the
1.0 default while the canonical bills 16.0 is a bit-identical substitution
exploit (a 16x discount for typing ``acos`` instead of ``arccos``).

conftest resets weights to 1.0 per test, so these load the packaged table.
"""

from __future__ import annotations

import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import get_weight, load_weights

# alias -> canonical (same ufunc object under NumPy 2.x)
ALIAS_CANONICAL = [
    ("acos", "arccos"),
    ("acosh", "arccosh"),
    ("asin", "arcsin"),
    ("asinh", "arcsinh"),
    ("atan", "arctan"),
    ("atanh", "arctanh"),
    ("atan2", "arctan2"),
    ("pow", "power"),
    (
        "divmod",
        "floor_divide",
    ),  # divmod does >= floor_divide work; 16.0 is a conservative floor
]


def _cost(fn, *args) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn(*args)
        return b.flops_used


def test_ufunc_aliases_resolve_to_canonical_weight():
    load_weights()
    for alias, canon in ALIAS_CANONICAL:
        assert get_weight(alias) == get_weight(canon) == 16.0, (
            f"{alias} weight {get_weight(alias)} != {canon} {get_weight(canon)}"
        )


def test_ufunc_aliases_bill_identically_to_canonical():
    load_weights()
    v = fnp.asarray(np.random.rand(100))  # cost is shape-based; values irrelevant
    # v is float64 (numpy's default), so dtype_rate 2.0 applies on top of the
    # 16.0 transcendental weight: 100 * 2.0 * 16.0 = 3200. The parity
    # invariant (ca == cc) is what matters -- the absolute just tracks the
    # dtype-aware billing migration.
    #
    # divmod is the one entry that is NOT a true 1:1 alias: unlike
    # acos/arccos (literally the same ufunc object), np.divmod is a distinct
    # two-output ufunc (nin=2, nout=2) from np.floor_divide (nin=2, nout=1).
    # _UFUNC_ALIAS_RENAMES only borrows floor_divide's WEIGHT for divmod (a
    # conservative floor); divmod's flop_cost is nout=2 * numel(output), so
    # it must bill exactly 2x its canonical twin's flop_cost, not the same
    # value -- see tests/test_multi_output_ufunc_cost.py.
    with np.errstate(all="ignore"):  # arccosh(<1) etc. NaN harmlessly
        for alias, canon in ALIAS_CANONICAL:
            fa, fc = getattr(fnp, alias), getattr(fnp, canon)
            if alias in ("atan2", "pow", "divmod"):
                ca, cc = _cost(fa, v, v), _cost(fc, v, v)
            else:
                ca, cc = _cost(fa, v), _cost(fc, v)
            if alias == "divmod":
                assert ca == 2 * cc == 6400, f"divmod={ca} vs floor_divide={cc}"
            else:
                assert ca == cc == 3200, f"{alias}={ca} vs {canon}={cc} (want 3200)"
