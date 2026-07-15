from __future__ import annotations

import inspect
from pathlib import Path

import flopscope as f
import flopscope._budget as _budget
from flopscope._weights import load_weights, reset_weights

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "src" / "flopscope"
_BILLING = {"_budget.py", "_dtype_billing.py", "_accounting.py", "_flops.py"}

def _run(call) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        call()
    return b.flops_used

def measure_raw(call) -> int:
    reset_weights()          # unit mode: weight=1, dtype_rate=1
    return _run(call)

def measure_billed(call) -> int | str:
    load_weights()
    try:
        return _run(call)
    finally:
        reset_weights()

def _rel(fname: str) -> str | None:
    p = Path(fname).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return None

def capture_cost_site(call) -> tuple[str, int] | None:
    """Hook deduct/deduct_after; record the innermost flopscope-source frame
    that isn't billing machinery — the op's cost-computation site."""
    hits: list[tuple[str, int]] = []
    orig_d = _budget.BudgetContext.deduct
    orig_da = getattr(_budget.BudgetContext, "deduct_after", None)

    def _record():
        for fr in inspect.stack()[2:]:                 # skip _record + patched
            name = Path(fr.filename).name
            resolved = Path(fr.filename).resolve()
            if str(_SRC) in str(resolved) and name not in _BILLING:
                rel = _rel(fr.filename)
                if rel:
                    hits.append((rel, fr.lineno))
                    return

    def patched_d(self, *a, **k):
        if not hits:
            _record()
        return orig_d(self, *a, **k)

    _budget.BudgetContext.deduct = patched_d
    if orig_da:

        def patched_da(self, *a, **k):
            if not hits:
                _record()
            return orig_da(self, *a, **k)

        _budget.BudgetContext.deduct_after = patched_da
    try:
        reset_weights()
        with f.BudgetContext(flop_budget=10**18, quiet=True):
            call()
    finally:
        _budget.BudgetContext.deduct = orig_d
        if orig_da:
            _budget.BudgetContext.deduct_after = orig_da
    return hits[0] if hits else None


import numpy as np

_DTYPES = {
    "int16": np.int16,
    "float32": np.float32,
    "float64": np.float64,
    "complex128": np.complex128,
}


def measure_op(make, scalable: bool) -> dict:
    """make(dtype, scale=1) -> zero-arg callable running the op on that input."""
    raw = measure_raw(make(np.float32, 1))
    raw2x = measure_raw(make(np.float32, 2)) if scalable else ""
    site = capture_cost_site(make(np.float32, 1))
    billed = {}
    for name, dt in _DTYPES.items():
        try:
            billed[name] = measure_billed(make(dt, 1))
        except Exception:
            billed[name] = "raises"
    return {
        "raw_flop_cost": raw,
        "raw_flop_cost_2x": raw2x,
        "billed": billed,
        "cost_impl": site,
    }
