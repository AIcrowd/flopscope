"""Lock-in pins for the pointwise ``out=`` billing doctrine.

The billed dtype for a pointwise call is the widest participating buffer:
``max(compute width, store width)``. A wider ``out=`` store is a real
materialization (chargeable under the post-#150 copy rule), so it bills at
the wider rate; a narrower ``out=`` never discounts the compute loop that
actually runs. Both directions hold ``out=``-casting at exact parity with
the equivalent ``astype`` call -- see ``src/flopscope/_dtype_billing.py``
and the "Which dtype prices a call" section of ``docs/reference/cost-model.md``
for the full rationale. These tests pin the current, intended behavior; they
do not change it.
"""

import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def test_out_cast_equals_astype_parity():
    load_weights()
    a32 = np.ones(1000, dtype=np.float32)
    via_out = _billed(
        lambda: fnp.add(a32, 0.0, out=np.empty(1000, np.float64))  # pyright: ignore[reportArgumentType]
    )
    via_astype = _billed(lambda: fnp.astype(a32, np.float64))
    assert via_out == via_astype == 2000  # no cast arbitrage either direction


def test_narrowing_out_never_discounts():
    load_weights()
    a64 = np.ones(1000, dtype=np.float64)
    via_narrowing_out = _billed(
        lambda: fnp.add(a64, a64, out=np.empty(1000, np.float32))  # pyright: ignore[reportArgumentType]
    )
    via_no_out = _billed(lambda: fnp.add(a64, a64))
    assert via_narrowing_out == via_no_out == 2000
