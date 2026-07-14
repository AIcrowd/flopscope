"""Four-factor billing: charged = int(flop_cost * dtype_rate * complex_factor * weight)."""

import numpy as np
import pytest

import flopscope as f
from flopscope._weights import load_weights

# Inputs built outside any BudgetContext (input construction is billed).
_f32 = np.ones(10, dtype=np.float32)
_f64 = np.ones(10, dtype=np.float64)
_c128 = np.ones(10, dtype=np.complex128)


def _charge(op_name, flop_cost, dtypes, override=None) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            op_name,
            flop_cost=flop_cost,
            subscripts=None,
            shapes=(),
            dtypes=dtypes,
            complex_factor_override=override,
        ):
            pass
        return b.flops_used


def test_unit_mode_real_dtypes_bill_flop_cost():
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 10


def test_unit_mode_complex_factor_still_applies():
    # complex_factor is math, not policy: active even under unit rates/weights
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 60
    assert _charge("add", 10, (np.dtype("complex128"),)) == 20


def test_production_rates_compose():
    load_weights()
    assert _charge("multiply", 10, (np.dtype("float32"),)) == 10
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 20
    assert _charge("multiply", 10, (np.dtype("complex64"),)) == 60
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 120
    assert _charge("add", 10, (np.dtype("complex128"),)) == 40


def test_override_bypasses_registry_factor():
    assert (
        _charge("einsum", 960, (np.dtype("complex128"),), override=3968 / 960) == 3968
    )


def test_dtype_neutral_and_unmigrated():
    assert _charge("einsum_path", 1, ()) == 1  # declared neutral
    assert _charge("einsum_path", 1, None) == 1  # unmigrated site (until Task 9)


def test_resolved_dtype_recorded():
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            "multiply",
            flop_cost=1,
            subscripts=None,
            shapes=(),
            dtypes=(np.dtype("float32"), np.dtype("float64")),
        ):
            pass
        assert b._op_log[-1].resolved_dtype == "float64"


def test_unsupported_dtype_fails_closed_before_charging():
    load_weights()
    from flopscope.errors import UnsupportedDtypeError

    # float128 is unavailable on this platform; object exercises the same fail-closed path
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with pytest.raises(UnsupportedDtypeError):
            with b.deduct(
                "multiply",
                flop_cost=1,
                subscripts=None,
                shapes=(),
                dtypes=(np.dtype("object"),),
            ):
                pass
        assert b.flops_used == 0
