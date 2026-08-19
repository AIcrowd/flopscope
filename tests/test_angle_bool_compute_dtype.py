"""``angle`` must bill the dtype numpy actually computes in, including bool.

numpy computes ``angle`` in float64 for bool input and preserves float width
otherwise. Billing bool at the float16 rate charged half price.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights


def _billed(fn) -> int:
    load_weights()
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


def test_angle_bool_bills_like_int32_not_like_float16():
    """bool and int32 both compute in float64, so they must bill identically."""
    n = 1000
    b = fnp.array(np.ones(n, dtype=bool))
    i = fnp.array(np.ones(n, dtype=np.int32))
    assert _billed(lambda: fnp.angle(b)) == _billed(lambda: fnp.angle(i))


def test_angle_bool_bills_above_float32():
    """float64 computation costs strictly more than a genuine float32 loop."""
    n = 1000
    b = fnp.array(np.ones(n, dtype=bool))
    f32 = fnp.array(np.ones(n, dtype=np.float32))
    assert _billed(lambda: fnp.angle(b)) > _billed(lambda: fnp.angle(f32))


@pytest.mark.parametrize(
    "dtype, twin",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ],
)
def test_angle_float_and_complex_widths_are_preserved(dtype, twin):
    """The fix must not move float/complex inputs: only bool floors at f64."""
    n = 1000
    x = fnp.array(np.ones(n, dtype=dtype))
    y = fnp.array(np.ones(n, dtype=twin))
    assert _billed(lambda: fnp.angle(x)) == _billed(lambda: fnp.angle(y))


@pytest.mark.parametrize(
    "dtype, expected_rate_of",
    [
        (np.int8, np.float16),
        (np.uint8, np.float16),
        (np.int16, np.float32),
        (np.uint16, np.float32),
    ],
)
def test_angle_narrow_integer_widths_keep_the_size_mapped_loop(dtype, expected_rate_of):
    """Only bool is the NEP-50 anomaly; every other integer width still bills
    the same-size float loop angle actually computes in (matches sin/cos/etc.),
    so narrowing this fix to bool alone must not move int8/16 or uint8/16."""
    n = 1000
    x = fnp.array(np.ones(n, dtype=dtype))
    y = fnp.array(np.ones(n, dtype=expected_rate_of))
    assert _billed(lambda: fnp.angle(x)) == _billed(lambda: fnp.angle(y))


def test_angle_value_is_unchanged():
    """A billing change must not alter the returned values."""
    base = np.linspace(-2.0, 2.0, 64)
    x = fnp.array(base)
    with flops.budget(10**15, quiet=True):
        got = np.asarray(fnp.angle(x))
    assert np.allclose(got, np.angle(base))
