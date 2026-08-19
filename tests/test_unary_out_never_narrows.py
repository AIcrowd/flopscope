"""A narrow ``out=`` must never lower a unary float-loop op's bill.

``docs/reference/cost-model.md`` publishes the invariant that ``out=`` alone
never shrinks the loop: the compute dtype numpy selects is a function of the
OPERANDS, and a destination buffer can only widen the price on top of it.

The compute-dtype override for ``_UNARY_FLOAT_LOOP_OPS`` and
``_UNARY_FLOAT64_MIN_OPS`` used to resolve the operand dtype TOGETHER with the
``out=`` dtype before mapping, so a narrow destination pulled the resolution
down and defeated the override -- ``angle(bool_array, out=float16_buffer)``
billed exactly half of ``angle(bool_array)``, and ``i0``/``sinc`` did the same
for bool/int8/uint8/int16/uint16. 22 (op, input dtype, out dtype) cells
across this family billed at exactly 0.5x.

These assertions are RATE-sensitive, so every measurement loads the real
weights first: ``tests/conftest.py``'s autouse ``reset_weights()`` installs
unit weights that erase dtype-rate differences and would make all of this
vacuous. Every operand AND every ``out=`` buffer is built outside the budget
context, so a measured delta is the op alone and not the allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._pointwise import _UNARY_FLOAT64_MIN_OPS, _UNARY_FLOAT_LOOP_OPS
from flopscope._weights import load_weights

#: modf/frexp take a 2-tuple ``out=`` and are priced through
#: ``multi_store_billing_dtypes``, which already drops a destination that is
#: not pricier than the buffer numpy would allocate for that slot. They are
#: covered by their own suite, not by this single-``out=`` grid.
_OPS = sorted((_UNARY_FLOAT_LOOP_OPS | _UNARY_FLOAT64_MIN_OPS) - {"modf", "frexp"})

_INPUT_DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]
_OUT_DTYPES = [np.float16, np.float32, np.float64]

_N = 200


def _billed(fn) -> int:
    load_weights()
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.mark.parametrize("op_name", _OPS)
def test_out_never_lowers_the_bill(op_name):
    """Across the whole family grid, ``out=`` is monotone non-decreasing."""
    fn = getattr(fnp, op_name)
    offenders = []
    for in_dtype in _INPUT_DTYPES:
        operand = fnp.array(np.ones(_N, dtype=in_dtype))
        try:
            baseline = _billed(lambda fn=fn, operand=operand: fn(operand))
        except Exception:  # noqa: BLE001 - op does not accept this dtype at all
            continue
        for out_dtype in _OUT_DTYPES:
            buffer = fnp.array(np.empty(_N, dtype=out_dtype))
            try:
                with_out = _billed(
                    lambda fn=fn, operand=operand, buffer=buffer: fn(
                        operand, out=buffer
                    )
                )
            except Exception:  # noqa: BLE001 - refused forms are not under test
                continue
            if with_out < baseline:
                offenders.append(
                    (
                        np.dtype(in_dtype).name,
                        np.dtype(out_dtype).name,
                        baseline,
                        with_out,
                    )
                )
    assert not offenders, f"{op_name}: narrow out= lowered the bill: {offenders}"


@pytest.mark.parametrize(
    "op_name, in_dtype",
    [
        ("angle", np.bool_),
        ("i0", np.bool_),
        ("i0", np.int8),
        ("i0", np.uint8),
        ("i0", np.int16),
        ("i0", np.uint16),
        ("sinc", np.bool_),
        ("sinc", np.int8),
        ("sinc", np.uint8),
        ("sinc", np.int16),
        ("sinc", np.uint16),
    ],
)
@pytest.mark.parametrize("out_dtype", [np.float16, np.float32])
def test_measured_half_price_cells_now_match_the_plain_call(
    op_name, in_dtype, out_dtype
):
    """The exact cells a client-side sweep measured at 0.5x, pinned at 1.0x.

    Each of these billed exactly half the plain call before the fix; naming a
    destination buys nothing here, so the price must be identical.
    """
    fn = getattr(fnp, op_name)
    operand = fnp.array(np.ones(_N, dtype=in_dtype))
    buffer = fnp.array(np.empty(_N, dtype=out_dtype))
    baseline = _billed(lambda: fn(operand))
    with_out = _billed(lambda: fn(operand, out=buffer))
    assert with_out == baseline


@pytest.mark.parametrize(
    "op_name, in_dtype",
    [("angle", np.bool_), ("i0", np.int8), ("sinc", np.uint16)],
)
def test_a_genuinely_wider_out_still_widens(op_name, in_dtype):
    """``out=`` must still be able to RAISE the bill, or the fix over-corrected.

    A complex destination is strictly wider than the float64 loop these
    operands resolve to, so it must cost strictly more than the plain call.
    """
    fn = getattr(fnp, op_name)
    operand = fnp.array(np.ones(_N, dtype=in_dtype))
    buffer = fnp.array(np.empty(_N, dtype=np.complex128))
    baseline = _billed(lambda: fn(operand))
    try:
        widened = _billed(lambda: fn(operand, out=buffer))
    except Exception:  # pragma: no cover - op refuses a complex destination
        pytest.skip(f"{op_name} does not accept a complex out=")
    assert widened > baseline
