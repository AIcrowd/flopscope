"""A ufunc producing k outputs must bill for k outputs, not one.

``divmod`` writes a quotient array AND a remainder array; ``modf`` writes a
fractional-part array AND an integral-part array; ``frexp`` writes a mantissa
array AND an exponent array. Under the cost model's own "every byte written
is metered" principle (see docs/reference/cost-model.md), a call that writes
``nout`` full output buffers must bill ``nout`` times what a same-shape
single-output call bills -- billing only the first output priced these ops
at exactly half (divmod) or a flat fraction (modf/frexp vs. a comparable
single-output unary) of their honest cost.
"""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

N = 100_000


def billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


def test_divmod_bills_both_outputs():
    x = fnp.array(np.full(N, 7.0))
    y = fnp.array(np.full(N, 3.0))
    separate = billed(lambda: fnp.floor_divide(x, y)) + billed(lambda: fnp.mod(x, y))
    assert billed(lambda: fnp.divmod(x, y)) == separate


@pytest.mark.parametrize("op", ["modf", "frexp"])
def test_two_output_unaries_bill_both(op):
    x = fnp.array(np.linspace(1.0, 9.0, N))
    single = billed(lambda: fnp.sqrt(x))
    assert billed(lambda: getattr(fnp, op)(x)) == 2 * single


def test_divmod_result_still_correct():
    x = fnp.array(np.full(1000, 7.0))
    y = fnp.array(np.full(1000, 3.0))
    with flops.budget(10**15, quiet=True):
        q, r = fnp.divmod(x, y)
    assert np.allclose(np.asarray(q), 2.0)
    assert np.allclose(np.asarray(r), 1.0)


# ---------------------------------------------------------------------------
# out= must not double-count on top of the nout fix. The dtype-rate axis
# (widest-participating-buffer doctrine, see tests/test_multi_output_out_
# billing.py) is orthogonal to the cell-count axis this file exercises: a
# natural out= destination must still be price-neutral against the bare call,
# even now that the bare call itself charges nout cells instead of one.
# ---------------------------------------------------------------------------


def test_divmod_out_matches_bare_with_natural_destinations():
    x = fnp.array(np.full(1000, 7.0))
    y = fnp.array(np.full(1000, 3.0))
    o1 = fnp.array(np.zeros(1000))
    o2 = fnp.array(np.zeros(1000))
    bare = billed(lambda: fnp.divmod(x, y))
    with_out = billed(lambda: fnp.divmod(x, y, out=(o1, o2)))
    assert with_out == bare


@pytest.mark.parametrize("op", ["modf", "frexp"])
def test_unary_multi_out_matches_bare_with_natural_destinations(op):
    x = fnp.array(np.linspace(1.0, 9.0, 1000))
    np_func = getattr(np, op)
    natural = tuple(np.empty(1000, r.dtype) for r in np_func(np.asarray(x)))
    fs_func = getattr(fnp, op)
    bare = billed(lambda: fs_func(x))
    with_out = billed(lambda: fs_func(x, out=natural))
    assert with_out == bare


# ---------------------------------------------------------------------------
# Every numpy ufunc with nout > 1 that flopscope wraps must be covered above.
# A hand-picked list of three is how a fourth (added later, or missed now)
# goes unnoticed -- enumerate the real numpy ufunc surface instead.
# ---------------------------------------------------------------------------


def test_multi_output_ufunc_surface_is_exactly_divmod_modf_frexp():
    found = sorted(
        name
        for name in dir(fnp)
        if not name.startswith("_")
        and isinstance(getattr(np, name, None), np.ufunc)
        and getattr(np, name).nout > 1
    )
    assert found == ["divmod", "frexp", "modf"]
