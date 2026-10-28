"""``corrcoef`` bills the clip numpy runs on its result (#340).

After its two divide passes numpy's ``corrcoef`` clips the result into
[-1, 1], ``np.clip(c.real, -1, 1, out=c.real)``, and clips ``c.imag`` as well
for a complex result. The bill priced cov, the square roots and the divides
and stopped there. It now equals numpy's own steps written out in flopscope
ops. With a single variable numpy returns ``c / c``, one divide.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._pointwise import _corrcoef_cost


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _corrcoef_steps(x, y=None, rowvar=True):
    """numpy's ``corrcoef`` in flopscope ops, in numpy's order."""
    c = fnp.cov(x, y, rowvar=rowvar)
    if c.ndim == 0:
        return fnp.divide(c, c)
    stddev = fnp.sqrt(fnp.real(fnp.diag(c)))
    c = fnp.divide(c, stddev[:, None])
    c = fnp.divide(c, stddev[None, :])
    return fnp.clip(c, -1, 1)


CASES = [
    # (x shape, y shape or None, rowvar)
    ((8, 1000), None, True),
    ((300, 500), None, True),
    ((1000, 8), None, False),
    ((3, 50), (4, 50), True),
    ((50,), (50,), True),
    ((1, 100), None, True),
    ((100,), None, True),
]


@pytest.mark.parametrize(("xshape", "yshape", "rowvar"), CASES)
def test_corrcoef_bills_numpys_steps(xshape, yshape, rowvar):
    rng = np.random.default_rng(0)
    xn = rng.standard_normal(xshape)
    yn = None if yshape is None else rng.standard_normal(yshape)
    x = fnp.asarray(xn)
    y = None if yn is None else fnp.asarray(yn)
    billed, out = _billed(lambda: fnp.corrcoef(x, y, rowvar=rowvar))
    steps, _ = _billed(lambda: _corrcoef_steps(x, y, rowvar))
    assert billed == steps
    np.testing.assert_allclose(np.asarray(out), np.corrcoef(xn, yn, rowvar=rowvar))


def test_the_clip_is_two_bounds_per_element():
    x = np.random.default_rng(1).standard_normal((20, 30))
    f = 20
    cov_part = 2 * f * f * 30 + 2 * f * 30
    assert _corrcoef_cost(x) == cov_part + 2 * f * f + f + 2 * f * f


def test_a_complex_result_clips_both_parts():
    rng = np.random.default_rng(2)
    xc = rng.standard_normal((20, 30)) + 1j * rng.standard_normal((20, 30))
    assert _corrcoef_cost(xc, complex_result=True) - _corrcoef_cost(xc) == 2 * 20 * 20


def test_the_wrapper_passes_a_complex_result_on(monkeypatch):
    import flopscope._pointwise as pointwise

    seen = []
    real_cost = pointwise._corrcoef_cost

    def spy(*args, **kwargs):
        seen.append(kwargs.get("complex_result"))
        return real_cost(*args, **kwargs)

    monkeypatch.setattr(pointwise, "_corrcoef_cost", spy)
    rng = np.random.default_rng(3)
    xr = rng.standard_normal((6, 40))
    xc = xr + 1j * rng.standard_normal((6, 40))
    with flops.BudgetContext(flop_budget=2**60, quiet=True):
        fnp.corrcoef(fnp.asarray(xc))
        fnp.corrcoef(fnp.asarray(xr))
        fnp.corrcoef(fnp.asarray(xr), dtype=np.complex128)
        fnp.corrcoef(fnp.asarray(xr), fnp.asarray(xc[:2]))
    assert seen == [True, False, True, True]
