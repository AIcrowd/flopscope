"""``isclose``/``allclose`` accept and bill array tolerances (#278).

numpy broadcasts ``rtol=`` and ``atol=`` against ``a`` and ``b`` inside the
tolerance core (``|a - b| <= atol + rtol * |b|``), so an array tolerance can
enlarge the result. A tracked tolerance reached numpy still wrapped and raised
the internal "reached numpy.isclose from inside a flopscope wrapper" error; a
plain-numpy one ran but was billed on the shape of ``a`` and ``b`` alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weights for one test (conftest resets them)."""
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _isclose_cost(numel):
    return 6 * numel


def _allclose_cost(numel):
    return 7 * numel - 1


def test_issue_repro_tracked_tolerances():
    X = np.random.default_rng(0).standard_normal((512, 1024))
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        one, tracked = fnp.asarray(1.0), fnp.asarray(X)
        before = int(ctx.flops_used)
        close = fnp.isclose(one, one, rtol=tracked)
        isclose_cost = int(ctx.flops_used) - before
        before = int(ctx.flops_used)
        all_close = fnp.allclose(one, one, rtol=tracked)
        allclose_cost = int(ctx.flops_used) - before
    assert np.array_equal(np.asarray(close), np.isclose(1.0, 1.0, rtol=X))
    assert all_close == np.allclose(1.0, 1.0, rtol=X)
    assert isclose_cost == _isclose_cost(X.size)
    assert allclose_cost == _allclose_cost(X.size)


CASES = [
    # a shape, b shape, rtol shape, atol shape, broadcast shape
    ((), (), (512, 1024), None, (512, 1024)),
    ((), (), None, (300,), (300,)),
    ((1, 4), (1, 4), (3, 1), None, (3, 4)),
    ((4,), (4,), (2, 1), (3, 1, 1), (3, 2, 4)),
    ((5, 5), (5, 5), (), (), (5, 5)),
]


def _operands(a_shape, b_shape, rtol_shape, atol_shape, tracked):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(a_shape)
    b = a + 1e-6 * rng.standard_normal(b_shape)
    kwargs = {}
    if rtol_shape is not None:
        kwargs["rtol"] = rng.uniform(1e-7, 1e-4, rtol_shape)
    if atol_shape is not None:
        kwargs["atol"] = rng.uniform(1e-9, 1e-5, atol_shape)
    numpy_kwargs = dict(kwargs)
    if tracked:
        kwargs = {k: fnp.asarray(v) for k, v in kwargs.items()}
    return a, b, kwargs, numpy_kwargs


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "rtol_shape", "atol_shape", "shape"), CASES
)
def test_isclose_bills_the_broadcast_shape(
    a_shape, b_shape, rtol_shape, atol_shape, shape, tracked
):
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        a, b, kwargs, numpy_kwargs = _operands(
            a_shape, b_shape, rtol_shape, atol_shape, tracked
        )
        before = int(ctx.flops_used)
        out = fnp.isclose(a, b, **kwargs)
        cost = int(ctx.flops_used) - before
    expected = np.isclose(a, b, **numpy_kwargs)
    assert np.asarray(out).shape == shape == expected.shape
    assert np.array_equal(np.asarray(out), expected)
    assert cost == _isclose_cost(int(np.prod(shape)))


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "rtol_shape", "atol_shape", "shape"), CASES
)
def test_allclose_bills_the_broadcast_shape(
    a_shape, b_shape, rtol_shape, atol_shape, shape, tracked
):
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        a, b, kwargs, numpy_kwargs = _operands(
            a_shape, b_shape, rtol_shape, atol_shape, tracked
        )
        before = int(ctx.flops_used)
        out = fnp.allclose(a, b, **kwargs)
        cost = int(ctx.flops_used) - before
    assert out == np.allclose(a, b, **numpy_kwargs)
    assert cost == _allclose_cost(int(np.prod(shape)))


def test_scalar_tolerances_bill_as_before():
    a = np.linspace(0.0, 1.0, 50)
    cost, _ = _billed(lambda: fnp.isclose(a, a, rtol=1e-3, atol=1e-6))
    assert cost == _isclose_cost(50)
    cost, _ = _billed(lambda: fnp.allclose(a, a, rtol=1e-3, atol=1e-6, equal_nan=True))
    assert cost == _allclose_cost(50)


def _symmetric(n, seed):
    t = np.random.default_rng(seed).standard_normal((n, n))
    return t + t.T


def test_dense_array_tolerance_drops_the_symmetry_discount():
    n = 6
    data = _symmetric(n, 2)
    rtol = np.random.default_rng(3).uniform(1e-7, 1e-4, (n, n))
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        S = flops.as_symmetric(data, symmetry=(0, 1))
        before = int(ctx.flops_used)
        out = fnp.isclose(S, S, rtol=rtol)
        cost = int(ctx.flops_used) - before
    assert cost == _isclose_cost(n * n)
    assert getattr(out, "symmetry", None) is None
    assert np.array_equal(np.asarray(out), np.isclose(data, data, rtol=rtol))


def test_dense_array_tolerance_never_tags_an_asymmetric_result():
    # Symmetric operands, but a tolerance that is loose at (0, 1) only: the
    # result differs at (0, 1) and (1, 0), so it must not carry their tag.
    n = 4
    base = np.arange(1.0, n * n + 1).reshape(n, n)
    base = base + base.T
    rtol = np.full((n, n), 1e-9)
    rtol[0, 1] = 1.0
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        S = flops.as_symmetric(base, symmetry=(0, 1))
        T = flops.as_symmetric(base + 1e-3, symmetry=(0, 1))
        before = int(ctx.flops_used)
        out = fnp.isclose(S, T, rtol=rtol)
        cost = int(ctx.flops_used) - before
    raw = np.asarray(out)
    assert raw[0, 1] and not raw[1, 0]
    assert getattr(out, "symmetry", None) is None
    assert cost == _isclose_cost(n * n)


def test_symmetric_array_tolerance_keeps_the_symmetry():
    n = 6
    data = _symmetric(n, 4)
    tol = np.abs(_symmetric(n, 5)) * 1e-5
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        S = flops.as_symmetric(data, symmetry=(0, 1))
        R = flops.as_symmetric(tol, symmetry=(0, 1))
        before = int(ctx.flops_used)
        out = fnp.isclose(S, S, rtol=R)
        cost = int(ctx.flops_used) - before
        before = int(ctx.flops_used)
        fnp.isclose(S, S)
        scalar_tol = int(ctx.flops_used) - before
    assert cost == scalar_tol < _isclose_cost(n * n)
    assert getattr(out, "symmetry", None) is not None


@pytest.mark.parametrize("op", ["isclose", "allclose"])
def test_wider_array_tolerance_sets_the_billed_width(production_weights, op):
    a32 = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    a64 = a32.astype(np.float64)
    f = getattr(fnp, op)
    narrow, _ = _billed(lambda: f(a32, a32, rtol=1e-3))
    wide_tol, _ = _billed(lambda: f(a32, a32, rtol=np.full(64, 1e-3)))
    wide, _ = _billed(lambda: f(a64, a64))
    assert wide_tol == wide == 2 * narrow
