"""``fnp.matmul`` accepts a destination, and the contraction helper returns it.

Two things are pinned here.

The feature: ``matmul`` takes ``out=`` like every other contraction that can.
``vecdot``/``matvec``/``vecmat`` already did, because they forward ``**kwargs``
into the shared helper; ``matmul`` did not, which made the surface arbitrary.

The defect the feature exposes: the helper set ``result = out`` and then, when
an output symmetry was inferred, returned a *new* ``SymmetricTensor`` over the
destination's memory instead of the destination. NumPy and ``fnp.einsum`` both
return ``out`` itself, and a caller who writes ``r = matmul(a, b, out=arena)``
and then also uses ``arena`` should not be holding two objects of different
types over one buffer.

Probes here run in both directions on purpose. A rate probe built only from a
neutral store (an object destination, which ``store_billing_dtypes`` is
*designed* to ignore) cannot tell a correct implementation from one that never
folds the destination at all -- that blindness is exactly how a 2x fft
under-bill survived an earlier sweep. So a wide destination must raise the
bill, and a neutral one must leave it alone. Content is asserted separately
from cost for the same reason: a destination that is billed correctly and
never written is invisible to any cost assertion.
"""

from typing import Any

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 6


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def _pair(dtype: Any = np.float64, n: int = N):
    rng = np.random.default_rng(0)
    return rng.random((n, n)).astype(dtype), rng.random((n, n)).astype(dtype)


def _symmetric(n=N):
    a = np.random.default_rng(2).random((n, n))
    return a + a.T


# --- the feature ---------------------------------------------------------


def test_matmul_accepts_a_destination_and_writes_it():
    load_weights()
    a, b = _pair()
    dest = np.empty((N, N))
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        r = fnp.matmul(a, b, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert r is dest
    np.testing.assert_allclose(dest, a @ b)


def test_matmul_declares_out_rather_than_swallowing_it_in_kwargs():
    """A declared parameter is what enrols matmul in the derived out= sweep."""
    import inspect

    assert "out" in inspect.signature(fnp.matmul).parameters


def test_dot_and_inner_still_refuse_a_destination():
    """numpy's dot demands exact dtype and C-contiguity and inner has no out=,
    so neither inherits the destination contract."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        for op in (fnp.dot, fnp.inner):
            with pytest.raises(TypeError):
                op(a, b, out=np.empty((N, N)))  # pyright: ignore[reportArgumentType, reportCallIssue]


# --- return identity: the defect the feature exposes ----------------------


def test_matmul_returns_the_destination_when_the_output_is_symmetric():
    """``matmul(S, S)`` infers a symmetric output, which is the branch that used
    to mint a new SymmetricTensor over the caller's buffer."""
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        s = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        dest = np.empty((N, N))
        r = fnp.matmul(s, s, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert r is dest
        np.testing.assert_allclose(dest, np.asarray(s) @ np.asarray(s))


def test_matmul_returns_a_tagged_destination_itself():
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        s = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        dest = fnp.zeros((N, N))
        assert fnp.matmul(s, s, out=dest) is dest


def test_sibling_contractions_also_return_their_destination():
    """The fix lives in the shared helper, so vecdot/matvec inherit it."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        d1 = np.empty(N)
        assert fnp.vecdot(a, b, out=d1) is d1
        d2 = np.empty(N)
        assert fnp.matvec(a, np.ones(N), out=d2) is d2


# --- billing: both directions --------------------------------------------


def test_a_same_dtype_destination_does_not_change_the_bill():
    load_weights()
    a, b = _pair(np.float32)
    bare = _billed(lambda: fnp.matmul(a, b))
    same = _billed(lambda: fnp.matmul(a, b, out=np.empty((N, N), np.float32)))  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert same == bare


def test_a_wider_destination_raises_the_bill():
    """The upward direction. Omitting it is how a destination that is never
    folded into the rate passes for correct."""
    load_weights()
    a, b = _pair(np.float32)
    bare = _billed(lambda: fnp.matmul(a, b))
    wide = _billed(lambda: fnp.matmul(a, b, out=np.empty((N, N), np.float64)))  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert wide > bare


def test_a_neutral_destination_does_not_lower_the_bill():
    """The downward direction: a non-numeric store must not launder the rate."""
    load_weights()
    a, b = _pair(np.float64)
    bare = _billed(lambda: fnp.matmul(a, b))
    neutral = _billed(lambda: fnp.matmul(a, b, out=np.empty((N, N), dtype=object)))  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert neutral == bare


# --- inherited from _normalize_out ---------------------------------------


def test_a_one_tuple_destination_bills_and_returns_like_a_bare_one():
    load_weights()
    a, b = _pair(np.float32)
    wide = np.empty((N, N), np.float64)
    bare_bill = _billed(lambda: fnp.matmul(a, b, out=wide))  # pyright: ignore[reportArgumentType, reportCallIssue]
    tuple_bill = _billed(lambda: fnp.matmul(a, b, out=(wide,)))  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert tuple_bill == bare_bill
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        returned = fnp.matmul(a, b, out=(wide,))  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert returned is wide


def test_a_bad_container_is_refused_before_anything_is_billed():
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as bud:
        before = bud.flops_used
        with pytest.raises(TypeError):
            fnp.matmul(a, b, out=[np.empty((N, N))])  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert bud.flops_used == before


# --- symmetry: matmul must behave exactly like its siblings ---------------


def test_a_tag_on_the_destination_is_voided_not_raised():
    """An inferred tag is dropped quietly across the whole family now."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = fnp.zeros((N, N))
        fnp.matmul(a, b, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert getattr(dest, "symmetry", None) is None
        np.testing.assert_allclose(np.asarray(dest), a @ b)


# --- aligning the contraction family with the pointwise gate --------------


def test_a_validated_tag_on_the_destination_is_an_error():
    """A tag the caller paid ``as_symmetric`` to validate is a claim about that
    buffer. Contradicting it is a mistake worth surfacing -- which is what the
    pointwise factories already do, via the same gate."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        with pytest.raises((ValueError, f.SymmetryError)):
            fnp.matmul(a, b, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]


def test_an_inferred_tag_on_the_destination_is_not_an_error():
    """The converse: a tag merely inferred from a constant fill is dropped
    quietly, so an ordinary scratch arena keeps working."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = fnp.zeros((N, N))
        fnp.matmul(a, b, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert getattr(dest, "symmetry", None) is None


def test_the_siblings_gain_the_same_gate():
    """The gate lives in the shared helper, so vecdot inherits it. Batched
    operands so the result is 2-D and a two-axis tag is meaningful."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        with pytest.raises((ValueError, f.SymmetryError)):
            fnp.vecdot(a[:, None, :], b[None, :, :], out=dest)


def test_einsum_stops_raising_on_a_merely_inferred_tag():
    """einsum lifted the destination's own tag into a requirement on the
    result, so a scratch arena from fnp.zeros made a legal contraction raise.
    A validated tag must still raise."""
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        arena = fnp.zeros((N, N))
        fnp.einsum("ij,jk->ik", a, b, out=arena)
        np.testing.assert_allclose(np.asarray(arena), a @ b)
        assert getattr(arena, "symmetry", None) is None


def test_einsum_still_raises_on_a_validated_tag():
    load_weights()
    a, b = _pair()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        with pytest.raises((ValueError, f.SymmetryError)):
            fnp.einsum("ij,jk->ik", a, b, out=dest)


# --- the destination must actually be written ----------------------------


def test_a_destination_numpy_ignored_is_still_written(monkeypatch):
    """numpy has shipped functions that silently hardcode out=None (hfft,
    ifft2, irfft2 on 2.0-2.4). The contraction path returns ``out``, so if that
    ever happened here the caller would receive an untouched buffer AS the
    result, at full price. Simulate it rather than wait for a numpy that does."""
    load_weights()
    a, b = _pair()
    real = np.matmul

    def ignores_out(x, y, **kwargs):
        kwargs.pop("out", None)  # pretend numpy dropped the destination
        return real(x, y)

    monkeypatch.setattr(np, "matmul", ignores_out)
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        dest = np.zeros((N, N))
        r = fnp.matmul(a, b, out=dest)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert r is dest
        np.testing.assert_allclose(dest, a @ b)
