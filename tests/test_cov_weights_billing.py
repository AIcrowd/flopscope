"""``cov`` bills the weighted path numpy runs for ``fweights=``/``aweights=`` (#331).

With weights, numpy's ``cov`` checks them, replaces the plain mean by
``average(X, axis=1, weights=w)``, normalises an ``aweights=`` call by
``sum(w * aweights)`` and forms ``X * w`` before the Gram product. The bill
was the unweighted one. It now equals numpy's own steps written out in
flopscope ops; the O(1) scalar arithmetic on the weight sum stays unbilled,
as in the unweighted formula.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _cov_steps(X, fw=None, aw=None, ddof=1):
    """numpy's ``cov`` on a 2-D ``X`` of variables by samples, in numpy's order.

    Scalar arithmetic on the weight sum runs on Python floats.
    """
    if fw is None and aw is None:
        Xc = fnp.subtract(X, fnp.mean(X, axis=1)[:, None])
        return fnp.multiply(fnp.dot(Xc, Xc.T), 1.0 / (X.shape[1] - ddof))
    w = None
    if fw is not None:
        fnp.all(fnp.equal(fw, fnp.around(fw)))
        fnp.any(fnp.less(fw, 0))
        w = fw
    if aw is not None:
        fnp.any(fnp.less(aw, 0))
        w = aw if w is None else fnp.multiply(w, aw)
    assert w is not None
    w_sum = float(fnp.sum(w))
    avg = fnp.divide(fnp.sum(fnp.multiply(X, w), axis=1), w_sum)
    if ddof == 0:
        fact = w_sum
    elif aw is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * float(fnp.sum(fnp.multiply(w, aw))) / w_sum
    Xc = fnp.subtract(X, avg[:, None])
    return fnp.multiply(fnp.dot(Xc, fnp.multiply(Xc, w).T), 1.0 / fact)


def _weight_kwargs(kind, s, seed=0):
    """Fresh weight arrays, the same values for the same seed."""
    rng = np.random.default_rng(seed)
    aw = fnp.asarray(rng.random(s) + 0.1)
    fw = fnp.asarray(rng.integers(1, 4, s).astype(float))
    return {
        "aweights": {"aweights": aw},
        "fweights": {"fweights": fw},
        "both": {"fweights": fw, "aweights": aw},
    }[kind]


KINDS = ["aweights", "fweights", "both"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(1, 1000), (8, 1000)])
def test_issue_repro_weighted_cov_bills_numpys_steps(shape, kind):
    X = fnp.asarray(np.random.default_rng(1).standard_normal(shape))
    # Fresh weights per call: given both, numpy multiplies fweights in place.
    kw = _weight_kwargs(kind, shape[1])
    billed, got = _billed(lambda: fnp.cov(X, **kw))
    kw = _weight_kwargs(kind, shape[1])
    twin, ref = _billed(
        lambda: _cov_steps(X, fw=kw.get("fweights"), aw=kw.get("aweights"))
    )
    plain = {k: np.asarray(v) for k, v in _weight_kwargs(kind, shape[1]).items()}
    want = np.cov(
        np.asarray(X), fweights=plain.get("fweights"), aweights=plain.get("aweights")
    )
    assert np.array_equal(np.asarray(got), want)
    assert np.allclose(np.asarray(ref).squeeze(), want)
    assert billed == twin


CASES = [
    # x shape, rowvar, y shape, extra kwargs
    ((3, 50), True, None, {}),
    ((3, 50), True, None, {"ddof": 0}),
    ((3, 50), True, None, {"bias": True}),
    ((3, 50), True, None, {"ddof": 2}),
    ((50, 3), False, None, {}),
    ((50,), True, None, {}),
    ((2, 50), True, (3, 50), {}),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("shape", "rowvar", "y_shape", "extra"), CASES)
def test_weighted_cov_bills_numpys_steps(kind, shape, rowvar, y_shape, extra):
    rng = np.random.default_rng(2)
    xn = rng.standard_normal(shape)
    yn = None if y_shape is None else rng.standard_normal(y_shape)
    s = shape[-1] if rowvar else shape[0]
    x, y = fnp.asarray(xn), None if yn is None else fnp.asarray(yn)
    kw = _weight_kwargs(kind, s, seed=3)
    billed, got = _billed(lambda: fnp.cov(x, y, rowvar=rowvar, **extra, **kw))
    X = np.atleast_2d(xn if rowvar else xn.T)
    if yn is not None:
        X = np.concatenate([X, np.atleast_2d(yn if rowvar else yn.T)])
    ddof = extra.get("ddof", 0 if extra.get("bias") else 1)
    Xf, kw = fnp.asarray(X), _weight_kwargs(kind, s, seed=3)
    twin, _ = _billed(
        lambda: _cov_steps(Xf, fw=kw.get("fweights"), aw=kw.get("aweights"), ddof=ddof)
    )
    plain = {k: np.asarray(v) for k, v in _weight_kwargs(kind, s, seed=3).items()}
    want = np.cov(
        xn,
        yn,
        rowvar=rowvar,
        bias=bool(extra.get("bias", False)),
        ddof=extra.get("ddof"),
        fweights=plain.get("fweights"),
        aweights=plain.get("aweights"),
    )
    assert np.array_equal(np.asarray(got), want)
    assert billed == twin


@pytest.mark.parametrize("shape", [(1, 1000), (8, 1000), (3, 50)])
def test_unweighted_cov_is_unchanged(shape):
    X = fnp.asarray(np.random.default_rng(4).standard_normal(shape))
    billed, got = _billed(lambda: fnp.cov(X))
    twin, _ = _billed(lambda: _cov_steps(X))
    f, s = shape
    assert np.array_equal(np.asarray(got), np.cov(np.asarray(X)))
    assert billed == twin == 2 * f * f * s + 2 * f * s
