"""A destination or mask larger than the operands bills the broadcast write (#275).

numpy broadcasts a ufunc's operands against ``out=`` (and ``where=``) and
evaluates the loop over every output element; a gufunc such as ``matmul``
repeats its whole contraction for every batch element of a larger ``out``.
Each call here must bill exactly what the same op costs on the operands
materialised at the destination's shape. A destination of the operands' own
shape bills as before.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

M, N = 512, 2048

requires_matvec = pytest.mark.skipif(
    not hasattr(np, "matvec"), reason="requires numpy >= 2.2"
)


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def _full(a, shape):
    """``a`` broadcast to ``shape`` and copied, built outside any budget."""
    return fnp.asarray(np.broadcast_to(np.asarray(a), shape).copy())


def _zeros(shape):
    return fnp.asarray(np.zeros(shape))


def _operands():
    rng = np.random.default_rng(0)
    x = fnp.asarray(rng.standard_normal(N))
    return x, _full(x, (M, N))


def test_unary_out_bills_the_destination_issue_shape():
    x, x_full = _operands()
    billed, got = _bill(lambda: fnp.exp(x, out=_zeros((M, N))))
    want, ref = _bill(lambda: fnp.exp(x_full))
    assert billed == want == M * _bill(lambda: fnp.exp(x))[0]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_binary_out_bills_the_destination_issue_shape():
    x, x_full = _operands()
    billed, got = _bill(lambda: fnp.multiply(x, x, out=_zeros((M, N))))
    want, ref = _bill(lambda: fnp.multiply(x_full, x_full))
    assert billed == want == M * _bill(lambda: fnp.multiply(x, x))[0]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_where_mask_bills_the_broadcast_output():
    x, x_full = _operands()
    mask = fnp.asarray(np.random.default_rng(1).random((M, N)) > 0.5)
    want, _ = _bill(lambda: fnp.exp(x_full))
    with_out, _ = _bill(lambda: fnp.exp(x, where=mask, out=_zeros((M, N))))
    assert with_out == want
    # Without out=, numpy allocates the mask's broadcast shape.
    bare, got = _bill(lambda: fnp.exp(x, where=mask))
    assert got.shape == (M, N)
    assert bare == want


def test_matmul_batched_out_bills_every_batch_issue_shape():
    K, B = 128, 256
    X = fnp.asarray(np.random.default_rng(2).standard_normal((K, K)))
    billed, got = _bill(lambda: fnp.matmul(X, X, out=_zeros((B, K, K))))
    want, ref = _bill(lambda: fnp.matmul(_full(X, (B, K, K)), X))
    assert billed == want == B * _bill(lambda: fnp.matmul(X, X))[0]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_vecdot_batched_out_bills_every_batch():
    v = fnp.asarray(np.random.default_rng(3).standard_normal(64))
    billed, _ = _bill(lambda: fnp.vecdot(v, v, out=_zeros(8)))
    assert billed == 8 * _bill(lambda: fnp.vecdot(v, v))[0]


@pytest.mark.parametrize("kwargs", [{}, {"axis": -1}], ids=["default", "axis=-1"])
def test_vecdot_keepdims_out_bills_every_batch(kwargs):
    # keepdims=True adds a size-1 axis the pricing subscript has no label
    # for; the batch axes out= adds in front of it are still computed.
    B = 32
    rng = np.random.default_rng(4)
    P = fnp.asarray(rng.standard_normal((8, 64)))
    Q = fnp.asarray(rng.standard_normal((8, 64)))
    billed, got = _bill(
        lambda: fnp.vecdot(P, Q, keepdims=True, out=_zeros((B, 8, 1)), **kwargs)
    )
    P_full, Q_full = _full(P, (B, 8, 64)), _full(Q, (B, 8, 64))
    want, ref = _bill(lambda: fnp.vecdot(P_full, Q_full, keepdims=True, **kwargs))
    assert billed == want == B * _bill(lambda: fnp.vecdot(P, Q, keepdims=True))[0]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_vecdot_kept_axis_is_read_against_the_destination_rank():
    # numpy reads the kept axis's number against out's own rank: axis=-2
    # keeps it second from the end of a 4-D out, axis=1 at position 1 of a
    # 3-D one. A destination whose kept axis is not size 1 is refused and
    # costs only the call itself.
    B = 32
    rng = np.random.default_rng(5)
    A = fnp.asarray(rng.standard_normal((4, 64, 8)))
    C = fnp.asarray(rng.standard_normal((4, 64, 8)))
    billed, got = _bill(
        lambda: fnp.vecdot(A, C, axis=-2, keepdims=True, out=_zeros((B, 4, 1, 8)))
    )
    A_full, C_full = _full(A, (B, 4, 64, 8)), _full(C, (B, 4, 64, 8))
    want, ref = _bill(lambda: fnp.vecdot(A_full, C_full, axis=-2, keepdims=True))
    assert billed == want
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    P = fnp.asarray(rng.standard_normal((8, 64)))
    Q = fnp.asarray(rng.standard_normal((8, 64)))
    single = _bill(lambda: fnp.vecdot(P, Q, keepdims=True))[0]
    at_one = _bill(
        lambda: fnp.vecdot(P, Q, axis=1, keepdims=True, out=_zeros((B, 1, 8)))
    )[0]
    assert at_one == B * single
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        with pytest.raises(ValueError):
            fnp.vecdot(P, Q, keepdims=True, out=_zeros((B, 8, 5)))
    assert int(bc.flops_used) == single


@requires_matvec
@pytest.mark.parametrize(
    ("op", "mat_shape", "axes", "batched_axes"),
    [
        ("matvec", (8, 64), [(0, 1), (0,), (0,)], [(1, 2), (0,), (0,)]),
        ("vecmat", (64, 8), [(0,), (0, 1), (0,)], [(0,), (1, 2), (0,)]),
    ],
)
def test_axes_placed_output_core_axis_bills_every_batch(
    op, mat_shape, axes, batched_axes
):
    # axes= puts the result's core axis first, and numpy reads that number
    # against out's own rank: out (8, 32) holds 32 products, m leading.
    B = 32
    rng = np.random.default_rng(6)
    mat = fnp.asarray(rng.standard_normal(mat_shape))
    vec = fnp.asarray(rng.standard_normal(64))
    fn = getattr(fnp, op)

    def operands(m):
        return (m, vec) if op == "matvec" else (vec, m)

    billed, got = _bill(lambda: fn(*operands(mat), axes=axes, out=_zeros((8, B))))
    mat_full = _full(mat, (B, *mat_shape))
    want, ref = _bill(lambda: fn(*operands(mat_full), axes=batched_axes))
    assert billed == want == B * _bill(lambda: fn(*operands(mat)))[0]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_multi_output_and_clip_destinations_bill_the_write():
    x, x_full = _operands()
    pair = (_zeros((M, N)), _zeros((M, N)))
    assert _bill(lambda: fnp.modf(x, out=pair))[0] == _bill(lambda: fnp.modf(x_full))[0]
    pair = (_zeros((M, N)), _zeros((M, N)))
    assert (
        _bill(lambda: fnp.divmod(x, 2.0, out=pair))[0]
        == _bill(lambda: fnp.divmod(x_full, 2.0))[0]
    )
    assert (
        _bill(lambda: fnp.clip(x, -1.0, 1.0, out=_zeros((M, N))))[0]
        == _bill(lambda: fnp.clip(x_full, -1.0, 1.0))[0]
    )


def test_destination_of_the_operands_shape_is_unchanged():
    # The issue's control: out= of the broadcast operand shape (or a mask of
    # it, or where=True) bills exactly like no out=.
    x, _ = _operands()
    plain = _bill(lambda: fnp.exp(x))[0]
    assert _bill(lambda: fnp.exp(x, out=_zeros(N)))[0] == plain
    assert _bill(lambda: fnp.exp(x, where=True))[0] == plain
    mask = fnp.asarray(np.arange(N) % 2 == 0)
    assert _bill(lambda: fnp.exp(x, where=mask, out=_zeros(N)))[0] == plain
    y = fnp.asarray(np.ones((M, N)))
    assert (
        _bill(lambda: fnp.add(x, y, out=_zeros((M, N))))[0]
        == _bill(lambda: fnp.add(x, y))[0]
    )
    X = fnp.asarray(np.ones((16, 16)))
    single = _bill(lambda: fnp.matmul(X, X))[0]
    assert _bill(lambda: fnp.matmul(X, X, out=_zeros((16, 16))))[0] == single
