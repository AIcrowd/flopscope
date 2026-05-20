"""Extended tests to cover gaps in linalg property and decomposition wrappers."""

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope._symmetric import as_symmetric
from flopscope.numpy.linalg._decompositions import (
    cholesky,
    cholesky_cost,
    eig,
    eig_cost,
    eigh,
    eigh_cost,
    eigvals,
    eigvals_cost,
    eigvalsh,
    eigvalsh_cost,
    qr,
    qr_cost,
    svdvals,
    svdvals_cost,
)
from flopscope.numpy.linalg._properties import (
    cond,
    det,
    det_cost,
    matrix_norm,
    matrix_norm_cost,
    matrix_rank,
    norm,
    norm_cost,
    slogdet,
    slogdet_cost,
    trace,
    trace_cost,
    vector_norm,
    vector_norm_cost,
)

# ---------------------------------------------------------------------------
# Cost helper functions — direct testing for edge cases
# ---------------------------------------------------------------------------


def test_trace_cost_min1():
    assert trace_cost(0) == 1
    assert trace_cost(5) == 5


def test_det_cost_symmetric():
    n = 4
    assert det_cost(n, symmetric=True) == max(n**3, 1)


def test_slogdet_cost_symmetric():
    n = 4
    assert slogdet_cost(n, symmetric=True) == max(n**3, 1)


def test_norm_cost_1d_ord_none():
    assert norm_cost((10,), ord=None) == 20  # FMA=2: 2*numel


def test_norm_cost_1d_ord_inf():
    assert norm_cost((10,), ord=numpy.inf) == 20  # FMA=2: 2*numel


def test_norm_cost_1d_ord_minus_inf():
    assert norm_cost((10,), ord=-numpy.inf) == 20  # FMA=2: 2*numel


def test_norm_cost_1d_ord_0():
    assert norm_cost((10,), ord=0) == 20  # FMA=2: 2*numel


def test_norm_cost_1d_p_norm():
    # FMA=2: all vector norms cost 2*numel
    assert norm_cost((10,), ord=3) == 20  # FMA=2: 2*numel


def test_norm_cost_2d_fro():
    # FMA=2: Frobenius norm costs 2*numel
    assert norm_cost((4, 5), ord="fro") == 40  # FMA=2: 2*numel


def test_norm_cost_2d_nuc():
    # SVD-based: 4x baked in for weight=4 consistency
    m, n = 4, 5
    assert norm_cost((m, n), ord="nuc") == 4 * m * n * min(m, n)


def test_norm_cost_2d_minus2():
    m, n = 4, 5
    assert norm_cost((m, n), ord=-2) == 4 * m * n * min(m, n)


def test_norm_cost_2d_1():
    assert norm_cost((4, 5), ord=1) == 40  # FMA=2: 2*numel


def test_norm_cost_2d_minus1():
    assert norm_cost((4, 5), ord=-1) == 40  # FMA=2: 2*numel


def test_norm_cost_2d_inf():
    assert norm_cost((4, 5), ord=numpy.inf) == 40  # FMA=2: 2*numel


def test_norm_cost_2d_minus_inf():
    assert norm_cost((4, 5), ord=-numpy.inf) == 40  # FMA=2: 2*numel


def test_norm_cost_2d_fallback():
    # Unrecognised ord for 2-D triggers fallback: 2*numel (FMA=2)
    assert norm_cost((4, 5), ord="xyz") == 40  # FMA=2: 2*numel


def test_vector_norm_cost_p_norm():
    # FMA=2: all norms cost 2*numel (one multiply + accumulate per element)
    assert vector_norm_cost((10,), ord=3) == 20  # FMA=2: 2*numel


def test_vector_norm_cost_special_ords():
    for o in (None, 2, -2, 1, -1, numpy.inf, -numpy.inf, 0):
        assert vector_norm_cost((10,), ord=o) == 20  # FMA=2: 2*numel


def test_matrix_norm_cost_fro():
    # FMA=2: Frobenius norm costs 2*numel (one multiply + accumulate per element)
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord="fro") == 2 * m * n  # FMA=2


def test_matrix_norm_cost_nuc():
    # SVD-based: 4x baked in for weight=4 consistency
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord="nuc") == 4 * m * n * min(m, n)


def test_matrix_norm_cost_2():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=2) == 4 * m * n * min(m, n)


def test_matrix_norm_cost_minus2():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-2) == 4 * m * n * min(m, n)


def test_matrix_norm_cost_1():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=1) == 2 * m * n  # FMA=2


def test_matrix_norm_cost_minus1():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-1) == 2 * m * n  # FMA=2


def test_matrix_norm_cost_inf():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=numpy.inf) == 2 * m * n  # FMA=2


def test_matrix_norm_cost_minus_inf():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-numpy.inf) == 2 * m * n  # FMA=2


def test_matrix_norm_cost_fallback():
    assert matrix_norm_cost((3, 4), ord="xyz") == 2 * 12  # FMA=2: 2*numel


# ---------------------------------------------------------------------------
# Properties — SymmetricTensor paths
# ---------------------------------------------------------------------------


def test_det_symmetric_tensor_cost():
    n = 4
    data = numpy.eye(n)
    sym_a = as_symmetric(data, symmetry=(0, 1))
    with BudgetContext(flop_budget=10**6) as budget:
        det(sym_a)
    assert budget.flops_used == det_cost(n, symmetric=True)


def test_slogdet_symmetric_tensor_cost():
    n = 4
    data = numpy.eye(n)
    sym_a = as_symmetric(data, symmetry=(0, 1))
    with BudgetContext(flop_budget=10**6) as budget:
        slogdet(sym_a)
    assert budget.flops_used == slogdet_cost(n, symmetric=True)


# ---------------------------------------------------------------------------
# Properties — trace with positive/negative offset
# ---------------------------------------------------------------------------


def test_trace_positive_offset():
    A = numpy.arange(9, dtype=float).reshape(3, 3)
    with BudgetContext(flop_budget=10**6) as budget:
        result = trace(A, offset=1)
    assert numpy.isclose(result, numpy.trace(A, offset=1))


def test_trace_negative_offset():
    A = numpy.arange(9, dtype=float).reshape(3, 3)
    with BudgetContext(flop_budget=10**6) as budget:
        result = trace(A, offset=-1)
    assert numpy.isclose(result, numpy.trace(A, offset=-1))


# ---------------------------------------------------------------------------
# Properties — norm with axis parameter
# ---------------------------------------------------------------------------


def test_norm_with_tuple_axis():
    A = numpy.random.randn(3, 4, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        result = norm(A, axis=(0, 1))
    assert result.shape == (5,)


def test_norm_with_single_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = norm(A, axis=0)
    assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Properties — vector_norm with axis
# ---------------------------------------------------------------------------


def test_vector_norm_with_tuple_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = vector_norm(A, axis=(0, 1))
    assert numpy.ndim(result) == 0 or result is not None


def test_vector_norm_with_single_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = vector_norm(A, axis=0)
    assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Properties — matrix_norm with various ord values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ord_val", [1, -1, 2, -2, numpy.inf, -numpy.inf, "nuc"])
def test_matrix_norm_various_ords(ord_val):
    A = numpy.random.randn(4, 4)
    with BudgetContext(flop_budget=10**6):
        result = matrix_norm(A, ord=ord_val)
    assert numpy.isfinite(result)


# ---------------------------------------------------------------------------
# Decompositions — cost helpers
# ---------------------------------------------------------------------------


def test_cholesky_cost():
    assert cholesky_cost(1) == 1
    assert cholesky_cost(3) == max(27, 1)


def test_qr_cost_wide_matrix():
    m, n = 3, 5
    cost = qr_cost(m, n)
    assert cost == max(m * n * min(m, n), 1)


def test_eig_cost():
    assert eig_cost(0) == 1
    assert eig_cost(4) == 64


def test_eigh_cost():
    assert eigh_cost(0) == 1
    assert eigh_cost(3) == max(27, 1)


def test_eigvals_cost():
    assert eigvals_cost(4) == 64


def test_eigvalsh_cost():
    assert eigvalsh_cost(4) == max(64, 1)


def test_svdvals_cost():
    m, n = 4, 3
    assert svdvals_cost(m, n) == m * n * min(m, n)


# ---------------------------------------------------------------------------
# Decompositions — non-square / wrong dim error paths
# ---------------------------------------------------------------------------


def test_cholesky_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            cholesky(numpy.ones((3, 4)))


def test_qr_batched():
    """Batched QR works (ndim > 2 is supported)."""
    with BudgetContext(flop_budget=10**9):
        Q, R = qr(numpy.ones((2, 3, 4)))
    assert Q.shape[0] == 2  # batch dim preserved


def test_eig_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eig(numpy.ones((3, 4)))


def test_eigh_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigh(numpy.ones((3, 4)))


def test_eigvals_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigvals(numpy.ones((3, 4)))


def test_eigvalsh_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigvalsh(numpy.ones((3, 4)))


def test_svdvals_batched():
    """Batched svdvals works (ndim > 2 is supported)."""
    with BudgetContext(flop_budget=10**9):
        sv = svdvals(numpy.ones((2, 3, 4)))
    assert sv.shape == (2, 3)  # batch dim preserved, min(3,4)=3 singular values


# ---------------------------------------------------------------------------
# Decompositions — successful runs
# ---------------------------------------------------------------------------


def test_eigh_result():
    A = numpy.array([[2.0, 1.0], [1.0, 2.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        vals, vecs = eigh(A)
    assert budget.flops_used == eigh_cost(2)
    assert vals.shape == (2,)


def test_eigvals_result():
    A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    with BudgetContext(flop_budget=10**6):
        vals = eigvals(A)
    assert vals.shape == (2,)


def test_eigvalsh_result():
    A = numpy.array([[2.0, 1.0], [1.0, 2.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        vals = eigvalsh(A)
    assert budget.flops_used == eigvalsh_cost(2)
    assert vals.shape == (2,)


def test_svdvals_result():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6) as budget:
        sv = svdvals(A)
    assert budget.flops_used == svdvals_cost(3, 4)
    assert sv.shape == (3,)


# ---------------------------------------------------------------------------
# Properties — det/slogdet/cond/matrix_rank error paths
# ---------------------------------------------------------------------------


def test_det_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            det(numpy.ones((3, 4)))


def test_slogdet_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            slogdet(numpy.ones((3, 4)))


def test_cond_batched():
    """Batched cond works (ndim > 2 is supported)."""
    with BudgetContext(flop_budget=10**9):
        c = cond(numpy.ones((2, 3, 4)))
    assert c.shape == (2,)  # batch dim preserved


def test_matrix_rank_batched():
    """Batched matrix_rank works (ndim > 2 is supported)."""
    with BudgetContext(flop_budget=10**9):
        r = matrix_rank(numpy.ones((2, 3, 4)))
    assert r.shape == (2,)  # pyright: ignore[reportAttributeAccessIssue] # batch dim preserved
