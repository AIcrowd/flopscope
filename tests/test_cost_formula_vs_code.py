"""Regression coverage for analytical runtime costs under unit-weight resets.

These tests call ``reset_weights()`` in an autouse fixture, so the runtime
charges here continue to exercise the raw analytical formulas even though
packaged weights now autoload on a normal import path.
"""

from __future__ import annotations

import numpy
import pytest

import flopscope
from flopscope._budget import BudgetContext
from flopscope._weights import reset_weights


def _cost_of(fn, *args, **kwargs) -> int:
    """Run *fn* inside a budget and return FLOPs charged."""
    with BudgetContext(flop_budget=10**12) as b:
        fn(*args, **kwargs)
    return b.flops_used


@pytest.fixture(autouse=True)
def _reset_runtime_weights():
    reset_weights()
    yield
    reset_weights()


@pytest.fixture(autouse=True)
def _deterministic_numpy_random(monkeypatch):
    rng = numpy.random.default_rng(0)

    def _rand(*dims):
        if not dims:
            return float(rng.random())
        return rng.random(dims)

    def _randint(low, high=None, size=None, dtype=int):
        return rng.integers(low, high=high, size=size, dtype=dtype)

    monkeypatch.setattr(numpy.random, "rand", _rand)
    monkeypatch.setattr(numpy.random, "randint", _randint)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def we():
    """Fixture returning ``flopscope.numpy`` (the counted numpy surface)."""
    import flopscope.numpy

    return flopscope.numpy


# ---------------------------------------------------------------------------
# Counted Unary — numel(output)
# ---------------------------------------------------------------------------

_UNARY_NUMEL = [
    "abs",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "i0",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "nan_to_num",
    "negative",
    "positive",
    "rad2deg",
    "radians",
    "real",
    "reciprocal",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
    "angle",
    "frexp",
    "modf",
    "real_if_close",
    "imag",
    "bitwise_invert",
    "bitwise_not",
    "invert",
    "bitwise_count",
    "iscomplex",
    "isreal",
    "isneginf",
    "isposinf",
    "iscomplexobj",
    "isrealobj",
]


def _unary_input(name):
    a = numpy.random.rand(10, 10)
    if name in ("arccos", "arcsin"):
        return numpy.clip(a, 0.01, 0.99)
    if name == "arccosh":
        return numpy.abs(a) + 1.1
    if name in ("log", "log10", "log1p", "log2", "sqrt", "reciprocal"):
        return numpy.abs(a) + 0.1
    if name in ("angle", "real_if_close", "imag"):
        return a.astype(complex)
    if name in ("bitwise_invert", "bitwise_not", "invert", "bitwise_count"):
        return numpy.random.randint(0, 255, (10, 10))
    return a


@pytest.mark.parametrize("name", _UNARY_NUMEL)
def test_unary_numel(name, we):
    fn = getattr(we, name)
    inp = _unary_input(name)
    cost = _cost_of(fn, inp)
    assert cost == 100, f"{name}: expected numel=100, got {cost}"


def test_isclose_numel(we):
    a = numpy.random.rand(10, 10)
    assert _cost_of(we.isclose, a, a) == 100


def test_isnat_numel(we):
    dt = numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64")
    assert _cost_of(we.isnat, dt) == 2


# ---------------------------------------------------------------------------
# Counted Binary — numel(output)
# ---------------------------------------------------------------------------

_BINARY_NUMEL = [
    "add",
    "arctan2",
    "copysign",
    "divide",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "subtract",
    "true_divide",
    "ldexp",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
    "gcd",
    "lcm",
]


@pytest.mark.parametrize("name", _BINARY_NUMEL)
def test_binary_numel(name, we):
    if name in (
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "left_shift",
        "right_shift",
        "gcd",
        "lcm",
    ):
        a = numpy.random.randint(1, 255, (10, 10))
        b = numpy.random.randint(1, 255, (10, 10))
    elif name == "ldexp":
        a = numpy.random.rand(10, 10)
        b = numpy.ones((10, 10), dtype=int)
    else:
        a = numpy.random.rand(10, 10)
        b = numpy.random.rand(10, 10) + 0.1
    fn = getattr(we, name)
    cost = _cost_of(fn, a, b)
    assert cost == 100, f"{name}: expected numel=100, got {cost}"


def test_vecdot_batch_times_k(we):
    # formula: batch * K (output_size * contracted_axis)
    cost = _cost_of(we.vecdot, numpy.random.rand(5, 10), numpy.random.rand(5, 10))
    assert cost == 50, f"vecdot: expected 5*10=50, got {cost}"


# ---------------------------------------------------------------------------
# Counted Reduction — numel(input)
# ---------------------------------------------------------------------------

_REDUCTION_NUMEL = [
    "all",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "max",
    "min",
    "prod",
    "ptp",
    "sum",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmin",
    "nanprod",
    "nansum",
    "nanmedian",
    # std/var also cost numel(input) per sheet
    # (mean is excluded: it charges +1 divide for the scalar output orbit)
    "average",
    "nanmean",
    "std",
    "var",
    "nanstd",
    "nanvar",
]


@pytest.mark.parametrize("name", _REDUCTION_NUMEL)
def test_reduction_numel(name, we):
    # Updated for orbit-mapping cost model (PR #91 Task 7).
    # Full reduction of (10,10): prod(shape) - 1 = 100 - 1 = 99 additions.
    a = numpy.random.rand(10, 10)
    fn = getattr(we, name)
    cost = _cost_of(fn, a)
    assert cost == 99, f"{name}: expected orbit-mapping cost=99, got {cost}"


def test_mean_charges_sum_plus_one_divide(we):
    # Task 9: mean charges sum-cost + num_output_orbits divides.
    # Full reduction of (10,10) dense: sum cost = 99, scalar output → 1 divide.
    # Total = 100.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.mean, a)
    assert cost == 100, f"mean: expected sum_cost(99) + 1 divide = 100, got {cost}"


def test_median_tier2_cost(we):
    # Task 10: median uses Tier-2 model: num_output_orbits × axis_dim.
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100,
    # scalar output → 1 orbit. Cost = 1 * 100 = 100.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.median, a)
    assert cost == 100, f"median: expected Tier-2 cost=100, got {cost}"


def test_percentile_tier2_cost(we):
    # Task 11: percentile uses Tier-2 model: num_output_orbits × axis_dim.
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100,
    # scalar output → 1 orbit. Cost = 1 * 100 = 100.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.percentile, a, q=50)
    assert cost == 100, f"percentile: expected Tier-2 cost=100, got {cost}"


@pytest.mark.parametrize("name", ["nanpercentile"])
def test_nanpercentile_numel(name, we):
    # nanpercentile still uses the old orbit-mapping model.
    # Full reduction of (10,10): prod(shape) - 1 = 100 - 1 = 99 additions.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(getattr(we, name), a, q=50)
    assert cost == 99, f"{name}: expected orbit-mapping cost=99, got {cost}"


def test_quantile_tier2_cost(we):
    # Task 11: quantile uses Tier-2 model: num_output_orbits × axis_dim.
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100,
    # scalar output → 1 orbit. Cost = 1 * 100 = 100.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.quantile, a, q=0.5)
    assert cost == 100, f"quantile: expected Tier-2 cost=100, got {cost}"


@pytest.mark.parametrize("name", ["nanquantile"])
def test_nanquantile_numel(name, we):
    # nanquantile still uses the old orbit-mapping model.
    # Full reduction of (10,10): prod(shape) - 1 = 100 - 1 = 99 additions.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(getattr(we, name), a, q=0.5)
    assert cost == 99, f"{name}: expected orbit-mapping cost=99, got {cost}"


@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
def test_cumulative_numel(name, we):
    # Updated for orbit-mapping cost model (PR #91 Task 7).
    # axis=0 reduction of (10,10): 10 cols * (10-1) additions = 90.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(getattr(we, name), a, axis=0)
    assert cost == 90, f"{name}: expected orbit-mapping cost=90, got {cost}"


# ---------------------------------------------------------------------------
# Contractions — MNK / custom
# ---------------------------------------------------------------------------


def test_matmul_mnk(we):
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
    # First cell of each output orbit is a free copy.
    assert (
        _cost_of(we.matmul, numpy.random.rand(10, 10), numpy.random.rand(10, 10))
        == 1900
    )


def test_dot_mnk(we):
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
    # First cell of each output orbit is a free copy.
    assert (
        _cost_of(we.dot, numpy.random.rand(10, 10), numpy.random.rand(10, 10)) == 1900
    )


def test_inner_n(we):
    # migrated to einsum "i,i->" cost path: 2*n-1 = 39 for n=20
    assert _cost_of(we.inner, numpy.random.rand(20), numpy.random.rand(20)) == 39


def test_vdot_n(we):
    # migrated to einsum "i,i->" cost path: 2*n-1 = 39 for n=20
    assert _cost_of(we.vdot, numpy.random.rand(20), numpy.random.rand(20)) == 39


def test_outer_mn(we):
    assert _cost_of(we.outer, numpy.random.rand(10), numpy.random.rand(15)) == 150


def test_tensordot_contracted(we):
    # tensordot uses the old dense formula: a.size*b.size/contracted = 5*4*4*3/4 = 60
    # (tensordot has its own cost model, separate from the accumulation-based einsum model)
    assert (
        _cost_of(
            we.tensordot,
            numpy.random.rand(5, 4),
            numpy.random.rand(4, 3),
            axes=([1], [0]),
        )
        == 60
    )


def test_kron_numel_output(we):
    assert _cost_of(we.kron, numpy.random.rand(3, 3), numpy.random.rand(2, 2)) == 36


def test_cross_6n(we):
    # cross charges r.size * 3
    assert _cost_of(we.cross, numpy.random.rand(5, 3), numpy.random.rand(5, 3)) == 45


def test_einsum_mnk(we):
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
    # First cell of each output orbit is a free copy.
    assert (
        _cost_of(
            we.einsum, "ij,jk->ik", numpy.random.rand(10, 10), numpy.random.rand(10, 10)
        )
        == 1900
    )


def test_einsum_path_cost_1(we):
    assert (
        _cost_of(
            we.einsum_path,
            "ij,jk->ik",
            numpy.random.rand(10, 10),
            numpy.random.rand(10, 10),
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Linalg — decompositions, solvers, properties
# ---------------------------------------------------------------------------


class TestLinalgDecompositions:
    def test_cholesky_n3(self, we):
        S = numpy.eye(8) + numpy.random.rand(8, 8)
        S = S @ S.T
        assert _cost_of(we.linalg.cholesky, S) == 512

    def test_qr_mnk(self, we):
        assert _cost_of(we.linalg.qr, numpy.random.rand(10, 5)) == 250

    @pytest.mark.parametrize("name", ["eig", "eigvals"])
    def test_eig_n3(self, name, we):
        assert _cost_of(getattr(we.linalg, name), numpy.random.rand(8, 8)) == 512

    @pytest.mark.parametrize("name", ["eigh", "eigvalsh"])
    def test_eigh_n3(self, name, we):
        S = numpy.eye(8) + numpy.random.rand(8, 8)
        S = S @ S.T
        assert _cost_of(getattr(we.linalg, name), S) == 512

    def test_svd_mnk(self, we):
        assert _cost_of(we.linalg.svd, numpy.random.rand(10, 5)) == 250

    def test_svdvals_mnk(self, we):
        assert _cost_of(we.linalg.svdvals, numpy.random.rand(10, 5)) == 250


class TestLinalgSolvers:
    def test_solve_n3(self, we):
        assert (
            _cost_of(we.linalg.solve, numpy.random.rand(8, 8), numpy.random.rand(8))
            == 512
        )

    def test_inv_n3(self, we):
        assert _cost_of(we.linalg.inv, numpy.random.rand(8, 8)) == 512

    def test_lstsq_mnk(self, we):
        assert (
            _cost_of(we.linalg.lstsq, numpy.random.rand(10, 5), numpy.random.rand(10))
            == 250
        )

    def test_pinv_mnk(self, we):
        assert _cost_of(we.linalg.pinv, numpy.random.rand(10, 5)) == 250

    def test_tensorsolve_n3(self, we):
        assert (
            _cost_of(
                we.linalg.tensorsolve,
                numpy.eye(4).reshape(2, 2, 2, 2),
                numpy.random.rand(2, 2),
            )
            == 64
        )

    def test_tensorinv_n3(self, we):
        assert _cost_of(we.linalg.tensorinv, numpy.eye(4).reshape(2, 2, 2, 2)) == 64


class TestLinalgProperties:
    def test_det_n3(self, we):
        assert _cost_of(we.linalg.det, numpy.random.rand(8, 8)) == 512

    def test_slogdet_n3(self, we):
        assert _cost_of(we.linalg.slogdet, numpy.random.rand(8, 8)) == 512

    def test_cond_mnk(self, we):
        assert _cost_of(we.linalg.cond, numpy.random.rand(8, 8)) == 512

    def test_matrix_rank_mnk(self, we):
        assert _cost_of(we.linalg.matrix_rank, numpy.random.rand(10, 5)) == 250

    def test_trace(self, we):
        assert _cost_of(we.trace, numpy.random.rand(8, 8)) == 8

    def test_linalg_trace(self, we):
        assert _cost_of(we.linalg.trace, numpy.random.rand(8, 8)) == 8

    def test_vector_norm_numel(self, we):
        assert (
            _cost_of(we.linalg.vector_norm, numpy.random.rand(20)) == 40
        )  # FMA=2: 2*numel

    def test_matrix_norm_numel(self, we):
        assert (
            _cost_of(we.linalg.matrix_norm, numpy.random.rand(8, 8)) == 128
        )  # FMA=2: 2*numel

    def test_norm_vector_numel(self, we):
        assert _cost_of(we.linalg.norm, numpy.random.rand(20)) == 40  # FMA=2: 2*numel

    def test_norm_matrix_numel(self, we):
        assert (
            _cost_of(we.linalg.norm, numpy.random.rand(8, 8)) == 128
        )  # FMA=2: 2*numel


class TestLinalgDelegates:
    def test_matmul_mnk(self, we):
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
        # First cell of each output orbit is a free copy.
        assert (
            _cost_of(
                we.linalg.matmul, numpy.random.rand(10, 10), numpy.random.rand(10, 10)
            )
            == 1900
        )

    def test_outer_mn(self, we):
        assert (
            _cost_of(we.linalg.outer, numpy.random.rand(10), numpy.random.rand(15))
            == 150
        )

    def test_vecdot(self, we):
        assert (
            _cost_of(
                we.linalg.vecdot, numpy.random.rand(5, 10), numpy.random.rand(5, 10)
            )
            == 50
        )

    def test_cross(self, we):
        assert (
            _cost_of(we.linalg.cross, numpy.random.rand(5, 3), numpy.random.rand(5, 3))
            == 45
        )

    def test_matrix_power(self, we):
        # k=4: ceil(log2(4))=2, popcount(4)=1, (2+1-1)*8^3 = 1024
        assert _cost_of(we.linalg.matrix_power, numpy.random.rand(8, 8), 4) == 1024


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


class TestPolynomial:
    def test_polyval_m_times_deg(self, we):
        # Updated for FMA=2 unification (spec 2026-05-20): polyval formula doubled m*deg → 2*m*deg.
        # 5 coeffs → deg=4, m=20 → 2*20*4 = 160
        assert (
            _cost_of(
                we.polyval,
                numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                numpy.random.rand(20),
            )
            == 160
        )

    def test_polyadd(self, we):
        assert _cost_of(we.polyadd, numpy.ones(5), numpy.ones(3)) == 5

    def test_polysub(self, we):
        assert _cost_of(we.polysub, numpy.ones(5), numpy.ones(3)) == 5

    def test_polyder(self, we):
        assert _cost_of(we.polyder, numpy.ones(5)) == 5

    def test_polyint(self, we):
        assert _cost_of(we.polyint, numpy.ones(5)) == 5

    def test_polymul(self, we):
        assert _cost_of(we.polymul, numpy.ones(5), numpy.ones(3)) == 15

    def test_polydiv(self, we):
        assert _cost_of(we.polydiv, numpy.ones(5), numpy.ones(3)) == 15

    def test_polyfit(self, we):
        x = numpy.random.rand(20)
        assert _cost_of(we.polyfit, x, numpy.random.rand(20), 2) == 360

    def test_poly(self, we):
        assert _cost_of(we.poly, numpy.ones(5)) == 25

    def test_roots(self, we):
        assert _cost_of(we.roots, numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 64


# ---------------------------------------------------------------------------
# Sorting / Search / Set
# ---------------------------------------------------------------------------


class TestSorting:
    def test_sort_nlogn(self, we):
        assert _cost_of(we.sort, numpy.random.rand(100)) == 700

    def test_argsort_nlogn(self, we):
        assert _cost_of(we.argsort, numpy.random.rand(100)) == 700

    def test_sort_complex_nlogn(self, we):
        assert _cost_of(we.sort_complex, numpy.random.rand(100)) == 700

    def test_partition_n(self, we):
        assert _cost_of(we.partition, numpy.random.rand(100), 50) == 100

    def test_argpartition_n(self, we):
        assert _cost_of(we.argpartition, numpy.random.rand(100), 50) == 100

    def test_searchsorted(self, we):
        # 10 * ceil(log2(64)) = 60
        assert (
            _cost_of(
                we.searchsorted,
                numpy.sort(numpy.random.rand(64)),
                numpy.random.rand(10),
            )
            == 60
        )

    def test_digitize(self, we):
        assert (
            _cost_of(
                we.digitize, numpy.random.rand(10), numpy.sort(numpy.random.rand(64))
            )
            == 60
        )

    def test_unique_nlogn(self, we):
        assert _cost_of(we.unique, numpy.random.rand(100)) == 700


class TestSetOps:
    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(
                "in1d",
                marks=pytest.mark.skipif(
                    not hasattr(numpy, "in1d"), reason="numpy 2.4+ removed in1d"
                ),
            ),
            "isin",
            "intersect1d",
            "union1d",
            "setdiff1d",
            "setxor1d",
        ],
    )
    def test_set_op_cost(self, name, we):
        # (100+50)*ceil(log2(150)) = 150*8 = 1200
        cost = _cost_of(
            getattr(we, name), numpy.random.rand(100), numpy.random.rand(50)
        )
        assert cost == 1200, f"{name}: expected 1200, got {cost}"


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


class TestWindows:
    def test_bartlett_n(self, we):
        assert _cost_of(we.bartlett, 20) == 20

    def test_hamming_n(self, we):
        # Updated for FMA=2 unification (spec 2026-05-20): formula doubled n → 2*n.
        assert _cost_of(we.hamming, 20) == 40

    def test_hanning_n(self, we):
        # Updated for FMA=2 unification (spec 2026-05-20): formula doubled n → 2*n.
        assert _cost_of(we.hanning, 20) == 40

    def test_blackman_3n(self, we):
        assert _cost_of(we.blackman, 20) == 60

    def test_kaiser_3n(self, we):
        assert _cost_of(we.kaiser, 20, 5.0) == 60


# ---------------------------------------------------------------------------
# Statistics — corrcoef / cov
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_corrcoef_2f2s(self, we):
        # 3 features, 10 samples → 2*3^2*10 = 180
        assert _cost_of(we.corrcoef, numpy.random.rand(3, 10)) == 180

    def test_cov_2f2s(self, we):
        assert _cost_of(we.cov, numpy.random.rand(3, 10)) == 180

    def test_interp_n_log_xp(self, we):
        # 10 * ceil(log2(32)) = 50
        assert (
            _cost_of(
                we.interp,
                numpy.random.rand(10) * 31,
                numpy.arange(32, dtype=float),
                numpy.random.rand(32),
            )
            == 50
        )


# ---------------------------------------------------------------------------
# Formerly-free ops (spot checks)
# ---------------------------------------------------------------------------


class TestFreeOps:
    def test_append_numel_values(self, we):
        assert _cost_of(we.append, numpy.array([1, 2, 3]), [4, 5]) == 2

    def test_delete_num_deleted(self, we):
        assert _cost_of(we.delete, numpy.array([1, 2, 3, 4, 5]), [0, 2]) == 2

    def test_insert_numel_values(self, we):
        assert _cost_of(we.insert, numpy.array([1, 2, 3]), 1, [10, 20]) == 2

    def test_trim_zeros_num_trimmed(self, we):
        assert _cost_of(we.trim_zeros, numpy.array([0, 0, 1, 2, 0, 0])) == 4

    def test_diag_1d(self, we):
        # 1D->2D: cost = numel(output) = 3*3 = 9
        assert _cost_of(we.diag, numpy.array([1, 2, 3])) == 9

    def test_diag_2d(self, we):
        assert _cost_of(we.diag, numpy.random.rand(5, 5)) == 5

    def test_fill_diagonal(self, we):
        assert _cost_of(we.fill_diagonal, numpy.zeros((5, 5)), 1.0) == 5

    def test_copyto_with_where(self, we):
        mask = numpy.array([True, False] * 5)
        assert _cost_of(we.copyto, numpy.zeros(10), numpy.ones(10), where=mask) == 5

    def test_copyto_no_where(self, we):
        assert _cost_of(we.copyto, numpy.zeros(10), numpy.ones(10)) == 10

    def test_arange(self, we):
        assert _cost_of(we.arange, 20) == 20

    def test_full(self, we):
        assert _cost_of(we.full, (3, 4), 1.0) == 12

    def test_concatenate(self, we):
        assert (
            _cost_of(we.concatenate, [numpy.random.rand(5), numpy.random.rand(3)]) == 8
        )


# ---------------------------------------------------------------------------
# FFT (spot checks)
# ---------------------------------------------------------------------------


class TestFFT:
    def test_fft_5nlogn(self, we):
        assert _cost_of(we.fft.fft, numpy.random.rand(64)) == 1920

    def test_rfft_5_half_nlogn(self, we):
        assert _cost_of(we.fft.rfft, numpy.random.rand(64)) == 960


# ---------------------------------------------------------------------------
# Random — numel(output)
# ---------------------------------------------------------------------------


class TestRandom:
    def test_rand(self, we):
        assert _cost_of(we.random.rand, 100) == 100

    def test_randn(self, we):
        assert _cost_of(we.random.randn, 100) == 100

    def test_normal_positional_size(self, we):
        """Regression: size passed as positional arg must be detected."""
        assert _cost_of(we.random.normal, 0, 1, 100) == 100

    def test_uniform_positional_size(self, we):
        assert _cost_of(we.random.uniform, 0, 1, 100) == 100

    def test_beta_positional_size(self, we):
        assert _cost_of(we.random.beta, 2, 5, 100) == 100

    def test_normal_kwarg_size(self, we):
        assert _cost_of(we.random.normal, 0, 1, size=50) == 50

    def test_normal_scalar(self, we):
        assert _cost_of(we.random.normal, 0, 1) == 1

    def test_permutation_numel(self, we):
        assert _cost_of(we.random.permutation, 100) == 100

    def test_shuffle_numel(self, we):
        assert _cost_of(we.random.shuffle, numpy.arange(100)) == 100

    def test_choice_with_replacement(self, we):
        assert (
            _cost_of(we.random.choice, numpy.arange(200), size=100, replace=True) == 100
        )


# ---------------------------------------------------------------------------
# Stats distributions — numel(input)
# ---------------------------------------------------------------------------


class TestStats:
    """All stats methods charge numel(input) * 1 = numel(input) FLOPs."""

    def test_norm_pdf(self, we):
        assert _cost_of(flopscope.stats.norm.pdf, numpy.random.rand(100)) == 100

    def test_norm_cdf(self, we):
        assert _cost_of(flopscope.stats.norm.cdf, numpy.random.rand(100)) == 100

    def test_norm_ppf(self, we):
        assert (
            _cost_of(flopscope.stats.norm.ppf, numpy.random.rand(100) * 0.98 + 0.01)
            == 100
        )

    def test_uniform_pdf(self, we):
        assert _cost_of(flopscope.stats.uniform.pdf, numpy.random.rand(100)) == 100

    def test_uniform_cdf(self, we):
        assert _cost_of(flopscope.stats.uniform.cdf, numpy.random.rand(100)) == 100

    def test_uniform_ppf(self, we):
        assert _cost_of(flopscope.stats.uniform.ppf, numpy.random.rand(100)) == 100

    def test_expon_pdf(self, we):
        assert _cost_of(flopscope.stats.expon.pdf, numpy.random.rand(100)) == 100

    def test_cauchy_pdf(self, we):
        assert _cost_of(flopscope.stats.cauchy.pdf, numpy.random.rand(100)) == 100

    def test_logistic_cdf(self, we):
        assert _cost_of(flopscope.stats.logistic.cdf, numpy.random.rand(100)) == 100

    def test_laplace_ppf(self, we):
        assert (
            _cost_of(flopscope.stats.laplace.ppf, numpy.random.rand(100) * 0.98 + 0.01)
            == 100
        )

    def test_lognorm_pdf(self, we):
        assert (
            _cost_of(
                flopscope.stats.lognorm.pdf,
                numpy.abs(numpy.random.rand(100)) + 0.1,
                0.5,
            )
            == 100
        )

    def test_truncnorm_cdf(self, we):
        assert (
            _cost_of(flopscope.stats.truncnorm.cdf, numpy.random.rand(100), -2, 2)
            == 100
        )

    def test_scalar_input(self, we):
        """Scalar input should charge 1 FLOP."""
        assert _cost_of(flopscope.stats.norm.pdf, 0.0) == 1
