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
    "bitwise_invert",
    "bitwise_not",
    "invert",
    "bitwise_count",
    "iscomplex",
    "isreal",
    "isneginf",
    "isposinf",
    # iscomplexobj/isrealobj (dtype predicates) and real/imag (component
    # extraction — a view, no arithmetic) are free (0 FLOPs), not counted_unary.
]


def _unary_input(name):
    a = numpy.random.rand(10, 10)
    if name in ("arccos", "arcsin"):
        return numpy.clip(a, 0.01, 0.99)
    if name == "arccosh":
        return numpy.abs(a) + 1.1
    if name in ("log", "log10", "log1p", "log2", "sqrt", "reciprocal"):
        return numpy.abs(a) + 0.1
    if name in ("angle", "real_if_close"):
        return a.astype(complex)
    if name in ("bitwise_invert", "bitwise_not", "invert", "bitwise_count"):
        return numpy.random.randint(0, 255, (10, 10))
    return a


@pytest.mark.parametrize("name", _UNARY_NUMEL)
def test_unary_numel(name, we):
    fn = getattr(we, name)
    inp = _unary_input(name)
    cost = _cost_of(fn, inp)
    # real_if_close and angle both take complex input (see _unary_input).
    # real_if_close's registry complex_factor is 2.0 (test-and-patch: checks
    # the imaginary part against a tolerance and conditionally copies the real
    # component, structurally matching isnan/isfinite); angle's is likewise
    # 2.0 (a complex value is two real components, one unit per component) --
    # unit dtype rates here, so only the complex factor applies: 100 * 2.0 =
    # 200. Every other op in this list bills numel(input) = 100 unchanged
    # (real input).
    expected = 200 if name in ("real_if_close", "angle") else 100
    assert cost == expected, f"{name}: expected {expected}, got {cost}"


def test_isclose_numel(we):
    a = numpy.random.rand(10, 10)
    assert _cost_of(we.isclose, a, a) == 6 * 100  # 6/elem tolerance core


def test_isnat_refuses_its_only_valid_input_dtype(we):
    """isnat's sole valid input kind is datetime64/timedelta64 -- both
    outside the numeric allowlist, so every real call is now refused before
    a cost formula ever runs; there is no numeric substitute for an op whose
    entire domain is non-numeric."""
    from flopscope.errors import UnsupportedDtypeError

    dt = numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64")
    with pytest.raises(UnsupportedDtypeError):
        _cost_of(we.isnat, dt)


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


def test_vecdot_fma2(we):
    # FMA=2: 5 outputs * (2*10 - 1) = 5*19 = 95
    cost = _cost_of(we.vecdot, numpy.random.rand(5, 10), numpy.random.rand(5, 10))
    assert cost == 95, f"vecdot: expected 5*(2*10-1)=95, got {cost}"


# ---------------------------------------------------------------------------
# Counted Reduction — numel(input)
# ---------------------------------------------------------------------------

_REDUCTION_NUMEL = [
    "all",
    "any",
    "argmax",
    "argmin",
    # count_nonzero is excluded: dedicated wrapper now charges numel(input) axis-independently
    # (always numel, not numel-1 from skeleton); see test_count_nonzero_bills_numel_axis_independent
    "cumprod",
    "cumsum",
    "max",
    "min",
    "prod",
    # ptp is excluded: 2-pass formula (2*numel - M); see test_ptp_two_passes
    "sum",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmin",
    "nanprod",
    "nansum",
    # nanmedian is excluded: Tier-2 cost (num_output_orbits × axis_dim); see test_nanmedian_tier2_full_reduction
    # mean is excluded: it charges +1 divide for the scalar output orbit
    # nanmean is excluded: billed identically to mean (reduction + M divides); see test_nanmean_matches_mean
    # average is excluded: now matches mean cost (reduction + M divides); see test_average_matches_mean_and_bills_weight_pipeline
    # std/var/nanstd/nanvar are excluded: 4-pass formula; see test_variance_family_cost
]


@pytest.mark.parametrize("name", _REDUCTION_NUMEL)
def test_reduction_numel(name, we):
    # Updated for orbit-mapping cost model (PR #91 Task 7).
    # Full reduction of (10,10): prod(shape) - 1 = 100 - 1 = 99 additions.
    a = numpy.random.rand(10, 10)
    fn = getattr(we, name)
    cost = _cost_of(fn, a)
    assert cost == 99, f"{name}: expected orbit-mapping cost=99, got {cost}"


def test_variance_family_cost(we):
    # 4-pass honest cost for full reduction of (10,10) dense (N=100, M=1 scalar):
    # 2*pointwise(100) + reduce(op_factor=2, extra_ops=2*1) + 1 div
    # pointwise = 100; reduce: orbit-mapping 99 + 2 extra = 101 → *2 op_factor? No:
    # compute_reduction_accumulation_cost with op_factor=2, extra_ops=2: total = 2*99 + 2 = 200
    # var = 2*100 + 200 = 400, std = 400 + 1 = 401
    a = numpy.random.rand(10, 10)
    assert _cost_of(we.var, a) == 400, f"var: expected 400, got {_cost_of(we.var, a)}"
    assert _cost_of(we.std, a) == 401, f"std: expected 401, got {_cost_of(we.std, a)}"
    assert _cost_of(we.nanvar, a) == 400, (
        f"nanvar: expected 400, got {_cost_of(we.nanvar, a)}"
    )
    assert _cost_of(we.nanstd, a) == 401, (
        f"nanstd: expected 401, got {_cost_of(we.nanstd, a)}"
    )


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


def test_nanmedian_tier2_cost(we):
    # nanmedian now uses Tier-2 model (same as median): num_output_orbits × axis_dim.
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100, scalar → 1 orbit.
    # Cost = 1 * 100 = 100.  (Was 99 from _counted_reduction skeleton.)
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.nanmedian, a)
    assert cost == 100, f"nanmedian: expected Tier-2 cost=100, got {cost}"


def test_nanmean_charges_sum_plus_one_divide(we):
    # nanmean: billed identically to mean (reduction + per-output divide).
    # Full reduction of (10,10) dense: sum cost = 99, scalar output → 1 divide.
    # Total = 100. (Was 99 from _counted_reduction skeleton.)
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.nanmean, a)
    assert cost == 100, f"nanmean: expected sum_cost(99) + 1 divide = 100, got {cost}"


def test_percentile_tier2_cost(we):
    # Task 11: percentile uses Tier-2 model: num_output_orbits × (axis_dim + 4*q.size).
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100, scalar q
    # (q.size=1) → +4, scalar output → 1 orbit. Cost = 1 * (100 + 4) = 104.
    # (Task 3: per-quantile gather+interpolate term, was flat axis_dim=100.)
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.percentile, a, q=50)
    assert cost == 104, f"percentile: expected Tier-2 cost=104, got {cost}"


@pytest.mark.parametrize("name", ["nanpercentile"])
def test_nanpercentile_numel(name, we):
    # nanpercentile now uses Tier-2 model (same as percentile):
    # num_output_orbits × (axis_dim + 4*q.size).
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100, scalar q
    # (q.size=1) → +4, scalar output → 1 orbit. Cost = 1 * (100 + 4) = 104.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(getattr(we, name), a, q=50)
    assert cost == 104, f"{name}: expected Tier-2 cost=104, got {cost}"


def test_quantile_tier2_cost(we):
    # Task 11: quantile uses Tier-2 model: num_output_orbits × (axis_dim + 4*q.size).
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100, scalar q
    # (q.size=1) → +4, scalar output → 1 orbit. Cost = 1 * (100 + 4) = 104.
    # (Task 3: per-quantile gather+interpolate term, was flat axis_dim=100.)
    a = numpy.random.rand(10, 10)
    cost = _cost_of(we.quantile, a, q=0.5)
    assert cost == 104, f"quantile: expected Tier-2 cost=104, got {cost}"


@pytest.mark.parametrize("name", ["nanquantile"])
def test_nanquantile_numel(name, we):
    # nanquantile now uses Tier-2 model (same as quantile):
    # num_output_orbits × (axis_dim + 4*q.size).
    # Full reduction of (10,10) dense: axis_dim = prod(shape) = 100, scalar q
    # (q.size=1) → +4, scalar output → 1 orbit. Cost = 1 * (100 + 4) = 104.
    a = numpy.random.rand(10, 10)
    cost = _cost_of(getattr(we, name), a, q=0.5)
    assert cost == 104, f"{name}: expected Tier-2 cost=104, got {cost}"


@pytest.mark.skipif(
    not hasattr(numpy, "cumulative_sum"), reason="requires numpy >= 2.1"
)
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
    # tensordot partial-contraction now routes through einsum (FMA=2).
    # (5,4)·(4,3) axes=([1],[0]) -> "ab,bc->ac"; einsum cost = 5*3*(2*4-1) = 105
    assert (
        _cost_of(
            we.tensordot,
            numpy.random.rand(5, 4),
            numpy.random.rand(4, 3),
            axes=([1], [0]),
        )
        == 105
    )


def test_kron_numel_output(we):
    assert _cost_of(we.kron, numpy.random.rand(3, 3), numpy.random.rand(2, 2)) == 36


def test_cross_6n(we):
    # cross charges a.shape[0] * 3 * 3 (3 ops/output element, 6 mul + 3 sub)
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
        # cholesky_cost(8) = 8^3//3 = 170
        assert _cost_of(we.linalg.cholesky, S) == 170

    def test_qr_mnk(self, we):
        # qr_cost(10,5,mode="reduced"): k=5, factor=2*10*5*5-2*5^3//3=500-83=417, 2*factor=834
        assert _cost_of(we.linalg.qr, numpy.random.rand(10, 5)) == 834

    @pytest.mark.parametrize("name", ["eig", "eigvals"])
    def test_eig_n3(self, name, we):
        # eig: 25*n^3=25*512=12800; eigvals: 10*n^3=10*512=5120 (n=8, PROVISIONAL)
        expected = {"eig": 25 * 8**3, "eigvals": 10 * 8**3}
        assert (
            _cost_of(getattr(we.linalg, name), numpy.random.rand(8, 8))
            == expected[name]
        )

    @pytest.mark.parametrize("name", ["eigh", "eigvalsh"])
    def test_eigh_n3(self, name, we):
        S = numpy.eye(8) + numpy.random.rand(8, 8)
        S = S @ S.T
        # eigh: 9*n^3=9*512=4608; eigvalsh: 4*n^3//3=4*512//3=682 (n=8, PROVISIONAL)
        expected = {"eigh": 9 * 8**3, "eigvalsh": 4 * 8**3 // 3}
        assert _cost_of(getattr(we.linalg, name), S) == expected[name]

    def test_svd_mnk(self, we):
        # full_matrices=True (default), non-square (10,5): 4*a^2*b+22*b^3
        # a=10, b=5: 4*100*5+22*125=2000+2750=4750
        assert _cost_of(we.linalg.svd, numpy.random.rand(10, 5)) == 4750
        # thin (full_matrices=False): 6*10*25+20*125=1500+2500=4000
        assert (
            _cost_of(we.linalg.svd, numpy.random.rand(10, 5), full_matrices=False)
            == 4000
        )

    def test_svdvals_mnk(self, we):
        # values-only: 2*10*25+2*125=500+250=750
        assert _cost_of(we.linalg.svdvals, numpy.random.rand(10, 5)) == 750


class TestLinalgSolvers:
    def test_solve_n3(self, we):
        # solve_cost(8, nrhs=1): 2*8^3//3 + 2*8^2*1 = 341 + 128 = 469
        assert (
            _cost_of(we.linalg.solve, numpy.random.rand(8, 8), numpy.random.rand(8))
            == 469
        )

    def test_inv_n3(self, we):
        # inv_cost(8): 2*8^3 = 1024
        assert _cost_of(we.linalg.inv, numpy.random.rand(8, 8)) == 1024

    def test_lstsq_mnk(self, we):
        # lstsq_cost(10,5,b_cols=1,b_ndim=1):
        #   k=5, svd=svd_cost(10,5,with_vectors=True)=4000
        #   ut_b=matmul_cost(5,10,1)=2*5*10*1-5*1=95
        #   divide=5*1=5
        #   reconstruction=matmul_cost(5,5,1)=2*5*5*1-5*1=45
        #   total=4000+95+5+45=4145
        assert (
            _cost_of(we.linalg.lstsq, numpy.random.rand(10, 5), numpy.random.rand(10))
            == 4145
        )

    def test_pinv_mnk(self, we):
        # pinv_cost(10,5): svd(with_vecs)=4000+threshold=5+diag_scale=25+matmul(5,5,10)=450
        assert _cost_of(we.linalg.pinv, numpy.random.rand(10, 5)) == 4480

    def test_tensorsolve_n3(self, we):
        # tensorsolve_cost((2,2,2,2)): n=prod(trailing 2)=4; 2*4^3//3 + 2*4^2 = 42 + 32 = 74
        assert (
            _cost_of(
                we.linalg.tensorsolve,
                numpy.eye(4).reshape(2, 2, 2, 2),
                numpy.random.rand(2, 2),
            )
            == 74
        )

    def test_tensorinv_n3(self, we):
        # tensorinv_cost((2,2,2,2)): n=prod(leading 2)=4; 2*4^3 = 128
        assert _cost_of(we.linalg.tensorinv, numpy.eye(4).reshape(2, 2, 2, 2)) == 128


class TestLinalgProperties:
    def test_det_n3(self, we):
        # det_cost(8) = 2*8^3//3 + 8 = 341 + 8 = 349
        assert _cost_of(we.linalg.det, numpy.random.rand(8, 8)) == 349

    def test_slogdet_n3(self, we):
        # slogdet_cost(8) = 2*8^3//3 + 18*8 = 341 + 144 = 485
        # (LU term + n log|diag| calls: abs + 16/elem log + reduce = 18n)
        assert (
            _cost_of(we.linalg.slogdet, numpy.random.rand(8, 8))
            == 2 * 8**3 // 3 + 18 * 8
        )

    def test_cond_mnk(self, we):
        # cond_cost(8,8): values-only SVD(8,8)=2*8*64+2*512=1024+1024=2048, +1=2049
        assert _cost_of(we.linalg.cond, numpy.random.rand(8, 8)) == 2049

    def test_matrix_rank_mnk(self, we):
        # matrix_rank_cost(10,5): svd_vals(10,5)+min(10,5)=750+5=755
        assert _cost_of(we.linalg.matrix_rank, numpy.random.rand(10, 5)) == 755

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
        # FMA=2: 5 outputs * (2*10 - 1) = 5*19 = 95
        assert (
            _cost_of(
                we.linalg.vecdot, numpy.random.rand(5, 10), numpy.random.rand(5, 10)
            )
            == 95
        )

    def test_cross(self, we):
        # linalg.cross charges out_size * 3 (3 ops/output element, 6 mul + 3 sub)
        assert (
            _cost_of(we.linalg.cross, numpy.random.rand(5, 3), numpy.random.rand(5, 3))
            == 45
        )

    def test_matrix_power(self, we):
        # k=4: floor(log2(4))=2, popcount(4)=1, num_ops=(2+1-1)=2 matmuls
        # matmul_cost(8,8,8) = 2*512 - 64 = 960; total = 2*960 = 1920
        assert _cost_of(we.linalg.matrix_power, numpy.random.rand(8, 8), 4) == 1920


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
        # n=5, m=1: t=min(1,4)=1; cost=1*5 - 1*2//2 = 4
        assert _cost_of(we.polyder, numpy.ones(5)) == 4

    def test_polyint(self, we):
        assert _cost_of(we.polyint, numpy.ones(5)) == 5

    def test_polymul(self, we):
        assert _cost_of(we.polymul, numpy.ones(5), numpy.ones(3)) == 22

    def test_polydiv(self, we):
        assert _cost_of(we.polydiv, numpy.ones(5), numpy.ones(3)) == 22

    def test_polyfit(self, we):
        # m=20, deg=2 (n=3 Vandermonde columns): m*deg (Vandermonde build)
        # + lstsq_cost(20, 3, ncols=1) = 40 + 1755 = 1795. polyfit_cost now
        # delegates to lstsq_cost (SVD least-squares), replacing the old
        # normal-equations-shaped 2*m*(deg+1)^2 estimate (360 here) that
        # billed 3-13x cheaper than the identical solve through lstsq.
        x = numpy.random.rand(20)
        assert _cost_of(we.polyfit, x, numpy.random.rand(20), 2) == 1795

    def test_poly(self, we):
        # n=5: (3*25 + 5) // 2 = 80 // 2 = 40  (was 2*25=50)
        assert _cost_of(we.poly, numpy.ones(5)) == 40

    def test_roots(self, we):
        # degree=4 (len=5 -> n=4); eigvals_cost(4)=10*64=640 (PROVISIONAL)
        assert _cost_of(we.roots, numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 10 * 4**3


# ---------------------------------------------------------------------------
# Sorting / Search / Set
# ---------------------------------------------------------------------------


class TestSorting:
    def test_sort_nlogn(self, we):
        assert _cost_of(we.sort, numpy.random.rand(100)) == 700

    def test_argsort_nlogn(self, we):
        assert _cost_of(we.argsort, numpy.random.rand(100)) == 700

    def test_sort_complex_nlogn(self, we):
        # np.sort_complex ALWAYS returns a complex array (its own docstring:
        # "Always returns a sorted complex array") -- it lexicographically
        # compares real-then-imaginary parts even for a real float64 input,
        # so the registry's complex_factor=2.0 (ordering compare) applies
        # unconditionally: base cost 700 (n=100 * ceil(log2 100)=7) * factor
        # 2.0 = 1400. Compute-dtype conformance sweep (Task 8) fixed
        # sort_complex to resolve its real numpy output dtype (complex64/128,
        # never the raw real input dtype), closing the undercount this test
        # used to pin (base cost alone, at rate/factor 1.0 = 700).
        assert _cost_of(we.sort_complex, numpy.random.rand(100)) == 1400

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

    def test_intersect1d_cost(self, we):
        # Default assume_unique=False: numpy unique()-sorts both inputs first.
        # sort_cost(100) + sort_cost(50) + sort_cost(150) = 700 + 350 + 1200 = 2250
        from flopscope._flops import sort_cost as _sc

        cost = _cost_of(we.intersect1d, numpy.random.rand(100), numpy.random.rand(50))
        assert cost == _sc(100) + _sc(50) + _sc(150)

    def test_intersect1d_cost_assume_unique(self, we):
        # assume_unique=True: only sort_cost(n+m)
        from flopscope._flops import sort_cost as _sc

        cost = _cost_of(
            we.intersect1d,
            numpy.random.rand(100),
            numpy.random.rand(50),
            assume_unique=True,
        )
        assert cost == _sc(150)


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


class TestWindows:
    def test_bartlett_4n(self, we):
        # Updated: compare+div+add+select per sample (FMA=2); 4 ops/point
        assert _cost_of(we.bartlett, 20) == 4 * 20

    def test_hamming_18n(self, we):
        # Updated: cos@16 + mul + sub per sample (kaiser-family derived
        # constant convention); 18 ops/point.
        assert _cost_of(we.hamming, 20) == 18 * 20

    def test_hanning_18n(self, we):
        # Updated: cos@16 + mul + sub per sample (kaiser-family derived
        # constant convention); 18 ops/point.
        assert _cost_of(we.hanning, 20) == 18 * 20

    def test_blackman_40n(self, we):
        # Updated: 2 cos evals @16 + 8 arith per sample; 40 ops/point
        assert _cost_of(we.blackman, 20) == 40 * 20

    def test_kaiser_23n(self, we):
        # Updated: 1 Bessel I0 @16 + 7 arith per sample; 23 ops/point
        assert _cost_of(we.kaiser, 20, 5.0) == 23 * 20


# ---------------------------------------------------------------------------
# Statistics — corrcoef / cov
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_corrcoef_2f2s(self, we):
        # 3 features, 10 samples → f=3, s=10
        # cov: 2*9*10 + 2*3*10 = 180 + 60 = 240
        # + normalization: 2*9 + 3 = 21
        # total: 261
        assert _cost_of(we.corrcoef, numpy.random.rand(3, 10)) == 261

    def test_cov_2f2s(self, we):
        # 3 features, 10 samples → 2*f^2*s + 2*f*s = 2*9*10 + 2*3*10 = 180 + 60 = 240
        assert _cost_of(we.cov, numpy.random.rand(3, 10)) == 240

    def test_interp_n_log_xp(self, we):
        # 3*10 + 10*ceil(log2(32)) = 30 + 10*5 = 80
        assert (
            _cost_of(
                we.interp,
                numpy.random.rand(10) * 31,
                numpy.arange(32, dtype=float),
                numpy.random.rand(32),
            )
            == 80
        )


# ---------------------------------------------------------------------------
# Formerly-free ops (spot checks)
# ---------------------------------------------------------------------------


class TestFreeOps:
    def test_append_numel_output(self, we):
        # np.append = concatenate([arr, values]); bills numel(output) = arr.size + values.size
        assert (
            _cost_of(we.append, numpy.array([1, 2, 3]), [4, 5]) == 5
        )  # was 2 (values.size only)

    def test_delete_numel_output(self, we):
        # np.delete copies surviving elements; bills numel(output) = arr.size - deleted
        assert (
            _cost_of(we.delete, numpy.array([1, 2, 3, 4, 5]), [0, 2]) == 3
        )  # was 2 (num_deleted)

    def test_insert_numel_output(self, we):
        # np.insert copies all elements; bills numel(output) = arr.size + values.size
        assert (
            _cost_of(we.insert, numpy.array([1, 2, 3]), 1, [10, 20]) == 5
        )  # was 2 (values.size only)

    def test_trim_zeros_numel_input(self, we):
        # value scan for the nonzero boundary = numel(input)
        assert _cost_of(we.trim_zeros, numpy.array([0, 0, 1, 2, 0, 0])) == 6

    def test_diag_1d(self, we):
        # 1-D construct: cost = v.shape[0] = 3 (Task 7)
        assert _cost_of(we.diag, numpy.array([1, 2, 3])) == 3

    def test_diag_2d(self, we):
        # 2-D extract: numpy.diag returns a view -> 0 (Task 7)
        assert _cost_of(we.diag, numpy.random.rand(5, 5)) == 0

    def test_fill_diagonal(self, we):
        assert _cost_of(we.fill_diagonal, numpy.zeros((5, 5)), 1.0) == 5

    def test_copyto_same_dtype_where_free(self, we):
        mask = numpy.array([True, False] * 5)
        # copyto bills per element selected by where mask -> popcount(mask) = 5
        assert _cost_of(we.copyto, numpy.zeros(10), numpy.ones(10), where=mask) == 5

    def test_copyto_same_dtype_free(self, we):
        # copyto bills per element written -> numel(dst) = 10
        assert _cost_of(we.copyto, numpy.zeros(10), numpy.ones(10)) == 10

    def test_copyto_value_changing_cast(self, we):
        # float -> int cast computes per element -> numel(dst)
        assert (
            _cost_of(
                we.copyto,
                numpy.zeros(10, dtype=numpy.int64),
                numpy.ones(10),
                casting="unsafe",
            )
            == 10
        )

    def test_arange(self, we):
        assert (
            _cost_of(we.arange, 20) == 2 * 20
        )  # migrated: arange bills 2*numel (start + i*step, FMA=2)

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
        assert _cost_of(we.random.uniform, 0, 1, 100) == 3 * 100

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
    """Stats methods charge composite cost_per_elem * numel(input) FLOPs (weight=1.0)."""

    def test_norm_pdf(self, we):
        assert _cost_of(flopscope.stats.norm.pdf, numpy.random.rand(100)) == 27 * 100

    def test_norm_cdf(self, we):
        assert _cost_of(flopscope.stats.norm.cdf, numpy.random.rand(100)) == 48 * 100

    def test_norm_ppf(self, we):
        assert (
            _cost_of(flopscope.stats.norm.ppf, numpy.random.rand(100) * 0.98 + 0.01)
            == 83 * 100
        )

    def test_uniform_pdf(self, we):
        assert _cost_of(flopscope.stats.uniform.pdf, numpy.random.rand(100)) == 100

    def test_uniform_cdf(self, we):
        assert (
            _cost_of(flopscope.stats.uniform.cdf, numpy.random.rand(100)) == 4 * 100
        )  # sub+div+2 select

    def test_uniform_ppf(self, we):
        assert _cost_of(flopscope.stats.uniform.ppf, numpy.random.rand(100)) == 100

    def test_expon_pdf(self, we):
        # old: == 100 (cost_per_elem=1, weight=16.0); now: 22*100 (composite, weight 1.0)
        assert _cost_of(flopscope.stats.expon.pdf, numpy.random.rand(100)) == 22 * 100

    def test_cauchy_pdf(self, we):
        assert (
            _cost_of(flopscope.stats.cauchy.pdf, numpy.random.rand(100)) == 6 * 100
        )  # pure-arithmetic: 6 FLOPs/elem

    def test_logistic_cdf(self, we):
        # old: == 100 (cost_per_elem=1, weight=16.0); now: 21*100 (composite, weight 1.0)
        assert (
            _cost_of(flopscope.stats.logistic.cdf, numpy.random.rand(100)) == 21 * 100
        )

    def test_laplace_ppf(self, we):
        assert (
            _cost_of(flopscope.stats.laplace.ppf, numpy.random.rand(100) * 0.98 + 0.01)
            == 51 * 100  # composite: two eager log branches + edge selects
        )

    def test_lognorm_pdf(self, we):
        assert (
            _cost_of(
                flopscope.stats.lognorm.pdf,
                numpy.abs(numpy.random.rand(100)) + 0.1,
                0.5,
            )
            == 62 * 100  # composite: log + exp + arithmetic
        )

    def test_truncnorm_cdf(self, we):
        # old: == 100 (cost_per_elem=1, weight=16.0); now: 51*100 (composite, weight 1.0)
        assert (
            _cost_of(flopscope.stats.truncnorm.cdf, numpy.random.rand(100), -2, 2)
            == 51 * 100
        )

    def test_scalar_input(self, we):
        """Scalar input should charge cost_per_elem FLOPs (norm.pdf=27)."""
        assert _cost_of(flopscope.stats.norm.pdf, 0.0) == 27


# ---------------------------------------------------------------------------
# Issue #69 — pinned constants for fixes that changed cost formulas.
# These complement tests/test_issue_69_cost_parity.py by pinning the
# *formula* (not the parity) so accidental future tweaks fail loudly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,n,expected",
    [
        ((1000,), 1, 999),  # numel - 1
        ((1000,), 3, 2994),  # 3*1000 - 3*4//2 = 2994
        ((1000,), 10, 9945),  # 10*1000 - 10*11//2 = 9945
        ((50, 50), 1, 2450),  # along last axis: 50 * (50-1) = 2450
    ],
)
def test_diff_cost_pinned(shape, n, expected, we):
    a = we.asarray(numpy.zeros(shape))
    assert _cost_of(we.diff, a, n=n) == expected


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((10,), 16),  # one axis: 2 * 10 * max(10-2, 0) // 10 = 2*10*8//10 = 16
        ((50, 50), 9600),  # per-axis: 2*2500*48//50 = 4800; two axes: 9600
        ((20, 20, 20), 43200),  # per-axis: 2*8000*18//20 = 14400; three axes: 43200
    ],
)
def test_gradient_cost_pinned(shape, expected, we):
    f = we.asarray(numpy.zeros(shape))
    assert _cost_of(we.gradient, f) == expected


@pytest.mark.parametrize("size,expected", [(100, 1300), (1000, 13000)])
def test_unwrap_cost_pinned(size, expected, we):
    a = we.asarray(numpy.zeros(size))
    assert _cost_of(we.unwrap, a) == expected


def test_convolve_cost_pinned(we):
    a = we.asarray(numpy.zeros(200))
    v = we.asarray(numpy.zeros(50))
    # 2*200*50 - 200 - 50 = 19750
    assert _cost_of(we.convolve, a, v) == 19750


def test_cross_cost_pinned(we):
    a = we.asarray(numpy.zeros((100, 3)))
    b = we.asarray(numpy.zeros((100, 3)))
    # 3 ops per output element; output.size = 100*3 = 300, formula: a.shape[0]*3*3 = 900
    assert _cost_of(we.cross, a, b) == 900


def test_matrix_power_cost_pinned(we):
    a = we.asarray(numpy.eye(20))
    # n=4: 2 squarings, each matmul_cost(20,20,20) = 2*20^3 - 20^2 = 15600
    # total = 2 * 15600 = 31200
    assert _cost_of(we.linalg.matrix_power, a, 4) == 31200
