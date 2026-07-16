"""Four-factor billing: charged = int(flop_cost * dtype_rate * complex_factor * weight)."""

import numpy as np
import pytest

import flopscope as f
from flopscope._weights import load_weights


def _charge(op_name, flop_cost, dtypes, override=None) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            op_name,
            flop_cost=flop_cost,
            subscripts=None,
            shapes=(),
            dtypes=dtypes,
            complex_factor_override=override,
        ):
            pass
        return b.flops_used


def test_unit_mode_real_dtypes_bill_flop_cost():
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 10


def test_unit_mode_complex_factor_still_applies():
    # complex_factor is math, not policy: active even under unit rates/weights
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 60
    assert _charge("add", 10, (np.dtype("complex128"),)) == 20


def test_production_rates_compose():
    load_weights()
    assert _charge("multiply", 10, (np.dtype("float32"),)) == 10
    assert _charge("multiply", 10, (np.dtype("float64"),)) == 20
    assert _charge("multiply", 10, (np.dtype("complex64"),)) == 60
    assert _charge("multiply", 10, (np.dtype("complex128"),)) == 120
    assert _charge("add", 10, (np.dtype("complex128"),)) == 40


def test_override_bypasses_registry_factor():
    assert (
        _charge("einsum", 960, (np.dtype("complex128"),), override=3968 / 960) == 3968
    )


def test_dtype_neutral_and_none_rejected():
    assert _charge("einsum_path", 1, ()) == 1  # declared neutral
    with pytest.raises(TypeError):
        _charge("einsum_path", 1, None)  # dtypes= is required; None now rejected


def test_resolved_dtype_recorded():
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            "multiply",
            flop_cost=1,
            subscripts=None,
            shapes=(),
            dtypes=(np.dtype("float32"), np.dtype("float64")),
        ):
            pass
        assert b._op_log[-1].resolved_dtype == "float64"


def test_non_numeric_dtype_bills_neutral_rate():
    # object/str/datetime are not floating-point arithmetic: they bill at the
    # neutral rate 1.0 (numpy-compat requires them functional; no precision
    # exploit is possible through non-numeric kinds).
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            "multiply",
            flop_cost=10,
            subscripts=None,
            shapes=(),
            dtypes=(np.dtype("object"),),
        ):
            pass
        assert b.flops_used == 10  # rate 1.0, factor 1.0


@pytest.mark.skipif(
    not hasattr(np, "float128"), reason="float128 not available on this platform"
)
def test_float128_bills_extended_width_rate():
    # Extended precision is priced, not banned: float128 bills 2x float64
    # (rate 4.0 in fp32 units). Width packing through its mantissa still
    # loses: two f32-payload products packed into one float128 multiply cost
    # 4.0 vs the honest 2 x 1.0 = 2.0, and one f64-payload product costs 4.0
    # vs the honest 2.0.
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with b.deduct(
            "multiply",
            flop_cost=10,
            subscripts=None,
            shapes=(),
            dtypes=(np.dtype("float128"),),
        ):
            pass
        assert b.flops_used == 40  # 10 * rate 4.0


# ---------------------------------------------------------------------------
# Task 6: pointwise/reduction factories declare their billing dtypes
# ---------------------------------------------------------------------------

import flopscope.numpy as fnp  # noqa: E402

_a32 = fnp.asarray(np.ones(10, dtype=np.float32))
_b32 = fnp.asarray(np.ones(10, dtype=np.float32))
_z = fnp.asarray(np.ones(10, dtype=np.complex128))


def _cost(fn) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def test_elementwise_multiply_complex_bills_six_per_element():
    assert _cost(lambda: fnp.multiply(_z, _z)) == 60  # unit rates; factor 6


def test_elementwise_add_complex_bills_two_per_element():
    assert _cost(lambda: fnp.add(_z, _z)) == 20


def test_nep50_scalar_keeps_float32_billing():
    load_weights()  # production rates: f32=1.0, f64=2.0
    assert _cost(lambda: fnp.multiply(_a32, 2.0)) == 10  # stays float32
    assert _cost(lambda: fnp.multiply(_a32, _b32)) == 10
    _a64 = fnp.asarray(np.ones(10, dtype=np.float64))
    assert _cost(lambda: fnp.multiply(_a64, _a64)) == 20


def test_explicit_dtype_kwarg_raises_billing_dtype():
    load_weights()
    assert _cost(lambda: fnp.add(_a32, _b32, dtype=np.float64)) == 20


def test_reduction_sum_complex_bills_factor_two():
    # ratio-based: robust to the reduction skeleton's exact flop_cost formula
    _r = fnp.asarray(np.ones(10, dtype=np.float64))
    assert _cost(lambda: fnp.sum(_z)) == 2 * _cost(lambda: fnp.sum(_r))


# ---------------------------------------------------------------------------
# Task 6b: generic ufunc-method dispatch (np.<ufunc>.<method> on a
# FlopscopeArray) inherits its base ufunc's complex factor via fallback.
#
# ``multiply``/``add`` are routed to a dedicated fast path
# (FlopscopeArray._REDUCE_TO_WHEST) for ``.reduce``/``.accumulate``, so
# those methods on those two ufuncs do NOT exercise the generic dispatch
# sites this task changes. ``.outer`` always uses the generic path
# regardless of ufunc, and ``subtract`` is never whest-routed — so
# ``multiply.outer`` and ``subtract.reduce``/``subtract.accumulate`` are
# the calls that genuinely reach _counted_ufunc_outer /
# _counted_ufunc_reduce_generic / _counted_ufunc_accumulate_generic.
# ---------------------------------------------------------------------------

_r10 = fnp.asarray(np.ones(10, dtype=np.float64))


def test_generic_ufunc_outer_complex_bills_base_ufunc_factor():
    # multiply.outer always uses the generic path; base factor is multiply's (6).
    assert _cost(lambda: np.multiply.outer(_z, _z)) == 6 * _cost(
        lambda: np.multiply.outer(_r10, _r10)
    )


def test_generic_ufunc_reduce_complex_runs_and_bills_base_ufunc_factor():
    # subtract is not whest-routed, so subtract.reduce reaches the generic
    # fallback. Before this task's fix, dtypes=None there billed complex
    # inputs as dtype-neutral (factor 1) instead of raising OR correctly
    # scaling — this is the regression the naive "just add dtypes=" would
    # have caused (fail-closed RAISE on every complex generic-ufunc-method
    # call). Asserting it both runs AND bills factor 2 (subtract's factor)
    # covers that regression.
    assert _cost(lambda: np.subtract.reduce(_z)) == 2 * _cost(
        lambda: np.subtract.reduce(_r10)
    )


def test_generic_ufunc_accumulate_complex_runs_and_bills_base_ufunc_factor():
    assert _cost(lambda: np.subtract.accumulate(_z)) == 2 * _cost(
        lambda: np.subtract.accumulate(_r10)
    )


# ---------------------------------------------------------------------------
# Task 7: exact complex billing for the einsum/contraction family
# ---------------------------------------------------------------------------

_A32 = fnp.asarray(np.ones((8, 8), dtype=np.float32))
_A64 = fnp.asarray(np.ones((8, 8), dtype=np.float64))
_Ac64 = fnp.asarray(np.ones((8, 8), dtype=np.complex64))
_Ac128 = fnp.asarray(np.ones((8, 8), dtype=np.complex128))


def test_matmul_exact_complex_billing():
    # (8,8)@(8,8): flop_cost = 2*512 - 64 = 960; complex exact = 3968
    load_weights()
    assert _cost(lambda: fnp.matmul(_A32, _A32)) == 960  # f32 rate 1.0
    assert _cost(lambda: fnp.matmul(_A64, _A64)) == 1920  # f64 rate 2.0
    assert _cost(lambda: fnp.matmul(_Ac64, _Ac64)) == 3968  # c64: exact factor
    assert _cost(lambda: fnp.matmul(_Ac128, _Ac128)) == 7936  # c128: exact * 2.0


def test_einsum_elementwise_alias_matches_multiply():
    # einsum('i,i->i') must bill like multiply (factor 6), not a flat 4
    z = fnp.asarray(np.ones(10, dtype=np.complex128))
    assert _cost(lambda: fnp.einsum("i,i->i", z, z)) == _cost(
        lambda: fnp.multiply(z, z)
    )


def test_mixed_real_complex_matmul_bills_complex():
    load_weights()
    assert _cost(lambda: fnp.matmul(_A64, _Ac128)) == 7936  # result_type -> complex128


def test_einsum_path_stays_dtype_neutral():
    # einsum_path does no value arithmetic; a complex operand must not raise
    # and must bill the fixed bookkeeping cost regardless of dtype.
    assert _cost(lambda: fnp.einsum_path("ij,jk->ik", _A64, _A64)) == _cost(
        lambda: fnp.einsum_path("ij,jk->ik", _Ac128, _Ac128)
    )


def test_dot_exact_complex_billing_matches_matmul():
    load_weights()
    assert _cost(lambda: fnp.dot(_Ac128, _Ac128)) == _cost(
        lambda: fnp.matmul(_Ac128, _Ac128)
    )


def test_inner_exact_complex_billing():
    # inner('i,i->') on length-10 vectors: flop_cost = 2*10-1 = 19,
    # mu=10 (mults), adds=9 -> complex real total = 6*10 + 2*9 = 78.
    load_weights()
    z = fnp.asarray(np.ones(10, dtype=np.complex128))
    assert _cost(lambda: fnp.inner(z, z)) == 78 * 2  # c128 rate 2.0


def test_vdot_exact_complex_billing():
    # vdot conjugates one operand but is still routed through the same
    # "i,i->" accumulation skeleton as inner, so the exact total matches.
    load_weights()
    z = fnp.asarray(np.ones(10, dtype=np.complex128))
    assert _cost(lambda: fnp.vdot(z, z)) == 78 * 2  # c128 rate 2.0


def test_tensordot_full_inner_exact_complex_billing():
    # Full-inner tensordot (axes=ndim) takes the fast path that shares the
    # einsum accumulation object with matmul/dot; billing must be exact too.
    # "ab,ab->" on (8,8): flop_cost = 127, mu=64, adds=63 -> complex real
    # total = 6*64 + 2*63 = 510.
    load_weights()
    z = fnp.asarray(np.ones((8, 8), dtype=np.complex128))
    assert _cost(lambda: fnp.tensordot(z, z, axes=2)) == 510 * 2  # c128 rate 2.0


def test_tensordot_partial_contraction_exact_complex_billing():
    # Partial contraction (axes=1) routes through the general fallback branch
    # that still has an einsum accumulation object (no oversized symmetry,
    # rank well under 52) -- must also bill the exact complex factor, not a
    # flat/neutral rate. "ab,bd->ad" on (8,8)x(8,8): flop_cost = 960 (same
    # skeleton as the matmul case), complex real total = 3968.
    load_weights()
    z = fnp.asarray(np.ones((8, 8), dtype=np.complex128))
    assert _cost(lambda: fnp.tensordot(z, z, axes=1)) == 3968 * 2  # c128 rate 2.0


# --- Task 8a: _array_ops.py dtype declarations ---------------------------


def test_array_ops_creation_bills_output_dtype():
    # Cost-bearing creation declares the OUTPUT dtype: int64 (rate 2.0) bills
    # twice int32 (rate 1.0) for the same element count.
    load_weights()
    c32 = _cost(lambda: fnp.arange(100, dtype=np.int32))
    c64 = _cost(lambda: fnp.arange(100, dtype=np.int64))
    assert c32 > 0
    assert c64 == 2 * c32


def test_array_ops_free_op_bills_zero_regardless_of_dtype():
    # flop_cost=0 data-movement ops (reshape) bill 0 for any dtype, incl complex.
    load_weights()
    zc = fnp.asarray(np.ones(12, dtype=np.complex128))
    assert _cost(lambda: fnp.reshape(zc, (3, 4))) == 0


def test_astype_bills_heavier_of_src_and_dst_rate():
    # astype (weight 1.0, cost=numel for value-changing casts) bills at the
    # HEAVIER of the source and destination rate -- it reads the source and
    # writes the destination, so the pricier side dominates. Neither source-only
    # nor result_type is correct: source-only under-bills when dst is pricier
    # (complex64->float64), result_type over-bills a cross-kind narrowing
    # (float32+int32 promotes to float64).
    load_weights()
    a64 = fnp.asarray(np.ones(1000, dtype=np.float64))
    a32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    c64 = fnp.asarray(np.ones(1000, dtype=np.complex64))
    c128 = fnp.asarray(np.ones(1000, dtype=np.complex128))
    # source pricier than dest:
    assert _cost(lambda: fnp.astype(a64, np.int32)) == 2000
    assert _cost(lambda: fnp.astype(a32, np.int32)) == 1000  # no over-charge
    # dest pricier than source (the fixed under-bill): complex64->float64 must
    # match complex128->float64 -- same float64 output, same work.
    assert _cost(lambda: fnp.astype(c64, np.float64)) == 2000
    assert _cost(lambda: fnp.astype(c128, np.float64)) == 2000
    assert _cost(lambda: fnp.astype(a32, np.int64)) == 2000  # real-only dst pricier


def test_complex_movement_and_creation_do_not_raise():
    # Free / data-movement ops on complex values must not fail closed; they
    # bill at factor 1.0 (weight 0 -> 0 FLOPs), never raise UnsupportedDtypeError.
    load_weights()
    zc = fnp.asarray(np.ones(100, dtype=np.complex128))
    assert _cost(lambda: fnp.asarray(np.ones(50, dtype=np.complex128))) == 0
    assert _cost(lambda: fnp.ravel(zc)) == 0
    assert _cost(lambda: fnp.broadcast_to(zc, (2, 100))) == 0
    # astype complex->real is value-changing (cost=numel) billed at the heavier
    # of the complex128 / float64 rates (both 2.0), factor 1.0: 100 * 2.0 = 200.
    # Both the function form and the .astype() method form must not raise.
    assert _cost(lambda: fnp.astype(zc, np.float64)) == 200
    assert _cost(lambda: zc.astype(np.float64)) == 200


# ---------------------------------------------------------------------------
# Task 8b: sorting/counting/polynomial/window/unwrap/symmetric dtype
# declarations (_sorting_ops.py, _counting_ops.py, _polynomial.py,
# _window.py, _unwrap.py, _symmetric.py).
# ---------------------------------------------------------------------------


def test_sort_complex_bills_double_real_factor():
    # sort has registry complex_factor=2.0 (lexicographic real/imag compare);
    # unit dtype rates isolate the factor from the dtype-rate table.
    assert _cost(lambda: fnp.sort(_z)) == 2 * _cost(lambda: fnp.sort(_r10))


def test_unique_complex_bills_double_real_factor():
    # unique shares the sort-based cost family; same complex_factor=2.0.
    assert _cost(lambda: fnp.unique(_z)) == 2 * _cost(lambda: fnp.unique(_r10))


def test_polyval_complex_bills_quadruple_real_factor():
    # polyval has registry complex_factor=4.0.
    p_r = fnp.asarray(np.ones(5, dtype=np.float64))
    p_c = fnp.asarray(np.ones(5, dtype=np.complex128))
    x_r = fnp.asarray(np.ones(5, dtype=np.float64))
    assert _cost(lambda: fnp.polyval(p_c, x_r)) == 4 * _cost(
        lambda: fnp.polyval(p_r, x_r)
    )


def test_symmetrize_complex_bills_double_real_factor():
    # symmetrize has registry complex_factor=2.0.
    g = f.SymmetryGroup.symmetric(axes=(0, 1))
    r = fnp.asarray(np.ones((4, 4), dtype=np.float64))
    c = fnp.asarray(np.ones((4, 4), dtype=np.complex128))
    assert _cost(lambda: f.symmetrize(c, symmetry=g)) == 2 * _cost(
        lambda: f.symmetrize(r, symmetry=g)
    )


def test_trace_fp64_bills_double_fp32():
    # dtype-RATE test (not complex_factor): under production rates float64
    # is priced 2x float32 for the same element count.
    load_weights()
    m32 = fnp.asarray(np.eye(4, dtype=np.float32))
    m64 = fnp.asarray(np.eye(4, dtype=np.float64))
    assert _cost(lambda: fnp.trace(m64)) == 2 * _cost(lambda: fnp.trace(m32))


def test_unwrap_complex_fails_closed_before_numpy_call():
    # unwrap is registry-"illegal" for complex (phase unwrap is undefined for
    # complex-valued input); billing must raise before the numpy call runs,
    # instead of leaking numpy's own TypeError from inside the ufunc chain.
    from flopscope.errors import UnsupportedDtypeError

    zc = fnp.asarray(np.ones(10, dtype=np.complex128))
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        with pytest.raises(UnsupportedDtypeError):
            fnp.unwrap(zc)
        assert b.flops_used == 0


def test_window_ops_bill_their_float64_output():
    # Window ops take an int length but do genuine float64 arithmetic (numpy
    # windows always return float64), so they bill that width — consistent
    # with the random samplers, which are structurally identical
    # (scalar in, fixed-float64 out, real transcendental work).
    load_weights()
    assert _cost(lambda: fnp.bartlett(8)) == 64  # 4*8 * float64 rate 2.0
    assert _cost(lambda: fnp.fft.fftfreq(100)) == 200  # n * rate 2.0


def test_polyint_k_operand_is_billed():
    # polyint's `k` (integration constants) is a second operand: complex k
    # yields a complex result and must bill the 2x complex factor, not dodge it.
    load_weights()
    p = fnp.asarray(np.ones(5, dtype=np.float64))
    real_k = _cost(lambda: fnp.polyint(p, m=1, k=[5.0]))
    cplx_k = _cost(lambda: fnp.polyint(p, m=1, k=[1 + 2j]))
    assert cplx_k == 2 * real_k


def test_histogram_array_bins_dtype_is_billed():
    # Array bin edges are a second operand numpy promotes across; an fp64 or
    # complex edge array must not bill the same as an fp32 one.
    load_weights()
    d = fnp.asarray(np.ones(100, dtype=np.float32))
    b32 = np.linspace(0, 2, 5).astype(np.float32)
    # histogram's mirrored signature types bins as int|Sequence[int]|str; numpy
    # (and the runtime) accept float/array edges, so ignore the narrow hint.
    c32 = _cost(lambda: fnp.histogram(d, bins=fnp.asarray(b32)))  # type: ignore[arg-type]
    c64 = _cost(lambda: fnp.histogram(d, bins=fnp.asarray(b32.astype(np.float64))))  # type: ignore[arg-type]
    assert c32 > 0 and c64 == 2 * c32
    # int bins is a count, not data -> dtype irrelevant. Pin the exact value so
    # a future change to the int-bins path can't silently drift:
    # 100 elems * ceil(log2(10))=4 = 400.
    assert _cost(lambda: fnp.histogram(d, bins=10)) == 400


def test_histogram2d_histogramdd_array_bins_dtype_is_billed():
    # histogram2d/histogramdd fold array bin-edge dtypes in too; fp64 edges must
    # bill 2x fp32, while int-count bins stay unchanged.
    load_weights()
    x = fnp.asarray(np.ones(100, dtype=np.float32))
    y = fnp.asarray(np.ones(100, dtype=np.float32))
    e32 = np.linspace(0, 2, 5).astype(np.float32)
    e64 = e32.astype(np.float64)
    h2_32 = _cost(
        lambda: fnp.histogram2d(x, y, bins=[fnp.asarray(e32), fnp.asarray(e32)])
    )
    h2_64 = _cost(
        lambda: fnp.histogram2d(x, y, bins=[fnp.asarray(e64), fnp.asarray(e64)])
    )
    assert h2_32 > 0 and h2_64 == 2 * h2_32
    s = fnp.asarray(np.ones((100, 2), dtype=np.float32))
    hd_32 = _cost(lambda: fnp.histogramdd(s, bins=[fnp.asarray(e32), fnp.asarray(e32)]))
    hd_64 = _cost(lambda: fnp.histogramdd(s, bins=[fnp.asarray(e64), fnp.asarray(e64)]))
    assert hd_32 > 0 and hd_64 == 2 * hd_32


# ---------------------------------------------------------------------------
# Task 8c: fft/linalg/random/stats dtype declarations
# ---------------------------------------------------------------------------


def test_fft_complex_bills_by_rate_not_double_charged():
    # fft.* registry complex_factor is 1.0 (priced-in: the 5*n*ceil(log2(n))
    # formula already counts complex real-FLOPs), so a complex128 input must
    # bill exactly the dtype-RATE ratio over a float32 input of the same
    # shape (2.0 under production rates) -- NOT an extra factor stacked on
    # top of the rate (which would show up as a 4x ratio instead of 2x).
    load_weights()
    n = 16
    r32 = fnp.asarray(np.ones(n, dtype=np.float32))
    c128 = fnp.asarray(np.ones(n, dtype=np.complex128))
    real_cost = _cost(lambda: fnp.fft.fft(r32))
    complex_cost = _cost(lambda: fnp.fft.fft(c128))
    assert real_cost > 0
    assert complex_cost == 2 * real_cost


def test_linalg_inv_complex_bills_quadruple_real():
    # linalg.* registry complex_factor is 4.0.
    load_weights()
    a64 = fnp.asarray(np.eye(4, dtype=np.float64))
    ac128 = fnp.asarray(np.eye(4, dtype=np.complex128))
    real_cost = _cost(lambda: fnp.linalg.inv(a64))
    complex_cost = _cost(lambda: fnp.linalg.inv(ac128))
    assert real_cost > 0
    assert complex_cost == 4 * real_cost


def test_linalg_solve_complex_bills_quadruple_real():
    # Two-operand linalg site: both a and b must be declared.
    load_weights()
    a64 = fnp.asarray(np.eye(4, dtype=np.float64))
    b64 = fnp.asarray(np.ones(4, dtype=np.float64))
    ac128 = fnp.asarray(np.eye(4, dtype=np.complex128))
    bc128 = fnp.asarray(np.ones(4, dtype=np.complex128))
    real_cost = _cost(lambda: fnp.linalg.solve(a64, b64))
    complex_cost = _cost(lambda: fnp.linalg.solve(ac128, bc128))
    assert real_cost > 0
    assert complex_cost == 4 * real_cost


def test_random_standard_normal_dtype_bills_by_rate():
    # Generator.standard_normal(dtype=...) must declare the actual OUTPUT
    # dtype (read off the already-computed result), so float64 bills 2x
    # float32 under production dtype rates.
    load_weights()
    rng = fnp.random.default_rng(0)
    cost_f32 = _cost(lambda: rng.standard_normal(100, dtype=np.float32))
    cost_f64 = _cost(lambda: rng.standard_normal(100, dtype=np.float64))
    assert cost_f32 > 0
    assert cost_f64 == 2 * cost_f32


def test_random_movement_ops_on_complex_do_not_raise():
    # permutation/shuffle/choice(+Generator.permuted) relocate arbitrary
    # (possibly complex) caller-supplied data rather than sampling a new
    # value from a distribution, so they are movement ops (complex_factor
    # 1.0) that bill dtype-neutral. numpy permutes/shuffles/selects complex
    # arrays fine; these sites bill dtypes=() so a complex operand neither
    # raises nor incurs a spurious width multiplier. This pins the whole
    # surface (module-level fns + Generator.permuted, which the dedicated
    # neutrality test above does not separately exercise).
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        zc = np.ones(8, dtype=np.complex128)
        fnp.random.permutation(zc)
        fnp.random.shuffle(zc.copy())
        fnp.random.choice(zc, size=2, replace=False)
        rng = fnp.random.default_rng(0)
        rng.permutation(zc)
        rng.choice(zc, size=2, replace=False)
        rng.permuted(zc)
        rng.shuffle(zc.copy())


def test_stats_norm_pdf_bills_forced_float64():
    # stats.<dist>.pdf/cdf/ppf coerce the input to float64 before billing
    # (_deduct_and_call does `np.asarray(x, dtype=np.float64)`), so the
    # billing dtype is always float64 regardless of the caller's input dtype.
    load_weights()
    from flopscope import stats as fstats

    x32 = fnp.asarray(np.zeros(10, dtype=np.float32))
    x64 = fnp.asarray(np.zeros(10, dtype=np.float64))
    assert _cost(lambda: fstats.norm.pdf(x32)) == _cost(lambda: fstats.norm.pdf(x64))


def test_linalg_tensorsolve_and_solve_asymmetric_complex_operand():
    # A real `a` with a complex `b` still yields a complex result, so both
    # operands must be billed -> the 4x complex factor applies even when only
    # the RHS is complex. (Guards the multi-operand declaration; a symmetric
    # real->complex flip can't distinguish "both declared" from "one suffices".)
    load_weights()
    a = fnp.asarray(np.eye(6, dtype=np.float64))
    b_r = fnp.asarray(np.ones(6, dtype=np.float64))
    b_c = fnp.asarray(np.ones(6, dtype=np.complex128))
    assert _cost(lambda: fnp.linalg.solve(a, b_c)) == 4 * _cost(
        lambda: fnp.linalg.solve(a, b_r)
    )
    # tensorsolve delegates to solve; same asymmetric property.
    ta = fnp.asarray(np.eye(6, dtype=np.float64).reshape(6, 6))
    tb_r = fnp.asarray(np.ones(6, dtype=np.float64))
    tb_c = fnp.asarray(np.ones(6, dtype=np.complex128))
    assert _cost(lambda: fnp.linalg.tensorsolve(ta, tb_c)) == 4 * _cost(
        lambda: fnp.linalg.tensorsolve(ta, tb_r)
    )


def test_linalg_trace_accumulator_dtype_is_billed():
    # linalg.trace(real, dtype=complex) accumulates in complex -> must bill the
    # complex factor via the dtype= kwarg, not just the input dtype.
    load_weights()
    m = fnp.asarray(np.ones((8, 8), dtype=np.float64))
    real = _cost(lambda: fnp.linalg.trace(m))
    cplx = _cost(lambda: fnp.linalg.trace(m, dtype=np.complex128))
    assert cplx == 4 * real


# ---------------------------------------------------------------------------
# Task 10: exploit-kill and resolved-dtype invariant tests
#
# These encode the two reported undercounting exploits as assertions and
# prove the dtype-aware billing closes them: the load-bearing check in each
# is exploit_cost >= honest_cost (packing must not be profitable), with the
# concrete values pinned via derivation comments so the invariant stays
# meaningful even if a future rate/weight tweak shifts the literals.
# ---------------------------------------------------------------------------


def test_complex_packing_is_not_profitable_elementwise():
    # Complex-packing trick: bundle two independent real payloads (y1, y2)
    # into one complex64 array's real/imag lanes and run a single complex
    # multiply against the shared operand x, hoping to recover both x*y1
    # and x*y2 for less than the price of the two honest real multiplies.
    load_weights()
    x = fnp.asarray(np.ones(100, dtype=np.float32))
    y1 = fnp.asarray(np.ones(100, dtype=np.float32))
    y2 = fnp.asarray(np.ones(100, dtype=np.float32))
    packed = fnp.asarray(np.ones(100, dtype=np.complex64))
    honest = _cost(lambda: (fnp.multiply(x, y1), fnp.multiply(x, y2)))
    exploit = _cost(lambda: fnp.multiply(x, packed))
    # honest: 2 * (100 * rate=1.0(f32) * complex_factor=1.0(real) * weight=1.0) = 200
    # exploit: result_type(f32, c64)=c64; 100 * rate=1.0(c64) * complex_factor=6.0
    #          (multiply's registry factor) * weight=1.0 = 600
    assert honest == 200
    assert exploit == 600
    assert exploit >= honest  # packing loses 3x, not free


def test_complex_packing_is_not_profitable_matmul():
    # Same packing trick one level up: A @ (B1 + i*B2) in place of the two
    # honest matmuls A@B1 and A@B1.
    load_weights()
    A = fnp.asarray(np.ones((8, 8), dtype=np.float32))
    B1 = fnp.asarray(np.ones((8, 8), dtype=np.float32))
    Bc = fnp.asarray(np.ones((8, 8), dtype=np.complex64))
    honest = _cost(lambda: (fnp.matmul(A, B1), fnp.matmul(A, B1)))
    exploit = _cost(lambda: fnp.matmul(A, Bc))
    # honest: 2 * 960 ((8,8)@(8,8) f32, rate 1.0; matches test_matmul_exact_
    #         complex_billing's f32 pin) = 1920
    # exploit: result_type(f32, c64)=c64. The einsum accumulation (mu, adds)
    #          depends only on shapes/subscripts, not on which operand is
    #          nominally real, so this bills the SAME exact complex total as
    #          a genuine complex64 @ complex64 matmul: 3968 (matches test_
    #          matmul_exact_complex_billing's c64 pin).
    assert honest == 1920
    assert exploit == 3968
    assert exploit >= honest  # packing loses ~2x, not free


def test_width_packing_is_break_even_or_losing():
    # Width-packing trick: pack two fp32-precision multiplies into one fp64
    # multiply (e.g. lower/upper 32 bits of a float64 lane each carrying an
    # independent fp32 payload), hoping fp64 bills like a single fp32 op.
    load_weights()
    a32 = fnp.asarray(np.ones(100, dtype=np.float32))
    a64 = fnp.asarray(np.ones(100, dtype=np.float64))
    two_f32 = _cost(lambda: (fnp.multiply(a32, a32), fnp.multiply(a32, a32)))
    one_f64 = _cost(lambda: fnp.multiply(a64, a64))
    # two_f32: 2 * (100 * rate=1.0(f32) * weight=1.0) = 200
    # one_f64: 100 * rate=2.0(f64) * weight=1.0 = 200
    assert two_f32 == 200
    assert one_f64 == 200
    assert one_f64 >= two_f32  # break-even before any pack/unpack overhead


def test_output_downcast_does_not_discount():
    # fnp.matmul(a, b) takes no dtype= kwarg (fnp.matmul(a, b, dtype=np.int8)
    # raises TypeError -- confirmed while writing this test), so there is no
    # literal "requested narrow output" to pass. The invariant this locks
    # instead: billing resolves via np.result_type over the declared operand
    # dtypes, not the narrowest-looking one. int32 (rate 1.0) combined with
    # float32 (rate 1.0) numpy-promotes to float64 (rate 2.0) for the actual
    # compute (np.result_type(int32, float32) == float64), and the bill must
    # track that promotion exactly -- matching a direct float64 matmul --
    # rather than discounting to either input's own (lighter) rate.
    load_weights()
    a = fnp.asarray(np.ones((8, 8), dtype=np.int32))
    b = fnp.asarray(np.ones((8, 8), dtype=np.float32))
    plain = _cost(lambda: fnp.matmul(a, b))  # resolves float64
    f64 = fnp.asarray(np.ones((8, 8), dtype=np.float64))
    via_f64 = _cost(lambda: fnp.matmul(f64, f64))
    assert plain == 1920
    assert via_f64 == 1920
    assert plain == via_f64  # no discount for the int32/float32-looking mix


def test_requested_output_downcast_does_not_discount():
    # The other half of the resolved-dtype rule: an EXPLICIT narrow dtype=
    # request must not discount the bill. multiply folds dtype= into the
    # billing tuple, and np.result_type still resolves the wider operand
    # dtype, so asking for float32 output on float64 operands still bills at
    # the float64 rate (the real compute precision), not float32's.
    load_weights()
    w = fnp.asarray(np.ones(100, dtype=np.float64))
    plain = _cost(lambda: fnp.multiply(w, w))
    downcast_requested = _cost(lambda: fnp.multiply(w, w, dtype=np.float32))
    assert plain == 200
    assert downcast_requested == plain  # requesting a narrow dtype= is not a discount


def test_astype_to_complex_charges_and_back_charges():
    # Widening real->complex astype is a safe (lossless) cast: free. The
    # reverse, narrowing complex->real, discards the imaginary lane and is
    # charged.
    load_weights()
    r = fnp.asarray(np.ones(100, dtype=np.float64))
    z = fnp.asarray(np.ones(100, dtype=np.complex128))
    assert _cost(lambda: r.astype(np.complex128)) == 0  # widening: safe cast
    back = _cost(lambda: z.astype(np.float64))  # lossy: numel * rate * factor
    # astype's registry entry declares no complex_factor classification, so
    # complex_factor_for()'s default of 1.0 applies (relocation, not
    # arithmetic); the dtype rate is the heavier of source/dest
    # (heavier_billing_dtype(complex128, float64), tied at rate 2.0) ->
    # 100 * 2.0 * 1.0 = 200. (The brief's draft literal was
    # int(100 * 2.0 * 2.0) == 400, assuming a second dtype-like 2x factor
    # stacked on top of the rate; the implemented astype has no such factor.
    # 200 also matches the existing sibling pin in
    # test_complex_movement_and_creation_do_not_raise, so this isn't a new
    # number -- it's consistent with billing already locked in by Task 8a.)
    assert back == 200


# ---------------------------------------------------------------------------
# Task 11: real/imag are free-tier component extraction (view, no arithmetic)
# ---------------------------------------------------------------------------


def test_real_imag_bill_zero_flops():
    # np.real(z)/np.imag(z) extract a component of a complex value -- a
    # strided VIEW for real, a constant-fill for imag on real input -- no
    # floating-point arithmetic is performed. The fix hardcodes flop_cost=0
    # at the call site (not merely a zero weight), so this holds under BOTH
    # unit weights (default here -- conftest resets weights per test) and
    # production weights.
    assert _cost(lambda: fnp.real(_z)) == 0
    assert _cost(lambda: fnp.imag(_z)) == 0
    load_weights()
    assert _cost(lambda: fnp.real(_z)) == 0
    assert _cost(lambda: fnp.imag(_z)) == 0


def test_real_imag_still_return_correct_components():
    # Billing changed; behavior must not. real/imag still extract the
    # correct component values (functional correctness, not just cost).
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        z = fnp.asarray(np.array([3.0 + 4.0j, -1.0 - 2.0j]))
        r = fnp.real(z)
        i = fnp.imag(z)
    assert np.array_equal(np.asarray(r), [3.0, -1.0])
    assert np.array_equal(np.asarray(i), [4.0, -2.0])


def test_mixed_unpromotable_operands_bill_without_raising():
    # numpy's logical ufuncs accept any dtype mix (no common promotion); the
    # billing path must not raise where numpy succeeds. Bills at the heaviest
    # individual operand rate (float64 -> 2.0).
    load_weights()
    td = fnp.asarray(np.array([1, 2], dtype="timedelta64[s]"))
    fl = fnp.asarray(np.ones(2, dtype=np.float64))
    assert _cost(lambda: np.logical_and(td, fl)) == 4  # 2 elems * rate 2.0


def test_random_movement_ops_are_dtype_neutral_and_run_on_complex():
    # shuffle/permutation/choice relocate caller data (movement, factor 1.0).
    # Their shape[axis] cost counts the dtype-INDEPENDENT Fisher-Yates swap /
    # selection work, so they bill dtype-neutral: a complex/fp64 population
    # costs the SAME as fp32 (contrast a genuine sampler, which bills the width
    # of the values it synthesizes). They must also (a) not crash on a
    # FlopscopeArray operand and (b) accept complex.
    from flopscope._registry import REGISTRY

    load_weights()
    zc = fnp.asarray(np.arange(10, dtype=np.complex128))
    fr = fnp.asarray(np.arange(10, dtype=np.float32))
    fd = fnp.asarray(np.arange(10, dtype=np.float64))
    # in-place shuffle must not raise (regression: empty_like reentry) and
    # bills shape[0]=10 flat, with no 2x for the complex width.
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.random.shuffle(zc)
        assert b.flops_used == 10  # neutral: NOT 20
    # complex == fp64 == fp32 (all shape-based, dtype-blind).
    assert (
        _cost(lambda: fnp.random.permutation(zc))
        == _cost(lambda: fnp.random.permutation(fd))
        == _cost(lambda: fnp.random.permutation(fr))
    )
    # Generator path strips the FlopscopeArray operand and is likewise neutral.
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.random.default_rng(0).shuffle(
            fnp.asarray(np.arange(8, dtype=np.complex128))
        )
        assert b.flops_used == 8  # neutral: NOT 16
    # Registry hygiene: movement ops are classified 1.0 (numpy permutes complex
    # fine), NOT "illegal" -- so a future dtype declaration cannot wrongly raise.
    assert REGISTRY["random.shuffle"]["complex_factor"] == 1.0
    assert REGISTRY["random.Generator.shuffle"]["complex_factor"] == 1.0


def test_asarray_charges_value_changing_cast_like_astype():
    # asarray(x, dtype=) that changes values does the same work as astype and
    # is charged the same (numel at the heavier rate) -- not a free conversion.
    # No dtype / lossless widening stays free.
    load_weights()
    a32 = fnp.asarray(np.ones(100, dtype=np.float32))
    a64 = fnp.asarray(np.ones(100, dtype=np.float64))
    assert _cost(lambda: fnp.asarray(a32)) == 0  # no-op
    assert _cost(lambda: fnp.asarray(a32, dtype=np.float64)) == 0  # lossless widen
    assert _cost(lambda: fnp.asarray(a64, dtype=np.int32)) == 200  # lossy, f64 rate
    assert _cost(lambda: fnp.asarray(a32, dtype=np.int32)) == 100  # lossy, f32 rate


# ---------------------------------------------------------------------------
# Review follow-up: numpy's implicit reduction/accumulator dtype (width path)
# ---------------------------------------------------------------------------


def test_integer_accumulating_reductions_bill_int64_accumulator():
    # numpy widens a narrow-int sum/prod/cumsum to a 64-bit accumulator, so
    # billing must too -- else int32 reductions ride at the 32-bit rate while
    # numpy accumulates at 64-bit. Floats do NOT widen (float16 stays float16).
    load_weights()
    i32 = fnp.asarray(np.ones(1000, dtype=np.int32))
    u32 = fnp.asarray(np.ones(1000, dtype=np.uint32))
    f32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    f16 = fnp.asarray(np.ones(1000, dtype=np.float16))
    for red in (fnp.sum, fnp.prod, fnp.cumsum, fnp.cumprod):
        assert _cost(lambda r=red: r(i32)) == 2 * _cost(
            lambda r=red: r(f32)
        )  # int64 rate
        assert _cost(lambda r=red: r(u32)) == 2 * _cost(
            lambda r=red: r(f32)
        )  # uint64 rate
        assert _cost(lambda r=red: r(f16)) == _cost(lambda r=red: r(f32))  # no widening


def test_extremum_and_index_reductions_do_not_widen():
    # max/min do not promote (result is the input dtype); argmax/argmin return
    # an int64 INDEX but the work is input-dtype comparisons -- neither should
    # be billed at the int64 rate for an int32 input.
    load_weights()
    i32 = fnp.asarray(np.ones(1000, dtype=np.int32))
    f32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    for red in (fnp.max, fnp.min, fnp.argmax, fnp.argmin):
        assert _cost(lambda r=red: r(i32)) == _cost(
            lambda r=red: r(f32)
        )  # int32 rate 1.0


def test_mean_variance_of_integers_bill_float64():
    # mean/var/std of a bool/integer array are computed in float64 by numpy.
    load_weights()
    i32 = fnp.asarray(np.ones(1000, dtype=np.int32))
    f32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    for red in (fnp.mean, fnp.var, fnp.std):
        assert _cost(lambda r=red: r(i32)) == 2 * _cost(lambda r=red: r(f32))  # float64


def test_complex_variance_keeps_complex_factor():
    # A complex input keeps its own dtype through mean/var (not coerced to a
    # real), so complex variance still carries the var complex factor (2.5).
    load_weights()
    c64 = fnp.asarray(np.ones(1000, dtype=np.complex64))
    f32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    assert _cost(lambda: fnp.var(c64)) == int(2.5 * _cost(lambda: fnp.var(f32)))


def test_scalar_rng_draw_bills_output_dtype():
    # A size=None sampler draw returns a scalar; it must bill its output width,
    # not fall through to dtype-neutral. It bills identically to a 1-element
    # array draw (float64) and matches the array draw's float64 per-element
    # rate -- under the old behavior the scalar was billed neutral (half).
    load_weights()
    scalar = _cost(lambda: fnp.random.default_rng(0).standard_normal())
    size1 = _cost(lambda: fnp.random.default_rng(0).standard_normal(1))
    arr = _cost(lambda: fnp.random.default_rng(0).standard_normal(1000))
    assert scalar == size1 == arr // 1000


# ---------------------------------------------------------------------------
# Task 1: zero-work complex contractions charge cleanly
# ---------------------------------------------------------------------------


def _zero_work_charge(fn) -> int:
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def test_complex_einsum_transpose_charges_zero():
    z = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex128)
    import flopscope.numpy as fnp

    assert _zero_work_charge(lambda: fnp.einsum("ij->ji", z)) == 0
    assert _zero_work_charge(lambda: fnp.einsum("i->i", z[0])) == 0
    assert _zero_work_charge(lambda: fnp.einsum("ii->i", z)) == 0


def test_empty_complex_contractions_charge_zero():
    import flopscope.numpy as fnp

    z = np.ones((8, 8), dtype=np.complex128)
    z_empty = np.zeros((0, 8), dtype=np.complex128)
    assert _zero_work_charge(lambda: fnp.matmul(z_empty, z)) == 0
    assert (
        _zero_work_charge(
            lambda: fnp.dot(
                np.zeros((0,), np.complex128), np.zeros((0,), np.complex128)
            )
        )
        == 0
    )


def test_nonzero_complex_contractions_still_bill_exactly():
    # Regression guard: the fix must not disturb exact complex billing.
    # 8x8 complex128 matmul measured on this branch: 7936.
    import flopscope.numpy as fnp

    z = np.ones((8, 8), dtype=np.complex128)
    assert _zero_work_charge(lambda: fnp.matmul(z, z)) == 7936


# ---------------------------------------------------------------------------
# Task 3: _counted_reduction bills the requested accumulator (positional + keyword)
# ---------------------------------------------------------------------------


def _billed_with_production_rates(fn) -> tuple[int, str | None]:
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        dt = b.op_log[-1].resolved_dtype if b.op_log else None
        return b.flops_used, dt


def test_explicit_narrow_reduction_dtype_bills_narrow():
    import flopscope.numpy as fnp

    x = np.arange(1000, dtype=np.int32)
    # Implicit: numpy widens int32 sum to int64 -> rate 2.0 (existing behavior).
    assert _billed_with_production_rates(lambda: fnp.sum(x)) == (1998, "int64")
    # Explicit int32 accumulator: numpy runs 32-bit adds -> rate 1.0.
    assert _billed_with_production_rates(
        lambda: fnp.sum(x, dtype=np.int32)
    ) == (999, "int32")


def test_positional_reduction_dtype_is_billed():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    # numpy signature sum(a, axis, dtype, ...): dtype passed positionally.
    assert _billed_with_production_rates(
        lambda: fnp.sum(x, None, np.float64)
    ) == (1998, "float64")
    # keyword spelling bills identically
    assert _billed_with_production_rates(
        lambda: fnp.sum(x, dtype=np.float64)
    ) == (1998, "float64")


def test_narrowing_reduction_dtype_floors_at_input_rate():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float64)
    billed, dt = _billed_with_production_rates(lambda: fnp.sum(x, dtype=np.float32))
    assert billed == 1998  # f64 floor: no narrow discount
    assert dt == "float64"


def test_reduction_out_dtype_sets_accumulator():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    out = np.empty((), dtype=np.float64)
    # ufunc.reduce semantics: out= without dtype= IS the accumulator dtype.
    assert _billed_with_production_rates(
        lambda: fnp.sum(x, out=out)
    ) == (1998, "float64")


def test_ufunc_reduce_protocol_bills_requested_accumulator():
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.arange(1000, dtype=np.int32))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # auto-route notice
            np.add.reduce(arr, dtype=np.int32)
        assert b.flops_used - before == 999  # int32 accumulator, rate 1.0


# ---------------------------------------------------------------------------
# Task 4: mean/variance factories + generic ufunc-method paths resolve their
# accumulator dtype
# ---------------------------------------------------------------------------


def test_mean_variance_explicit_dtype_replaces_float64_default():
    import flopscope.numpy as fnp

    x = np.arange(1000, dtype=np.int32)
    # Implicit float64 compute (existing behavior — pins Task 4 didn't break it).
    assert _billed_with_production_rates(lambda: fnp.mean(x)) == (2000, "float64")
    # Explicit float32 compute: numpy honors it; bill rate 1.0.
    assert _billed_with_production_rates(
        lambda: fnp.mean(x, dtype=np.float32)
    ) == (1000, "float32")
    billed, dt = _billed_with_production_rates(lambda: fnp.var(x, dtype=np.float32))
    assert dt == "float32"
    billed64, _ = _billed_with_production_rates(lambda: fnp.var(x))
    assert billed * 2 == billed64  # rate 1.0 vs 2.0 on the same flop_cost


def test_mean_positional_dtype_is_billed():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    # np.mean(a, axis, dtype): positional dtype binds the wrapper's named param.
    assert _billed_with_production_rates(lambda: fnp.mean(x, None, np.float64))[1] == "float64"


def test_generic_ufunc_reduce_bills_requested_dtype():
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.ones(100, dtype=np.float32))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            np.subtract.reduce(arr, dtype=np.float64)
        # 99 subtract steps... reduction_cost(100 elems) at f64 rate 2.0.
        billed = b.flops_used - before
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b2:
        load_weights()
        arr2 = fnp.asarray(np.ones(100, dtype=np.float32))
        before2 = b2.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            np.subtract.reduce(arr2)
        baseline = b2.flops_used - before2
    assert billed == 2 * baseline  # f64-requested accumulation bills 2x f32
