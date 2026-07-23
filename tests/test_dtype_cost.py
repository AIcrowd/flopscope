"""Four-factor billing: charged = int(flop_cost * dtype_rate * complex_factor * weight)."""

import numpy as np
import pytest
from numpy.typing import DTypeLike

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
    # flop_cost=0 data-movement ops (transpose) bill 0 for any dtype, incl
    # complex. (reshape was the witness pre-Task-4; it now bills numel(input).)
    load_weights()
    zc = fnp.asarray(np.ones((3, 4), dtype=np.complex128))
    assert _cost(lambda: fnp.transpose(zc)) == 0


def test_astype_bills_and_still_resolves_heavier_dtype():
    # Option B: astype bills numel(input) at the HEAVIER of the source and
    # destination rate -- astype reads the source and writes the
    # destination, so the pricier side dominates. Neither source-only nor
    # result_type is the right resolution: source-only would under-resolve
    # when dst is pricier (complex64->float64); result_type over-resolves a
    # cross-kind narrowing (float32+int32 promotes to float64). All calls
    # below use the default copy=True, so all are real copies and bill
    # 1000 * rate(resolved) (* complex_factor 2.0 when resolved is complex).
    load_weights()
    a64 = fnp.asarray(np.ones(1000, dtype=np.float64))
    a32 = fnp.asarray(np.ones(1000, dtype=np.float32))
    c64 = fnp.asarray(np.ones(1000, dtype=np.complex64))
    c128 = fnp.asarray(np.ones(1000, dtype=np.complex128))
    # source pricier than dest:
    assert _billed_with_production_rates(lambda: fnp.astype(a64, np.int32)) == (
        2000,
        "float64",
    )
    assert _billed_with_production_rates(lambda: fnp.astype(a32, np.int32)) == (
        1000,
        "float32",
    )
    # dest pricier than source: complex64->float64 resolves to the real
    # float64 rate (complex64's own rate is lighter than float64's).
    assert _billed_with_production_rates(lambda: fnp.astype(c64, np.float64)) == (
        2000,
        "float64",
    )
    # complex128->float64 ties on rate (both 2.0); the tie keeps the source
    # (complex128, listed first), so the resolved dtype stays complex --
    # and the complex_factor (2.0) doubles the plain rate*numel cost.
    assert _billed_with_production_rates(lambda: fnp.astype(c128, np.float64)) == (
        4000,
        "complex128",
    )
    assert _billed_with_production_rates(lambda: fnp.astype(a32, np.int64)) == (
        2000,
        "int64",
    )  # real-only dst pricier


def test_complex_movement_and_creation_do_not_raise():
    # Free / data-movement ops on complex values must not fail closed; they
    # relocate whole complex values without raising UnsupportedDtypeError.
    load_weights()
    zc = fnp.asarray(np.ones(100, dtype=np.complex128))
    assert _cost(lambda: fnp.asarray(np.ones(50, dtype=np.complex128))) == 0
    # ravel is billed as of Task 4 (numel(input)); flip is the still-free
    # movement-op witness here.
    assert _cost(lambda: fnp.flip(zc)) == 0
    assert _cost(lambda: fnp.broadcast_to(zc, (2, 100))) == 0
    # astype complex->real must not raise despite discarding the imaginary
    # component -- both the function form and the .astype() method form
    # bill like copy (numel * rate(float64) * complex_factor(astype)=2.0 --
    # the RESOLVED dtype stays complex128, so the complex factor still
    # applies even though the result is real): 100 * 2.0 * 2.0 = 400.
    assert _cost(lambda: fnp.astype(zc, np.float64)) == 400
    assert _cost(lambda: zc.astype(np.float64)) == 400


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
    # Array bin edges are a second operand numpy promotes across. Task 8's
    # compute-dtype conformance sweep additionally fixed histogram's COUNTS
    # output: it is always the platform default int (numpy's bincount-style
    # accumulation), regardless of a/bins' own precision, so the bill is now
    # floored at that (>= float64-equivalent) rate unconditionally -- a
    # float32 `a` with float32 edges no longer bills below it, and float64
    # edges on the same float32 `a` no longer bill 2x more (both already hit
    # the same int64-counts floor).
    load_weights()
    d = fnp.asarray(np.ones(100, dtype=np.float32))
    b32 = np.linspace(0, 2, 5).astype(np.float32)
    # histogram's mirrored signature types bins as int|Sequence[int]|str; numpy
    # (and the runtime) accept float/array edges, so ignore the narrow hint.
    c32 = _cost(lambda: fnp.histogram(d, bins=fnp.asarray(b32)))  # type: ignore[arg-type]
    c64 = _cost(lambda: fnp.histogram(d, bins=fnp.asarray(b32.astype(np.float64))))  # type: ignore[arg-type]
    assert c32 > 0 and c64 == c32
    # int bins is a count, not data -> dtype irrelevant. Pin the exact value so
    # a future change to the int-bins path can't silently drift:
    # 100 elems * ceil(log2(10))=4 = 400 base cost, at the int64-floored
    # rate 2.0 (counts are always platform-default-int) and weight 4.0
    # (access tier, see cost-model repricing) = 3200.
    assert _cost(lambda: fnp.histogram(d, bins=10)) == 3200


def test_histogram2d_histogramdd_array_bins_dtype_is_billed():
    # histogram2d/histogramdd fold array bin-edge dtypes in too, but (like
    # plain histogram, see test_histogram_array_bins_dtype_is_billed) their
    # COUNTS output is always float64 regardless of x/y/sample's own
    # precision (Task 8), so float32 vs float64 edges on float32 data no
    # longer produce a 2x difference -- both hit the same float64-counts
    # floor.
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
    assert h2_32 > 0 and h2_64 == h2_32
    s = fnp.asarray(np.ones((100, 2), dtype=np.float32))
    hd_32 = _cost(lambda: fnp.histogramdd(s, bins=[fnp.asarray(e32), fnp.asarray(e32)]))
    hd_64 = _cost(lambda: fnp.histogramdd(s, bins=[fnp.asarray(e64), fnp.asarray(e64)]))
    assert hd_32 > 0 and hd_64 == hd_32


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


def test_requested_narrow_dtype_bills_the_loop_numpy_actually_runs():
    # Task 5 revision of "the other half of the resolved-dtype rule": an
    # explicit dtype= FORCES the ufunc loop (numpy casts operands on read
    # and computes at that width -- plain np.multiply(f64, f64,
    # dtype=np.float32) genuinely returns a float32 result), so billing the
    # requested float32 rate is the honest compute cost, not a discount --
    # the participant traded precision for a cheaper bill, which is exactly
    # what an explicit dtype= asks for.
    # (Contrast test_output_downcast_does_not_discount above: matmul takes
    # no dtype= kwarg at all, so it has no equivalent "requested narrower
    # loop" and that invariant is untouched by this revision.)
    load_weights()
    w = fnp.asarray(np.ones(100, dtype=np.float64))
    plain = _cost(lambda: fnp.multiply(w, w))
    narrow_requested = _cost(lambda: fnp.multiply(w, w, dtype=np.float32))
    assert plain == 200
    assert narrow_requested == 100  # float32 rate 1.0 vs float64 rate 2.0: half


def test_astype_to_complex_and_back_bills_like_copy():
    # Option B: both directions bill like copy now, regardless of
    # safe/unsafe-cast direction. Widening real->complex: heavier_billing_dtype
    # (float64, complex128) ties at rate 2.0, first (float64, real) wins, so
    # no complex_factor applies -- 100 * rate(float64)=2.0 = 200.
    load_weights()
    r = fnp.asarray(np.ones(100, dtype=np.float64))
    z = fnp.asarray(np.ones(100, dtype=np.complex128))
    assert _cost(lambda: r.astype(np.complex128)) == 200
    # narrowing complex->real: the dtype RESOLUTION is still the heavier of
    # source/dest (heavier_billing_dtype(complex128, float64), tied at rate
    # 2.0 -- the tie keeps the source, complex128) -> resolved dtype stays
    # complex, so the complex_factor (2.0) applies: 100 * 2.0 * 2.0 = 400.
    # Matches the sibling pin in test_complex_movement_and_creation_do_not_raise.
    assert _billed_with_production_rates(lambda: z.astype(np.float64)) == (
        400,
        "complex128",
    )


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
    # shuffle/permutation/choice relocate caller data (movement, registry
    # factor 2.0). Their shape[axis] cost counts the dtype-INDEPENDENT
    # Fisher-Yates swap / selection work and is billed via dtypes=(), so they
    # bill dtype-neutral (rate 1.0, factor 1.0 -- the registry factor is never
    # consulted): a complex/fp64 population costs the SAME as fp32 (contrast a
    # genuine sampler, which bills the width of the values it synthesizes).
    # They must also (a) not crash on a FlopscopeArray operand and (b) accept
    # complex.
    from flopscope._registry import REGISTRY

    load_weights()
    zc = fnp.asarray(np.arange(10, dtype=np.complex128))
    fr = fnp.asarray(np.arange(10, dtype=np.float32))
    fd = fnp.asarray(np.arange(10, dtype=np.float64))
    # in-place shuffle must not raise (regression: empty_like reentry) and
    # bills shape[0]=10 flat (x weight 4.0 access tier = 40), with no 2x for
    # the complex width.
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.random.shuffle(zc)
        assert b.flops_used == 40  # neutral: NOT 80
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
        assert b.flops_used == 32  # neutral: NOT 64
    # Registry hygiene: movement ops are classified 2.0 (numpy permutes complex
    # fine), NOT "illegal" -- so a future dtype declaration cannot wrongly raise.
    assert REGISTRY["random.shuffle"]["complex_factor"] == 2.0
    assert REGISTRY["random.Generator.shuffle"]["complex_factor"] == 2.0


def test_asarray_value_changing_cast_bills_like_astype():
    # Option B: asarray(x, dtype=) that actually converts the buffer does the
    # same real work as astype and bills the same way (numel at the heavier
    # of source/dest rate). The only free cases left are the genuine no-ops:
    # no dtype= at all, or a dtype= that already matches the input's dtype.
    load_weights()
    a32 = fnp.asarray(np.ones(100, dtype=np.float32))
    a64 = fnp.asarray(np.ones(100, dtype=np.float64))
    assert _cost(lambda: fnp.asarray(a32)) == 0  # no-op: no dtype= at all
    assert (
        _cost(lambda: fnp.asarray(a32, dtype=np.float32)) == 0
    )  # no-op: dtype= matches
    assert (
        _cost(lambda: fnp.asarray(a32, dtype=np.float64)) == 200
    )  # lossless widen: billed
    # lossy casts bill too; the resolved dtype still follows the same
    # heavier-of-source/dest rule as astype.
    assert _billed_with_production_rates(lambda: fnp.asarray(a64, dtype=np.int32)) == (
        200,
        "float64",
    )
    assert _billed_with_production_rates(lambda: fnp.asarray(a32, dtype=np.int32)) == (
        100,
        "float32",
    )


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
    assert _billed_with_production_rates(lambda: fnp.sum(x, dtype=np.int32)) == (
        999,
        "int32",
    )


def test_positional_reduction_dtype_is_billed():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    # numpy signature sum(a, axis, dtype, ...): dtype passed positionally.
    assert _billed_with_production_rates(lambda: fnp.sum(x, None, np.float64)) == (
        1998,
        "float64",
    )
    # keyword spelling bills identically
    assert _billed_with_production_rates(lambda: fnp.sum(x, dtype=np.float64)) == (
        1998,
        "float64",
    )


def test_narrowing_reduction_dtype_bills_the_requested_accumulator():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float64)
    billed, dt = _billed_with_production_rates(lambda: fnp.sum(x, dtype=np.float32))
    assert billed == 999  # explicit dtype IS the accumulator numpy runs
    assert dt == "float32"


def test_reduction_out_dtype_sets_accumulator():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    out = np.empty((), dtype=np.float64)
    # ufunc.reduce semantics: out= without dtype= widens the accumulator
    # when out is wider than the input (a narrower out would only cast the
    # final store -- see the next test).
    assert _billed_with_production_rates(lambda: fnp.sum(x, out=out)) == (
        1998,
        "float64",
    )


def test_reduction_out_narrower_does_not_narrow_the_accumulator():
    import flopscope.numpy as fnp

    # out= narrower than the input does not narrow numpy's loop -- the
    # intermediate keeps the input's width and only the final store casts.
    x = np.ones(1000, dtype=np.float64)
    out = np.empty((), dtype=np.float32)
    assert _billed_with_production_rates(lambda: fnp.sum(x, out=out)) == (
        1998,
        "float64",
    )


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
    assert _billed_with_production_rates(lambda: fnp.mean(x, dtype=np.float32)) == (
        1000,
        "float32",
    )
    billed, dt = _billed_with_production_rates(lambda: fnp.var(x, dtype=np.float32))
    assert dt == "float32"
    billed64, _ = _billed_with_production_rates(lambda: fnp.var(x))
    assert billed * 2 == billed64  # rate 1.0 vs 2.0 on the same flop_cost


def test_mean_positional_dtype_is_billed():
    import flopscope.numpy as fnp

    x = np.ones(1000, dtype=np.float32)
    # np.mean(a, axis, dtype): positional dtype binds the wrapper's named param.
    assert (
        _billed_with_production_rates(lambda: fnp.mean(x, None, np.float64))[1]
        == "float64"
    )


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


def _generic_ufunc_method_billed(method_call) -> int:
    """Delta-billed FLOPs for one generic ufunc-method call (production rates)."""
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.arange(1, 101, dtype=np.int32))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            method_call(arr)
        return b.flops_used - before


def test_float_only_ufunc_reduce_bills_its_float64_loop():
    # true_divide is not whest-routed, so its .reduce hits the generic path.
    # numpy has NO integer loop for it: an int32 reduce runs entirely in
    # float64 with no explicit dtype= — the bill must follow that loop, not
    # the int32 input (2x undercount otherwise). subtract on the same input
    # keeps its int32 loop (pinned as the no-widening baseline).
    assert _generic_ufunc_method_billed(lambda a: np.true_divide.reduce(a)) == 198
    assert _generic_ufunc_method_billed(lambda a: np.subtract.reduce(a)) == 99


def test_float_only_ufunc_accumulate_bills_its_float64_loop():
    assert _generic_ufunc_method_billed(lambda a: np.true_divide.accumulate(a)) == 198
    assert _generic_ufunc_method_billed(lambda a: np.subtract.accumulate(a)) == 99


# ---------------------------------------------------------------------------
# Task 6: elementwise wiring -- float-loop ufuncs bill their compute dtype
# ---------------------------------------------------------------------------


def test_int_transcendentals_bill_float64_compute():
    import flopscope.numpy as fnp

    x = np.arange(1, 1001, dtype=np.int32)
    # exp weight is 16.0: 1000 * 16 * rate. Pre-fix rate 1.0 -> 16000.
    assert _billed_with_production_rates(lambda: fnp.exp(x)) == (32000, "float64")
    assert _billed_with_production_rates(lambda: fnp.sqrt(x))[1] == "float64"


def test_int_divide_bills_float64_and_f32_stays_f32():
    import flopscope.numpy as fnp

    i = np.arange(1, 1001, dtype=np.int32)
    g = np.ones(1000, dtype=np.float32)
    assert _billed_with_production_rates(lambda: fnp.divide(i, i)) == (2000, "float64")
    # f32-stays-f32 invariant: no blanket f64 fold.
    assert _billed_with_production_rates(lambda: fnp.divide(g, g)) == (1000, "float32")


def test_float_power_always_bills_float64_minimum():
    import flopscope.numpy as fnp

    g = np.ones(1000, dtype=np.float32)
    billed, dt = _billed_with_production_rates(lambda: fnp.float_power(g, g))
    assert dt == "float64"
    billed64, _ = _billed_with_production_rates(
        lambda: fnp.float_power(g.astype(np.float64), g.astype(np.float64))
    )
    assert billed == billed64


def test_float_power_complex_keeps_registry_complex_factor():
    # The task brief expected complex float_power to fail closed; on this
    # numpy (2.2.6) np.float_power has native complex loops (its ``.types``
    # includes ``'DD->D'``/``'GG->G'``) and the registry classifies its
    # complex_factor as 5.5 (a real transcendental factor, not "illegal"),
    # so a plain call computes and returns -- adding a raise would be a new
    # error path for numpy-compatible code. This test pins the honest
    # billing instead, on both axes:
    #  - KIND: complex stays complex-kind, so the registry 5.5
    #    complex_factor applies (folding into a bare real float64 minimum
    #    would silently drop it);
    #  - WIDTH: float_power has no FF->F loop -- DD->D is its complex
    #    minimum, so complex64 inputs compute (and must bill) complex128.
    import flopscope.numpy as fnp

    z = np.ones(4, dtype=np.complex64)
    billed, dt = _billed_with_production_rates(lambda: fnp.float_power(z, z))
    assert dt == "complex128"  # no FF->F loop; DD->D minimum
    # 4 elems * rate(complex128)=2.0 * complex_factor(float_power)=5.5 * weight=16.0
    assert billed == 704
    billed128, _ = _billed_with_production_rates(
        lambda: fnp.float_power(z.astype(np.complex128), z.astype(np.complex128))
    )
    assert billed == billed128  # c64 pair bills exactly its true c128 compute


def test_small_int_unary_keeps_rate_one():
    import flopscope.numpy as fnp

    x8 = np.ones(1000, dtype=np.int8)
    # exp(int8) runs the float16 loop -> still rate 1.0; membership mapping
    # must not overbill sub-32-bit inputs.
    billed, dt = _billed_with_production_rates(lambda: fnp.exp(x8))
    assert dt == "float16"
    assert billed == 16000  # 1000 * weight 16 * rate 1.0


# ---------------------------------------------------------------------------
# Coordinator addendum: the same float-loop undercount class is live on
# ufunc.outer / ufunc.reduceat (Task 4's Minor(8), folded into Task 6)
# ---------------------------------------------------------------------------


def test_outer_float_only_binary_bills_float64_and_int_stays_int():
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        a = fnp.asarray(np.ones(10, dtype=np.int32))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # auto-route notice
            np.hypot.outer(a, a)
        billed_hypot = b.flops_used - before
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            np.add.outer(a, a)
        billed_add = b.flops_used - before
    assert billed_hypot == 200  # 10*10 dense numel * float64 rate 2.0
    assert billed_add == 100  # 10*10 dense numel * int32 rate 1.0 (unaffected)


def test_reduceat_float_only_binary_bills_float64_and_add_widens_too():
    # true_divide has no integer loop at all (int32 reduceat runs entirely
    # in float64); add keeps its native int32 PAIRWISE loop, but reduceat --
    # like reduce/sum -- runs add through numpy's integer-widening
    # accumulator, so an int32 input still bills int64 (see the dedicated
    # add/subtract accumulator pins below for that story in isolation).
    assert (
        _generic_ufunc_method_billed(lambda a: np.true_divide.reduceat(a, [0, 10]))
        == 200
    )
    assert _generic_ufunc_method_billed(lambda a: np.add.reduceat(a, [0, 10])) == 200


def _i64_method_billed(method_call) -> tuple[int, str | None]:
    """Delta-billed FLOPs + resolved dtype for one ufunc-method call on int64.

    int64 (rate 2.0) discriminates the input-rate floor: a comparison/logical
    ufunc whose loop OUTPUT is bool (rate 1.0) must still bill the int64 input
    rate, not the narrower bool rate.
    """
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.ones(10, dtype=np.int64))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # auto-route notice
            method_call(arr)
        return b.flops_used - before, b.op_log[-1].resolved_dtype


def test_outer_bool_loop_floors_at_input_rate():
    # less.outer's loop OUTPUT is bool, but the input is int64 -- billing the
    # bool rate would be NARROWER than the input (a 2x undercharge that
    # regresses pre-delta behavior). The floor pins it to int64.
    # add.outer (native int64 loop) is the matching baseline.
    assert _i64_method_billed(lambda a: np.less.outer(a, a)) == (200, "int64")
    assert _i64_method_billed(lambda a: np.add.outer(a, a)) == (200, "int64")


def test_reduceat_bool_loop_floors_at_input_rate():
    # logical_and.reduceat's loop OUTPUT is bool; the int64 input rate floors it.
    _, dt = _i64_method_billed(lambda a: np.logical_and.reduceat(a, [0, 5]))
    assert dt == "int64"


def test_outer_float_widening_survives_the_floor():
    # The floor is a MAX, so float-widening cases (float64 rate 2.0 >= int64
    # rate 2.0, and strictly > narrower int rates) are never lowered by it.
    assert _i64_method_billed(lambda a: np.hypot.outer(a, a)) == (200, "float64")


def test_reduceat_float_widening_survives_the_floor():
    assert _i64_method_billed(lambda a: np.true_divide.reduceat(a, [0, 5]))[1] == (
        "float64"
    )


# ---------------------------------------------------------------------------
# Final review: reduceat resolves its accumulator like reduce/sum; outer
# honors an explicit dtype=; a bool dtype= on a value-testing loop still
# bills the operands, not the bool output
# ---------------------------------------------------------------------------


def _reduceat_billed(method_call, *, n=1000, dtype: DTypeLike = np.int32) -> int:
    """Delta-billed FLOPs for one ``ufunc.reduceat`` call (production rates).

    Parametrized sibling of :func:`_generic_ufunc_method_billed` for the
    accumulator pins below, which need a longer array and non-default
    dtypes.
    """
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.ones(n, dtype=dtype))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # auto-route notice
            method_call(arr)
        return b.flops_used - before


def test_reduceat_add_multiply_use_sum_accumulator_dtype():
    # add/multiply.reduceat -- like sum/prod -- run through numpy's
    # integer-widening accumulator by default, regardless of the segment
    # indices: an int32 input bills int64. subtract has no such accumulator
    # and keeps its native int32 loop -- the contrast pin.
    assert _reduceat_billed(lambda a: np.add.reduceat(a, [0])) == 2000
    assert _reduceat_billed(lambda a: np.subtract.reduceat(a, [0])) == 1000


def test_reduceat_explicit_dtype_is_the_accumulator_not_a_discount():
    # An explicit dtype= on reduceat IS the accumulator numpy runs -- billed
    # exactly as requested, wider or narrower, mirroring reduce/sum.
    assert (
        _reduceat_billed(
            lambda a: np.add.reduceat(a, [0], dtype=np.float64), dtype=np.float32
        )
        == 2000
    )
    assert (
        _reduceat_billed(
            lambda a: np.subtract.reduceat(a, [0], dtype=np.float32), dtype=np.float64
        )
        == 1000
    )


def test_outer_explicit_dtype_resolves_like_the_pointwise_factories():
    # An explicit dtype= on outer forces the loop the same way it does for
    # the plain pointwise factories (Task 6): billed as requested, replacing
    # the operand-promoted default rather than discounting it.
    import warnings

    import flopscope.numpy as fnp

    def _outer_billed(dtype=None):
        with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
            load_weights()
            arr = fnp.asarray(np.ones(32, dtype=np.int32))
            before = b.flops_used
            kwargs = {} if dtype is None else {"dtype": dtype}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                np.multiply.outer(arr, arr, **kwargs)
            return b.flops_used - before, b.op_log[-1].resolved_dtype

    default_billed, default_dtype = _outer_billed()
    assert default_dtype == "int32"
    explicit_billed, explicit_dtype = _outer_billed(np.float64)
    assert explicit_dtype == "float64"
    assert explicit_billed == 2 * default_billed


def test_bool_dtype_kwarg_on_value_testing_ufuncs_bills_operand_width():
    # dtype=bool on less/logical_and/equal names the output of a
    # value-testing loop -- the loop still reads full-width operands, so it
    # must not discount to the bool rate. Same principle as the input-rate
    # floor above, applied to the plain pointwise factories instead of the
    # generic ufunc-method paths.
    import flopscope.numpy as fnp

    i64 = fnp.asarray(np.ones(10, dtype=np.int64))
    f64 = fnp.asarray(np.ones(10, dtype=np.float64))
    assert _billed_with_production_rates(lambda: fnp.less(i64, i64, dtype=bool)) == (
        20,
        "int64",
    )
    assert _billed_with_production_rates(
        lambda: fnp.logical_and(i64, i64, dtype=bool)
    ) == (20, "int64")
    assert _billed_with_production_rates(lambda: fnp.equal(f64, f64, dtype=bool)) == (
        20,
        "float64",
    )


def _ufunc_at_billed(at_call) -> tuple[int, str | None]:
    """Delta-billed FLOPs + resolved dtype for one ufunc.at call (production rates)."""
    import warnings

    import flopscope.numpy as fnp

    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        load_weights()
        arr = fnp.asarray(np.ones(1000, dtype=np.int32))
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # auto-route notice
            at_call(arr)
        return b.flops_used - before, b.op_log[-1].resolved_dtype


def test_ufunc_at_float_only_bills_loop_dtype():
    # ufunc.at applies float-only loops to integer arrays WITHOUT raising
    # (numpy casts the float result back in place with unsafe casting), so
    # the same compute-dtype rule as the other ufunc-method paths applies:
    # exp.at / true_divide.at on int32 run entirely in float64 and must
    # bill it. bitwise_or keeps its native int32 loop -- the contrast pins
    # that the fix stays scoped to float-only loops, not a blanket widening.
    # (slice indices are runtime-supported by ufunc.at; numpy's stubs type
    # the parameter narrower than the implementation accepts.)
    vals = np.ones(1000, dtype=np.int32)
    assert _ufunc_at_billed(
        lambda a: np.exp.at(a, slice(None))  # pyright: ignore[reportArgumentType]
    ) == (2000, "float64")
    assert _ufunc_at_billed(
        lambda a: np.true_divide.at(a, slice(None), vals)  # pyright: ignore[reportArgumentType]
    ) == (2000, "float64")
    assert _ufunc_at_billed(
        lambda a: np.bitwise_or.at(a, slice(None), vals)  # pyright: ignore[reportArgumentType]
    ) == (1000, "int32")


# ---------------------------------------------------------------------------
# Task 7: fft and linalg bill their compute dtype
# ---------------------------------------------------------------------------


def test_fft_bills_complex_working_precision():
    import flopscope.numpy as fnp

    sig_i = np.arange(1024, dtype=np.int32)
    sig_f32 = np.arange(1024, dtype=np.float32)
    sig_f64 = np.arange(1024, dtype=np.float64)
    # 5 * 1024 * 10 = 51200 raw; int -> complex128 path bills rate 2.0.
    assert _billed_with_production_rates(lambda: fnp.fft.fft(sig_i)) == (
        102400,
        "complex128",
    )
    # f32-stays-single invariant: complex64 path bills rate 1.0.
    assert _billed_with_production_rates(lambda: fnp.fft.fft(sig_f32)) == (
        51200,
        "complex64",
    )
    assert (
        _billed_with_production_rates(lambda: fnp.fft.fft(sig_f64))[1] == "complex128"
    )


def test_linalg_bills_lapack_compute_dtype():
    import flopscope.numpy as fnp

    a_i32 = (np.eye(8) * 3).astype(np.int32)
    b_i32 = np.ones(8, dtype=np.int32)
    a_f32 = a_i32.astype(np.float32)
    b_f32 = b_i32.astype(np.float32)
    # int inputs run LAPACK in float64: rate 2.0 (was 469 at int rate).
    assert _billed_with_production_rates(lambda: fnp.linalg.solve(a_i32, b_i32)) == (
        938,
        "float64",
    )
    # f32 keeps the single-precision driver: rate 1.0.
    assert _billed_with_production_rates(lambda: fnp.linalg.solve(a_f32, b_f32)) == (
        469,
        "float32",
    )
    assert _billed_with_production_rates(lambda: fnp.linalg.inv(a_i32))[1] == "float64"


def test_integer_matmul_family_not_promoted():
    import flopscope.numpy as fnp

    a = np.eye(4, dtype=np.int32) * 2
    # matmul/matrix_power run INTEGER arithmetic — no LAPACK, no promotion.
    # Guard against over-eager linalg mapping (sweep only catches undercharge).
    assert _billed_with_production_rates(lambda: fnp.matmul(a, a))[1] == "int32"
    assert _billed_with_production_rates(lambda: fnp.linalg.matrix_power(a, 2))[1] in (
        "int32",
        "int64",
    )  # numpy computes integer matmuls here


def test_linalg_trace_bills_integer_accumulator():
    import flopscope.numpy as fnp

    a_i32 = (np.eye(8) * 3).astype(np.int32)
    a_f32 = np.eye(8, dtype=np.float32)
    # numpy widens the trace accumulator like sum: trace(int32) runs int64.
    assert _billed_with_production_rates(lambda: fnp.linalg.trace(a_i32)) == (
        16,
        "int64",
    )
    # float input keeps its own dtype (no widening): PIN.
    assert _billed_with_production_rates(lambda: fnp.linalg.trace(a_f32)) == (
        8,
        "float32",
    )


def test_top_level_trace_bills_integer_accumulator():
    import flopscope.numpy as fnp

    a_i32 = (np.eye(8) * 3).astype(np.int32)
    a_f32 = np.eye(8, dtype=np.float32)
    # fnp.trace is a SEPARATE wrapper from fnp.linalg.trace; same int widening,
    # else a participant bypasses the fix by spelling `trace` without `linalg.`.
    assert _billed_with_production_rates(lambda: fnp.trace(a_i32)) == (16, "int64")
    # explicit narrow accumulator is honest at 32-bit (mirrors sum).
    assert _billed_with_production_rates(lambda: fnp.trace(a_i32, dtype=np.int32)) == (
        8,
        "int32",
    )
    assert _billed_with_production_rates(lambda: fnp.trace(a_f32)) == (8, "float32")


def test_matrix_power_inversion_bills_float64():
    import flopscope.numpy as fnp

    a_i32 = (np.eye(4) * 2).astype(np.int32)
    a_f32 = (np.eye(4) * 2).astype(np.float32)
    # n<0 inverts via LAPACK first (int -> float64 driver): rate 2.0.
    assert _billed_with_production_rates(
        lambda: fnp.linalg.matrix_power(a_i32, -1)
    ) == (224, "float64")
    # n>=0 runs integer matmul chain — no promotion: PIN.
    assert _billed_with_production_rates(lambda: fnp.linalg.matrix_power(a_i32, 2)) == (
        112,
        "int32",
    )
    # f32 keeps the single-precision driver even when inverting: PIN.
    assert _billed_with_production_rates(
        lambda: fnp.linalg.matrix_power(a_f32, -1)
    ) == (112, "float32")


def test_complex_movement_prices_both_components():
    import flopscope.numpy as fnp

    z64 = np.ones(1000, dtype=np.complex64)
    z128 = np.ones(1000, dtype=np.complex128)
    f32 = np.ones(1000, dtype=np.float32)
    # conj / array on complex: one unit per real component.
    assert _billed_with_production_rates(lambda: fnp.conj(z64)) == (2000, "complex64")
    assert _billed_with_production_rates(lambda: fnp.conj(z128)) == (4000, "complex128")
    assert _billed_with_production_rates(lambda: fnp.conjugate(z64)) == (
        2000,
        "complex64",
    )
    assert _billed_with_production_rates(lambda: fnp.array(z64)) == (2000, "complex64")
    # the float32 baseline these are measured against
    assert _billed_with_production_rates(lambda: fnp.array(f32)) == (1000, "float32")


def test_complex_angle_prices_both_components():
    import flopscope.numpy as fnp

    z64 = np.ones(1000, dtype=np.complex64)
    f32 = np.ones(1000, dtype=np.float32)
    billed_angle = _billed_with_production_rates(lambda: fnp.angle(z64))
    billed_atan2 = _billed_with_production_rates(lambda: fnp.arctan2(f32, f32))
    assert billed_atan2 == (16000, "float32")
    assert billed_angle == (32000, "complex64")  # component convention: 2x


def test_fft_complex_billing_unchanged_by_the_floor():
    import flopscope.numpy as fnp

    sig64 = np.ones(1024, dtype=np.complex64)
    assert _billed_with_production_rates(lambda: fnp.fft.fft(sig64)) == (
        51200,
        "complex64",
    )


def test_casts_bill_like_copy_in_both_directions():
    import flopscope.numpy as fnp

    f64 = np.ones(1000, dtype=np.float64)
    c128 = np.ones(1000, dtype=np.complex128)
    # lossy and lossless casts both bill like copy now: representation
    # changes are only free for the true no-op (see the dedicated no-op
    # tests). Each `asarray(x)` call here has no dtype= of its own, so it
    # contributes 0; the chained `.astype(...)` is what bills.
    assert (
        _billed_with_production_rates(lambda: fnp.asarray(f64).astype(np.float32))[0]
        == 2000
    )
    assert (
        _billed_with_production_rates(lambda: fnp.asarray(f64).astype(np.int32))[0]
        == 2000
    )
    assert (
        _billed_with_production_rates(lambda: fnp.asarray(c128).astype(np.complex64))[0]
        == 4000
    )
    # here asarray itself performs the float64->float32 narrowing cast, no
    # chained astype:
    assert (
        _billed_with_production_rates(lambda: fnp.asarray(f64, dtype=np.float32))[0]
        == 2000
    )


# ---------------------------------------------------------------------------
# Task 3: copyto — priced per element written
# ---------------------------------------------------------------------------


def test_copyto_prices_elements_written():
    import flopscope.numpy as fnp

    src32 = np.ones(1000, dtype=np.float32)
    z64 = np.ones(1000, dtype=np.complex64)
    mask = np.zeros(1000, dtype=bool)
    mask[:250] = True

    def run(dst_dtype, src, **kw):
        dst = np.empty(1000, dtype=dst_dtype)
        return _billed_with_production_rates(lambda: fnp.copyto(dst, src, **kw))

    # same-dtype full copy: one unit per element written
    assert run(np.float32, src32) == (1000, "float32")
    # masked copy: per element selected
    assert run(np.float32, src32, where=mask) == (250, "float32")
    # complex copy: two real components per value
    assert run(np.complex64, z64) == (2000, "complex64")
    # 64-bit copy: rate axis unchanged
    assert run(np.float64, np.ones(1000)) == (2000, "float64")


# ---------------------------------------------------------------------------
# Task 5: pointwise dtype= forces the ufunc loop (replaces the operand
# promotion for billing, both narrowing and widening); out= alone still
# leaves the loop at the operand width (compute wide, store narrow).
# ---------------------------------------------------------------------------


def test_pointwise_dtype_forces_the_loop_billing():
    import flopscope.numpy as fnp

    f64 = np.ones(1000, dtype=np.float64)
    f32 = np.ones(1000, dtype=np.float32)
    i32 = np.arange(1, 1001, dtype=np.int32)
    # dtype= selects the narrow loop: operands cast on read, f32 arithmetic.
    assert _billed_with_production_rates(
        lambda: fnp.add(f64, f64, dtype=np.float32)
    ) == (1000, "float32")
    # widening dtype= unchanged
    assert _billed_with_production_rates(
        lambda: fnp.add(f32, f32, dtype=np.float64)
    ) == (2000, "float64")
    # out= alone keeps the wide loop (compute wide, store narrow)
    out32 = np.empty(1000, dtype=np.float32)
    assert _billed_with_production_rates(
        lambda: fnp.add(f64, f64, out=out32)  # pyright: ignore[reportArgumentType]
    ) == (2000, "float64")
    # transcendental: requested f32 loop bills f32 (weight 16)
    assert _billed_with_production_rates(lambda: fnp.exp(i32, dtype=np.float32)) == (
        16000,
        "float32",
    )


def test_float_power_forced_narrow_dtype_matches_numpys_own_rejection():
    # float_power has no float32 loop at all -- np.float_power.types is
    # ['dd->d', 'gg->g', 'DD->D', 'GG->G'], no 'ff->f' / 'FF->F' entry -- so
    # requesting dtype=float32 has no loop to select regardless of operand
    # width: plain np.float_power(f32, f32, dtype=np.float32) itself raises
    # TypeError ("No loop matching the specified signature and casting was
    # found") on this numpy (2.2.6).
    # The dtype=-replaces-promotion change only touches the BILLING tuple;
    # kwargs still reach the real numpy call unchanged, so flopscope raises
    # the identical error rather than silently succeeding at a loop numpy
    # itself refuses to run. (Whether a raised call's already-charged FLOPs
    # get rolled back is a separate, pre-existing deduct() concern this task
    # does not touch -- deduct() charges before the wrapped call runs, so
    # they do not; out of scope here.)
    # The reachable half of "the f64 minimum still applies" -- an
    # unrequested float32 operand pair still floors at float64 -- is already
    # locked by test_float_power_always_bills_float64_minimum above (the
    # family-mapping block that enforces it is untouched by this task, and
    # still runs downstream of the dtype=-replacement on the implicit-dtype
    # path); this pin covers the explicit-dtype=-requested half, which numpy
    # itself never allows to succeed at this width.
    import flopscope.numpy as fnp

    f32 = np.ones(1000, dtype=np.float32)
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        with pytest.raises(TypeError):
            fnp.float_power(f32, f32, dtype=np.float32)


def test_modf_dtype_forces_the_loop_billing():
    # _counted_unary_multi (modf/frexp) reads kwargs.get("dtype") with the
    # same append-vs-replace shape as _counted_unary; the fix must apply
    # there too. Ratio-based: robust to the exact pointwise flop_cost/weight.
    import flopscope.numpy as fnp

    f64 = np.ones(1000, dtype=np.float64)
    wide = _billed_with_production_rates(lambda: fnp.modf(f64))
    narrow = _billed_with_production_rates(lambda: fnp.modf(f64, dtype=np.float32))
    assert wide[1] == "float64"
    assert narrow[1] == "float32"
    assert narrow[0] * 2 == wide[0]  # rate 1.0 vs 2.0 on the same flop_cost


def test_divmod_dtype_forces_the_loop_billing():
    # _counted_binary_multi (divmod) reads kwargs.get("dtype") with the same
    # append-vs-replace shape as _counted_binary; the fix must apply there
    # too.
    import flopscope.numpy as fnp

    f64a = np.ones(1000, dtype=np.float64)
    f64b = np.ones(1000, dtype=np.float64) * 3
    wide = _billed_with_production_rates(lambda: fnp.divmod(f64a, f64b))
    narrow = _billed_with_production_rates(
        lambda: fnp.divmod(f64a, f64b, dtype=np.float32)
    )
    assert wide[1] == "float64"
    assert narrow[1] == "float32"
    assert narrow[0] * 2 == wide[0]  # rate 1.0 vs 2.0 on the same flop_cost
