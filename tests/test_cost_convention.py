"""Full-surface cost-convention lock (defensibility audit, 2026-06).

Conformance: charged == documented formula for curated custom-cost ops.
Completeness: every billed registry op is classified. See docs/reference/cost-model.md.

conftest resets weights to 1.0 per test, so charged == flop_cost for all assertions here.

IMPORTANT: All test arrays must be pre-built at module level (outside any _cost() call)
because fnp.asarray() is a billed op (numel FLOPs). Lambdas in OP_EXPECTATIONS must
not call fnp.asarray(), fnp.zeros(), or similar creation ops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
import flopscope.stats as fst
from flopscope._flops import sort_cost
from flopscope._registry import REGISTRY

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cost(fn) -> int:
    """Run fn inside a fresh BudgetContext (weights already reset by conftest)."""
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


# All pre-built arrays are created at module level, outside any BudgetContext.
# This avoids counting fnp.asarray() cost as part of the op under test.
_rng = np.random.default_rng(0)

_v10 = fnp.asarray(_rng.standard_normal(10))
_v10b = fnp.asarray(_rng.standard_normal(10))  # distinct from _v10
_v50 = fnp.asarray(_rng.standard_normal(50))
_v100 = fnp.asarray(_rng.standard_normal(100))
_sq10 = fnp.asarray(_rng.standard_normal((10, 10)))
_sq10_psd = fnp.asarray(_sq10.T @ _sq10 + np.eye(10))
_a3 = fnp.asarray(np.array([1.0, 2.0, 3.0]))
_b3 = fnp.asarray(np.array([4.0, 5.0, 6.0]))

# FFT inputs
_x64c = fnp.asarray(_rng.standard_normal(64).astype(complex))
_x64r = fnp.asarray(_rng.standard_normal(64))
_x88 = fnp.asarray(_rng.standard_normal((8, 8)))
_x444 = fnp.asarray(_rng.standard_normal((4, 4, 4)))
_x88c = fnp.asarray(_rng.standard_normal((8, 8)).astype(complex))
_x444c = fnp.asarray(_rng.standard_normal((4, 4, 4)).astype(complex))

# Sort / set ops
_int3x100 = fnp.asarray(_rng.integers(0, 10, (3, 100)))
_complex50 = fnp.asarray(_rng.standard_normal(50) + 1j * _rng.standard_normal(50))
_range100a = fnp.asarray(np.arange(100))
_range100b = fnp.asarray(np.arange(50, 150))
_range50 = fnp.asarray(np.arange(50))
_range25_75 = fnp.asarray(np.arange(25, 75))
_sorted_v100 = fnp.asarray(np.sort(np.asarray(_v100)))

# Poly inputs
_p4 = fnp.asarray(_rng.standard_normal(4))
_p5 = fnp.asarray(_rng.standard_normal(5))
_p6 = fnp.asarray(_rng.standard_normal(6))
_p10 = fnp.asarray(_rng.standard_normal(10))
_p11 = fnp.asarray(_rng.standard_normal(11))
# polyfit requires plain numpy arrays (flopscope guards against passing FlopscopeArray
# as y through the internal numpy call)
_v100_np = np.asarray(_v100)

# Histogram / digitize
_int100 = fnp.asarray(_rng.integers(0, 100, 100))
_linspace11 = fnp.asarray(np.linspace(-3, 3, 11))
_v100b = fnp.asarray(_rng.standard_normal(100))
_xy100 = fnp.asarray(_rng.standard_normal((100, 2)))

# Stats
_u100 = fnp.asarray(_rng.uniform(0.01, 0.99, 100))

# where inputs (condition must be pre-computed; v100>0 inside a lambda charges numel)
_v100_pos = fnp.asarray(np.asarray(_v100) > 0)  # bool mask, built outside BudgetContext

# Copy / gather / view ops
_sq10_2d = fnp.asarray(_rng.standard_normal((10, 10)))  # 2-D: diag extract → view → 0
_v5_1d = fnp.asarray(_rng.standard_normal(5))  # 1-D: diag construct → v.shape[0]=5
_sq3a = fnp.asarray(_rng.standard_normal((3, 3)))
_sq3b = fnp.asarray(_rng.standard_normal((3, 3)))
_idx10x3 = fnp.asarray(np.random.default_rng(7).integers(0, 10, (10, 3)))
_cond5 = fnp.asarray(np.array([True, False, True, True, False]))
_v5_float = fnp.asarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
_v8_bits = fnp.asarray(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8))

# ---------------------------------------------------------------------------
# COVERED_ELSEWHERE: ops with exact cost assertions in other test files.
# These are excluded from OP_EXPECTATIONS and need not appear in DEFERRED.
# ---------------------------------------------------------------------------

# test_cost_constant_unification.py
_CCU = {
    "linalg.svd",
    "linalg.svdvals",
    "linalg.norm",
    "linalg.matrix_rank",
    "linalg.cond",
    "linalg.pinv",
    "linalg.lstsq",
    "linalg.solve",
    "linalg.inv",
    "linalg.tensorsolve",
    "linalg.tensorinv",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.cholesky",
    "linalg.qr",
    "linalg.det",
    "linalg.slogdet",
    "linalg.matrix_norm",
    "linalg.vector_norm",
}

# test_fma2_cost_fixes.py
_FMA = {
    "tensordot",
    "linalg.multi_dot",
    "polymul",
    "convolve",
    "average",
    "var",
    "std",
    "nanvar",
    "trapezoid",
    "linspace",
    "geomspace",
    "logspace",
    "polydiv",
    "interp",
    "cross",
    "vander",
    "poly",
}

# test_ufunc_alias_parity.py — canonicals and aliases share the same ufunc object
_UFUNC = {
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "atan2",
    "pow",
    "divmod",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "arctan2",
    "power",
    "floor_divide",
}

# test_symmetric_cost.py — exact billing probes for (G+1)*n, (G+2)*n, k*(7n-1)
_SYMMETRIC = {
    "symmetrize",
    "as_symmetric",
    "is_symmetric",
    "random.symmetric",
}

# test_triage_price_pins.py — file I/O needs a tmp_path fixture, which this
# module's module-level-only array convention doesn't support; exact probes
# (4*size) live there instead (cost-model triage Task 10).
_TRIAGE_IO = {
    "save",
    "savez",
    "savez_compressed",
}

COVERED_ELSEWHERE: set[str] = _CCU | _FMA | _UFUNC | _SYMMETRIC | _TRIAGE_IO

# ---------------------------------------------------------------------------
# Registry categories where ALL members follow a simple family rule.
# counted_unary  → numel(output)  (weight varies per op, not tested here)
# counted_binary → numel(output)
# counted_reduction → numel(input) - numel(output) (or numel for cum-ops)
# test_completeness classifies every op in these categories by rule without
# requiring individual entries in OP_EXPECTATIONS or DEFERRED.
# ---------------------------------------------------------------------------

_UNARY_FAMILY = frozenset(
    op for op, e in REGISTRY.items() if e["category"] == "counted_unary"
)
_BINARY_FAMILY = frozenset(
    op for op, e in REGISTRY.items() if e["category"] == "counted_binary"
)
_REDUCTION_FAMILY = frozenset(
    op for op, e in REGISTRY.items() if e["category"] == "counted_reduction"
)

# ---------------------------------------------------------------------------
# OP_EXPECTATIONS: (callable, expected_int)
# Every expected value was verified by running the probe against current source.
# Key rule: lambdas MUST NOT call fnp.asarray() or any other billed creation op —
# all arrays must be pre-built at module level.
# ---------------------------------------------------------------------------

OP_EXPECTATIONS: dict[str, tuple] = {
    # ---- FFT family --------------------------------------------------------
    # Formula: 5 * N * ceil(log2(N)) for complex transforms
    # rfft / rfftn / rfft2: use N//2 (real half-spectrum)
    # irfft2 / irfftn: use n_out//2 * sum_axis(ceil(log2(si)))
    # ihfft: uses rfft_cost(n) = 5*(n//2)*ceil(log2(n)) — numpy ihfft = conj(rfft(a,n))
    "fft.fft": (
        lambda: fnp.fft.fft(_x64c),
        5 * 64 * int(math.ceil(math.log2(64))),  # 1920
    ),
    "fft.fftfreq": (
        lambda: fnp.fft.fftfreq(64),
        64,  # index grid scaled by 1/(n*d)
    ),
    "fft.rfftfreq": (
        lambda: fnp.fft.rfftfreq(64),
        64 // 2 + 1,
    ),
    "fft.ifft": (
        lambda: fnp.fft.ifft(_x64c),
        5 * 64 * int(math.ceil(math.log2(64))),  # 1920
    ),
    "fft.rfft": (
        lambda: fnp.fft.rfft(_x64r),
        5 * (64 // 2) * int(math.ceil(math.log2(64))),  # 960
    ),
    "fft.fft2": (
        lambda: fnp.fft.fft2(_x88),
        5 * 64 * (3 + 3),  # 1920
    ),
    "fft.fftn": (
        lambda: fnp.fft.fftn(_x444),
        5 * 64 * 3 * 2,  # 1920
    ),
    "fft.ifft2": (
        lambda: fnp.fft.ifft2(_x88),
        5 * 64 * (3 + 3),  # 1920
    ),
    "fft.ifftn": (
        lambda: fnp.fft.ifftn(_x444),
        5 * 64 * 6,  # 1920
    ),
    "fft.rfft2": (
        lambda: fnp.fft.rfft2(_x88),
        5 * (64 // 2) * (3 + 3),  # 960
    ),
    "fft.rfftn": (
        lambda: fnp.fft.rfftn(_x444),
        5 * (64 // 2) * 6,  # 960
    ),
    # irfft: complex input len 64 → output len 126; 5*(126//2)*ceil(log2(126))
    "fft.irfft": (
        lambda: fnp.fft.irfft(_x64c),
        5 * (126 // 2) * int(math.ceil(math.log2(126))),  # 2205
    ),
    # irfft2: (8,8) complex → output (8,14); n_out//2 = 56; log2 sums = 3+4=7
    "fft.irfft2": (
        lambda: fnp.fft.irfft2(_x88c),
        5 * (8 * 14 // 2) * (3 + 4),  # 1960
    ),
    # irfftn: (4,4,4) complex → output (4,4,6); n_out//2=48; log2 sums=2+2+3=7
    "fft.irfftn": (
        lambda: fnp.fft.irfftn(_x444c),
        5 * (4 * 4 * 6 // 2) * (2 + 2 + int(math.ceil(math.log2(6)))),  # 1680
    ),
    # hfft: numpy hfft(a,n) = irfft(conj(a),n); n_out=126 for input len 64; rfft_cost=5*(n//2)*ceil(log2(n))
    "fft.hfft": (
        lambda: fnp.fft.hfft(_x64r),
        5 * (126 // 2) * int(math.ceil(math.log2(126))),  # 2205
    ),
    # ihfft: numpy ihfft = conj(rfft(a,n)); uses rfft_cost(n) = 5*(n//2)*ceil(log2(n))
    "fft.ihfft": (
        lambda: fnp.fft.ihfft(_x64r),
        5 * (64 // 2) * int(math.ceil(math.log2(64))),  # 960
    ),
    # ---- Contraction (einsum family) --------------------------------------
    "matmul": (
        lambda: fnp.matmul(_sq10, _sq10),
        2 * 10**3 - 10**2,  # 1900
    ),
    "linalg.matmul": (
        lambda: fnp.linalg.matmul(_sq10, _sq10),
        2 * 10**3 - 10**2,  # 1900
    ),
    "dot": (
        lambda: fnp.dot(_sq10, _v10),
        10 * (2 * 10 - 1),  # 190 (matvec)
    ),
    "einsum": (
        lambda: fnp.einsum("ij,jk->ik", _sq10, _sq10),
        2 * 10**3 - 10**2,  # 1900
    ),
    "vdot": (
        lambda: fnp.vdot(_v100, _v100),
        2 * 100 - 1,  # 199
    ),
    "kron": (
        lambda: fnp.kron(_v10, _v10),
        10 * 10,  # 100
    ),
    # outer / linalg.outer: use two DISTINCT objects (same-object → symmetric orbit = 55)
    "outer": (
        lambda: fnp.outer(_v10, _v10b),
        10 * 10,  # 100
    ),
    "linalg.outer": (
        lambda: fnp.linalg.outer(_v10, _v10b),
        10 * 10,  # 100
    ),
    # inner 1D: always 2*n-1 regardless of same/different object
    "inner": (
        lambda: fnp.inner(_v10, _v10b),
        2 * 10 - 1,  # 19
    ),
    "linalg.matrix_power": (
        # k=3 (binary 11b) → 2 matmuls via binary exponentiation
        lambda: fnp.linalg.matrix_power(_sq10, 3),
        2 * (2 * 10**3 - 10**2),  # 3800
    ),
    # ---- Sort / select -----------------------------------------------------
    # Formula: n * ceil(log2(n)) per slice
    "sort": (
        lambda: fnp.sort(_v100),
        100 * int(math.ceil(math.log2(100))),  # 700
    ),
    "argsort": (
        lambda: fnp.argsort(_v100),
        100 * int(math.ceil(math.log2(100))),  # 700
    ),
    "searchsorted": (
        # _sorted_v100 is pre-sorted at module level
        lambda: fnp.searchsorted(_sorted_v100, _v50),
        50 * int(math.ceil(math.log2(100))),  # 350
    ),
    "unique": (
        lambda: fnp.unique(_v100),
        100 * int(math.ceil(math.log2(100))),  # 700
    ),
    "lexsort": (
        lambda: fnp.lexsort(_int3x100),  # pyright: ignore[reportArgumentType]  # 2-D keys array is valid
        3 * 100 * int(math.ceil(math.log2(100))),  # 2100
    ),
    "sort_complex": (
        lambda: fnp.sort_complex(_complex50),
        # complex_factor 2.0 (lexicographic compare touches both components);
        # unit dtype rates, so only the complex factor applies: 300 * 2 = 600.
        2 * 50 * int(math.ceil(math.log2(50))),  # 600
    ),
    "partition": (
        lambda: fnp.partition(_v100, 10),
        100,  # n per slice
    ),
    "argpartition": (
        lambda: fnp.argpartition(_v100, 10),
        100,
    ),
    # ---- Set ops -----------------------------------------------------------
    # Formula: n_total * ceil(log2(n_total)) where n_total = len(a) + len(b)
    "union1d": (
        lambda: fnp.union1d(_range100a, _range100b),
        200 * int(math.ceil(math.log2(200))),  # 1600
    ),
    # intersect1d default (assume_unique=False): numpy unique()-sorts both
    # inputs first, so cost = sort_cost(n) + sort_cost(m) + sort_cost(n+m).
    # n=m=100: 700 + 700 + 1600 = 3000
    "intersect1d": (
        lambda: fnp.intersect1d(_range100a, _range100b),
        sort_cost(100) + sort_cost(100) + sort_cost(200),
    ),
    "setdiff1d": (
        lambda: fnp.setdiff1d(_range100a, _range100b),
        200 * int(math.ceil(math.log2(200))),
    ),
    "setxor1d": (
        lambda: fnp.setxor1d(_range100a, _range100b),
        200 * int(math.ceil(math.log2(200))),
    ),
    "isin": (
        lambda: fnp.isin(_range50, _range25_75),
        100 * int(math.ceil(math.log2(100))),  # 700
    ),
    # ---- Generator family -------------------------------------------------
    "arange": (
        lambda: fnp.arange(100),
        2 * 100,  # 200
    ),
    # ---- Histogram / digitize ---------------------------------------------
    "histogram": (
        lambda: fnp.histogram(_v100, bins=10),
        100 * int(math.ceil(math.log2(10))),  # 400
    ),
    "histogram2d": (
        lambda: fnp.histogram2d(_v100, _v100b, bins=[10, 10]),
        100 * (int(math.ceil(math.log2(10))) + int(math.ceil(math.log2(10)))),  # 800
    ),
    "histogramdd": (
        # _xy100 is (100,2) pre-built FlopscopeArray
        lambda: fnp.histogramdd(_xy100, bins=[10, 10]),
        100 * 2 * int(math.ceil(math.log2(10))),  # 800
    ),
    "digitize": (
        lambda: fnp.digitize(_v100, _linspace11),
        100 * int(math.ceil(math.log2(11))),  # 400
    ),
    "bincount": (
        lambda: fnp.bincount(_int100),
        100,  # numel(x)
    ),
    # ---- Polynomial -------------------------------------------------------
    "polyval": (
        # Horner's method: deg 5 poly over 100 pts = 2*100*(6-1) = 1000
        lambda: fnp.polyval(_p6, _v100),
        2 * 100 * 5,  # 1000
    ),
    "polyadd": (
        lambda: fnp.polyadd(_p4, _p5),
        max(4, 5),  # 5
    ),
    "polysub": (
        lambda: fnp.polysub(_p4, _p5),
        max(4, 5),  # 5
    ),
    "polyder": (
        # polyder_cost(n=11, m=1): t=min(1,10)=1; cost=1*11 - 1*2//2 = 10
        lambda: fnp.polyder(_p11),
        10,
    ),
    "polyint": (
        lambda: fnp.polyint(_p11),
        11,
    ),
    "polyfit": (
        # plain numpy arrays required (FlopscopeArray causes internal tripwire)
        lambda: fnp.polyfit(_v100_np, _v100_np, 5),
        2 * 100 * (5 + 1) ** 2,  # 7200
    ),
    "roots": (
        # 10-coeff poly → 9×9 companion matrix; eigvals cost = 10*9^3
        lambda: fnp.roots(_p10),
        10 * 9**3,  # 7290
    ),
    # ---- Window (flop_cost at weight=1.0) ---------------------------------
    "bartlett": (
        lambda: fnp.bartlett(50),
        4 * 50,
    ),  # compare+div+add+select per sample (FMA=2)
    "blackman": (
        lambda: fnp.blackman(50),
        40 * 50,
    ),  # 2 cosine evals @16 + 8 arith per sample
    # cos@16 + mul + sub per sample (kaiser-family derived-constant convention)
    "hamming": (lambda: fnp.hamming(50), 18 * 50),
    "hanning": (lambda: fnp.hanning(50), 18 * 50),
    "kaiser": (lambda: fnp.kaiser(50, 14), 23 * 50),  # Bessel I0 @16 + 7 arith (FMA=2)
    # ---- Stats (fixed per-elem constants at weight=1.0) -------------------
    "stats.norm.pdf": (lambda: fst.norm.pdf(_v100), 27 * 100),
    "stats.norm.cdf": (lambda: fst.norm.cdf(_v100), 48 * 100),
    "stats.norm.ppf": (lambda: fst.norm.ppf(_u100), 83 * 100),
    "stats.lognorm.ppf": (lambda: fst.lognorm.ppf(_u100, 0.5), 106 * 100),
    "stats.truncnorm.ppf": (lambda: fst.truncnorm.ppf(_u100, -2, 2), 81 * 100),
    # audit-2 gap fixes (fix/cost-model-gaps):
    "stats.laplace.cdf": (lambda: fst.laplace.cdf(_v100), 40 * 100),
    "stats.laplace.ppf": (lambda: fst.laplace.ppf(_u100), 51 * 100),
    "stats.lognorm.pdf": (lambda: fst.lognorm.pdf(_u100, 0.5), 62 * 100),
    "stats.lognorm.cdf": (lambda: fst.lognorm.cdf(_v100, 0.5), 70 * 100),
    "stats.uniform.cdf": (lambda: fst.uniform.cdf(_u100), 4 * 100),
    "stats.cauchy.pdf": (lambda: fst.cauchy.pdf(_v100), 6 * 100),
    # ---- Selected reductions (not all reduction ops; family rule covers the rest) --
    # trapz: same formula as trapezoid (4*numel)
    "trapz": (lambda: fnp.trapz(_v100), 4 * 100),
    "sum": (lambda: fnp.sum(_v100), 99),  # numel - M
    "mean": (lambda: fnp.mean(_v100), 100),  # numel
    "std": (lambda: fnp.std(_v100), 4 * 100 + 1),
    "nanstd": (lambda: fnp.nanstd(_v100), 4 * 100 + 1),
    "ptp": (lambda: fnp.ptp(_v100), 2 * (100 - 1) + 1),
    "median": (lambda: fnp.median(_v100), 100),
    "percentile": (lambda: fnp.percentile(_v100, 50), 100),
    "quantile": (lambda: fnp.quantile(_v100, 0.5), 100),
    "nanquantile": (lambda: fnp.nanquantile(_v100, 0.5), 100),
    # ---- Diff / gradient --------------------------------------------------
    "diff": (lambda: fnp.diff(_v100), 99),
    "ediff1d": (lambda: fnp.ediff1d(_v100), 99),
    "gradient": (lambda: fnp.gradient(_v100), 2 * 100 - 4),  # 196
    # ---- Miscellaneous counted_custom -------------------------------------
    # clip: 2 bounds → 2 compare-selects/elem → 2*numel(output); old pin was 1*numel
    "clip": (lambda: fnp.clip(_v100, -1.0, 1.0), 200),
    # where (1-arg): cond.size = 100; equivalent to nonzero -> deducted under
    # "nonzero" (alias parity), charged numel at weight 1.0. This pin exercises
    # that path -- unit-weight flops_used is unaffected by the op_name it logs
    # under. 3-arg where (select) is now charged too: 4*numel(broadcast output)
    # at weight 4.0 (unit weight here -> 1*numel), see test_triage_price_pins.py.
    "where": (lambda: fnp.where(_v100_pos), 100),
    "tile": (lambda: fnp.tile(_v100, 3), 300),
    "repeat": (lambda: fnp.repeat(_v100, 3), 300),
    "corrcoef": (
        lambda: fnp.corrcoef(_sq10),
        2410,
    ),  # 2*f^2*s+2*f*s+2*f^2+f; f=s=10 -> 2000+200+200+10
    "cov": (lambda: fnp.cov(_sq10), 2200),  # 2*f^2*s+2*f*s; f=s=10 -> 2000+200
    "linalg.trace": (lambda: fnp.linalg.trace(_sq10), 10),
    "trace": (lambda: fnp.trace(_sq10), 10),
    "linalg.cross": (lambda: fnp.linalg.cross(_a3, _b3), 3 * 3),
    # ---- Copy / gather / view ops (audit-completion pins) ------------------
    # diag extract: 2-D (10,10) → view → 0 (Task 7)
    "diag": (lambda: fnp.diag(_sq10_2d), 0),
    # diagonal: returns a view → 0 FLOPs
    "diagonal": (lambda: fnp.diagonal(_sq10_2d), 0),
    # take_along_axis: gather tier weight 4.0; conftest resets weights → flop_cost = numel(output) = 30
    # (billing = weight * flop_cost = 4.0 * 30 = 120 in production; 1.0 * 30 = 30 here)
    "take_along_axis": (lambda: fnp.take_along_axis(_sq10_2d, _idx10x3, axis=1), 30),
    # argwhere: numel(input) at weight 1.0; _v100_pos is pre-built bool (100 elems)
    "argwhere": (lambda: fnp.argwhere(_v100_pos), 100),
    # bmat: numel(output) of 6×6 block matrix = 36
    "bmat": (lambda: fnp.bmat([[_sq3a, _sq3b], [_sq3a, _sq3b]]), 36),
    # fromiter: 10 elements
    "fromiter": (lambda: fnp.fromiter(range(10), float), 10),
    # compress: len(cond)=5 + 4*numel(out)=4*3=12 → 17
    "compress": (lambda: fnp.compress(_cond5, _v5_float), 17),
    # packbits: numel(input)=8
    "packbits": (lambda: fnp.packbits(_v8_bits), 8),
    # unwrap: 13 * numel(input) = 13 * 100 (steps 8/12, 3-arg where selects,
    # are charged like every other pass now that 3-arg where itself bills)
    "unwrap": (lambda: fnp.unwrap(_v100), 13 * 100),
}

# ---------------------------------------------------------------------------
# DEFERRED: ops not probed here, with explicit reasons.
# ---------------------------------------------------------------------------

DEFERRED: dict[str, str] = {
    # ---- counted_random_method ops (89 total) ------------------------------
    "random.rand": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.randn": "random_sampler family; flop_cost=numel; weight=16.0",
    "random.normal": "random_sampler family; flop_cost=numel; weight=16.0",
    "random.standard_normal": "random_sampler family; flop_cost=numel; weight=16.0",
    "random.uniform": "random_sampler affine map; flop_cost=3*numel (draw + low+(high-low)*U); weight=1.0; probed in test_family_defaults_random_sampler",
    "random.random": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.random_sample": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.ranf": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.sample": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.randint": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.random_integers": "blacklisted; deprecated alias; intentionally unsupported (raises AttributeError)",
    "random.exponential": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.poisson": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.binomial": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.beta": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.chisquare": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.choice": "random_sampler family; choice_cost formula",
    "random.default_rng": "free — constructs Generator; 0 FLOPs",
    "random.dirichlet": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.f": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.gamma": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.geometric": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.gumbel": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.hypergeometric": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.laplace": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.logistic": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.lognormal": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.logseries": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.multinomial": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.multivariate_normal": "composite: svd_cost(d,d,with_vectors=True) + 2Nd^2 + 16Nd",
    "random.negative_binomial": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.noncentral_chisquare": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.noncentral_f": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.pareto": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.permutation": "random_sampler family; cost=shape[axis]",
    "random.power": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.rayleigh": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.shuffle": "random_sampler family; cost=shape[axis]",
    "random.standard_cauchy": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.standard_exponential": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.standard_gamma": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.standard_t": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.triangular": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.vonmises": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.wald": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.weibull": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.zipf": "random_sampler family; flop_cost=numel; weight=1.0",
    "random.bytes": "random_sampler family; cost=length",
    "random.get_state": "free — state accessor",
    "random.seed": "free — state setter",
    "random.set_state": "free — state setter",
    # Generator methods
    "random.Generator.beta": "Generator family; numel",
    "random.Generator.binomial": "Generator family; numel",
    "random.Generator.bytes": "Generator family; cost=length",
    "random.Generator.chisquare": "Generator family; numel",
    "random.Generator.choice": "Generator family; choice_cost",
    "random.Generator.dirichlet": "Generator family; numel",
    "random.Generator.exponential": "Generator family; numel",
    "random.Generator.f": "Generator family; numel",
    "random.Generator.gamma": "Generator family; numel",
    "random.Generator.geometric": "Generator family; numel",
    "random.Generator.gumbel": "Generator family; numel",
    "random.Generator.hypergeometric": "Generator family; numel",
    "random.Generator.integers": "Generator family; numel",
    "random.Generator.laplace": "Generator family; numel",
    "random.Generator.logistic": "Generator family; numel",
    "random.Generator.lognormal": "Generator family; numel",
    "random.Generator.logseries": "Generator family; numel",
    "random.Generator.multinomial": "Generator family; numel",
    "random.Generator.multivariate_hypergeometric": "Generator family; numel",
    "random.Generator.multivariate_normal": "composite svd_cost(d,d,with_vectors=True)+2Nd^2+16Nd",
    "random.Generator.negative_binomial": "Generator family; numel",
    "random.Generator.noncentral_chisquare": "Generator family; numel",
    "random.Generator.noncentral_f": "Generator family; numel",
    "random.Generator.normal": "Generator family; numel (weight=16)",
    "random.Generator.pareto": "Generator family; numel",
    "random.Generator.permutation": "Generator family; shape[axis]",
    "random.Generator.permuted": "Generator family; numel(input)",
    "random.Generator.poisson": "Generator family; numel",
    "random.Generator.power": "Generator family; numel",
    "random.Generator.random": "Generator family; numel",
    "random.Generator.rayleigh": "Generator family; numel",
    "random.Generator.shuffle": "Generator family; shape[axis]",
    "random.Generator.standard_cauchy": "Generator family; numel",
    "random.Generator.standard_exponential": "Generator family; numel",
    "random.Generator.standard_gamma": "Generator family; numel",
    "random.Generator.standard_normal": "Generator family; numel (weight=16)",
    "random.Generator.standard_t": "Generator family; numel",
    "random.Generator.triangular": "Generator family; numel",
    "random.Generator.uniform": "Generator family; numel",
    "random.Generator.vonmises": "Generator family; numel",
    "random.Generator.wald": "Generator family; numel",
    "random.Generator.weibull": "Generator family; numel",
    "random.Generator.zipf": "Generator family; numel",
    "random.Generator.bit_generator": "free_random_method — attribute accessor",
    "random.Generator.spawn": "free_random_method — returns child generators",
    # RandomState methods
    "random.RandomState.beta": "RandomState family; numel",
    "random.RandomState.binomial": "RandomState family; numel",
    "random.RandomState.bytes": "RandomState family; cost=length",
    "random.RandomState.chisquare": "RandomState family; numel",
    "random.RandomState.choice": "RandomState family; choice_cost",
    "random.RandomState.dirichlet": "RandomState family; numel",
    "random.RandomState.exponential": "RandomState family; numel",
    "random.RandomState.f": "RandomState family; numel",
    "random.RandomState.gamma": "RandomState family; numel",
    "random.RandomState.geometric": "RandomState family; numel",
    "random.RandomState.gumbel": "RandomState family; numel",
    "random.RandomState.hypergeometric": "RandomState family; numel",
    "random.RandomState.laplace": "RandomState family; numel",
    "random.RandomState.logistic": "RandomState family; numel",
    "random.RandomState.lognormal": "RandomState family; numel",
    "random.RandomState.logseries": "RandomState family; numel",
    "random.RandomState.multinomial": "RandomState family; numel",
    "random.RandomState.multivariate_normal": "composite svd_cost(d,d,with_vectors=True)+2Nd^2+16Nd",
    "random.RandomState.negative_binomial": "RandomState family; numel",
    "random.RandomState.noncentral_chisquare": "RandomState family; numel",
    "random.RandomState.noncentral_f": "RandomState family; numel",
    "random.RandomState.normal": "RandomState family; numel (weight=16)",
    "random.RandomState.pareto": "RandomState family; numel",
    "random.RandomState.permutation": "RandomState family; shape[axis]",
    "random.RandomState.poisson": "RandomState family; numel",
    "random.RandomState.power": "RandomState family; numel",
    "random.RandomState.rand": "RandomState family; numel",
    "random.RandomState.randint": "RandomState family; numel",
    "random.RandomState.randn": "RandomState family; numel (weight=16)",
    "random.RandomState.random": "RandomState family; numel",
    "random.RandomState.random_integers": "RandomState family; numel",
    "random.RandomState.random_sample": "RandomState family; numel",
    "random.RandomState.rayleigh": "RandomState family; numel",
    "random.RandomState.shuffle": "RandomState family; shape[axis]",
    "random.RandomState.standard_cauchy": "RandomState family; numel",
    "random.RandomState.standard_exponential": "RandomState family; numel",
    "random.RandomState.standard_gamma": "RandomState family; numel",
    "random.RandomState.standard_normal": "RandomState family; numel (weight=16)",
    "random.RandomState.standard_t": "RandomState family; numel",
    "random.RandomState.tomaxint": "RandomState family; numel",
    "random.RandomState.triangular": "RandomState family; numel",
    "random.RandomState.uniform": "RandomState family; numel",
    "random.RandomState.vonmises": "RandomState family; numel",
    "random.RandomState.wald": "RandomState family; numel",
    "random.RandomState.weibull": "RandomState family; numel",
    "random.RandomState.zipf": "RandomState family; numel",
    "random.RandomState.get_state": "free_random_method — state accessor",
    "random.RandomState.seed": "free_random_method — state setter",
    "random.RandomState.set_state": "free_random_method — state setter",
    # ---- Stats ops ---------------------------------------------------------
    # All composite kernels, weight 1.0; exact values read from _deduct_and_call() args.
    # Pinned in OP_EXPECTATIONS: norm.pdf=27, norm.cdf=48, norm.ppf=83,
    #   lognorm.ppf=106, lognorm.pdf=62, lognorm.cdf=70, truncnorm.ppf=81,
    #   laplace.cdf=40, laplace.ppf=51, uniform.cdf=4, cauchy.pdf=6.
    # DEFERRED (not probed individually; formula documented in cost-model.md):
    "stats.uniform.pdf": "composite; 1 FLOP/elem (trivial range-check only); weight 1.0",
    "stats.uniform.ppf": "composite; 1 FLOP/elem; weight 1.0",
    "stats.expon.pdf": "composite; 22 FLOPs/elem (z+exp+where); weight 1.0",
    "stats.expon.cdf": "composite; 22 FLOPs/elem (z+exp+where); weight 1.0",
    "stats.expon.ppf": "composite; 27 FLOPs/elem (log1p+where); weight 1.0",
    "stats.cauchy.cdf": "composite; 20 FLOPs/elem (z+arctan+arith); weight 1.0",
    "stats.cauchy.ppf": "composite; 28 FLOPs/elem (tan+loc/scale+where); weight 1.0",
    "stats.logistic.pdf": "composite; 23 FLOPs/elem (z+exp+arith); weight 1.0",
    "stats.logistic.cdf": "composite; 21 FLOPs/elem (z+exp+arith); weight 1.0",
    "stats.logistic.ppf": "composite; 28 FLOPs/elem (log+loc/scale+where); weight 1.0",
    "stats.laplace.pdf": "composite; 22 FLOPs/elem (abs+exp+scale); weight 1.0",
    "stats.truncnorm.pdf": "composite; 28 FLOPs/elem (norm.pdf+cdf-norm); weight 1.0",
    "stats.truncnorm.cdf": "composite; 51 FLOPs/elem (affine+norm.cdf+selects); weight 1.0",
    # ---- counted_custom: copy / gather / scatter / structure ops ----------
    "array": "numel(input); plain copy",
    "full": "numel; scalar broadcast",
    "full_like": "numel; trivial",
    # Task 4: value-writing creation & layout copies (formerly free tier)
    "ones": "numel(output); constant fill",
    "ones_like": "numel(output); constant fill",
    "eye": "diagonal length written; structural constructor",
    "identity": "diagonal length written (=n); structural constructor",
    "copy": "numel(input); materializing copy",
    "reshape": "numel(input); billed regardless of view-vs-copy",
    "ravel": "numel(input); billed regardless of view-vs-copy",
    "require": "numel(input); billed regardless of view-vs-copy",
    "fft.fftshift": "numel(output); roll-based reindex, no arithmetic",
    "fft.ifftshift": "numel(output); roll-based reindex, no arithmetic",
    "diag": "pinned in OP_EXPECTATIONS (2-D extract: 0, view; 1-D construct: v.shape[0])",
    "concatenate": "numel(output); trivial copy",
    "concat": "numel(output); numpy 2.x alias for concatenate",
    "stack": "numel(output); trivial copy",
    "vstack": "numel(output); trivial copy",
    "dstack": "numel(output); trivial copy",
    "block": "numel(output); trivial copy",
    "bmat": "pinned in OP_EXPECTATIONS (numel(output) weight 1.0)",
    "roll": "numel(output); materializing copy",
    "hstack": "numel(output); materializing copy",
    "column_stack": "numel(output); materializing copy (1-D to 2-D columns)",
    "row_stack": "numel(output); alias for vstack",
    "tril": "elements at/below kth diagonal via _triangle_kept; batch leading dims multiply; weight 1.0",
    "triu": "elements at/above kth diagonal via _triangle_kept; batch leading dims multiply; weight 1.0",
    "einsum_path": "path planning only; returns list+string, no numeric FLOPs",
    "histogram_bin_edges": "numel(a); bin-edge computation",
    "pad": "numel(output) + mode extras (movement 0; linear_ramp/odd +(out-in); stat modes +stat cost); mode=<callable> raises",
    "resize": "numel(output)",
    "meshgrid": "numel(output) per array; sparse=True bills sum(input lengths); copy=False bills 1",
    "indices": "numel(dense output)",
    "isnan": "numel(input); element comparison",
    "isinf": "numel(input)",
    "isfinite": "numel(input)",
    "allclose": "numel(broadcast); test_cost_formula_vs_code.py",
    "array_equal": "numel(a); equality scan",
    "array_equiv": "numel(a); equiv scan",
    "asarray_chkfinite": "numel(input); finite check",
    "nonzero": "numel(input)",
    "flatnonzero": "numel(input)",
    "argwhere": "pinned in OP_EXPECTATIONS (numel(input) weight 1.0)",
    "select": "numel(output) * len(condlist)",
    "piecewise": "numel(input) * len(condlist); local_callback",
    "apply_along_axis": "numel(output); local_callback",
    "apply_over_axes": "numel(output); local_callback",
    "fromfunction": "numel(output); local_callback",
    "fromiter": "pinned in OP_EXPECTATIONS (numel(output) weight 1.0)",
    "diagonal": "pinned in OP_EXPECTATIONS (view; 0 FLOPs)",
    "linalg.diagonal": "delegates to fnp.diagonal; 0 FLOPs (view)",
    "take": "numel(output) gather",
    "take_along_axis": "pinned in OP_EXPECTATIONS (numel(output) weight 4.0 gather tier)",
    "choose": "numel(output) gather tier",
    "compress": "pinned in OP_EXPECTATIONS (len(cond)+4*numel(out) weight 1.0)",
    "extract": "numel(input) gather",
    "place": "numel(input) scatter",
    "put": "numel(indices) scatter at gather tier",
    "put_along_axis": "elements scattered; scatter tier weight 1.0",
    "putmask": "numel(input) scatter",
    "delete": "numel(output); surviving elements copied",
    "insert": "numel(output); materializing copy",
    "append": "numel(output) = arr.size + values.size; concatenate family",
    "copyto": "numel(dst) per element written (or popcount of where= when masked)",
    "trim_zeros": "numel(input); value scan, like nonzero",
    # Task 8: index generators -- numel of the returned index arrays, weight
    # 1.0, dtype-neutral (dtypes=()); pinned in
    # tests/test_triage_price_pins.py::test_index_generators_bill_their_outputs
    # and siblings.
    "ravel_multi_index": "numel(output) = N; dtype-neutral",
    "mask_indices": "numel(output) = 2*k; dtype-neutral (mask scan itself is unbilled -- see test_mask_indices_fnp_mask_func_hits_preexisting_nonzero_bug)",
    "tri": "numel(output); NOT dtype-neutral, bills the actual (possibly requested) output dtype -- mirrors full/ones/eye/identity",
    "tril_indices": "numel(output) = numel of the returned index arrays; dtype-neutral",
    "tril_indices_from": "numel(output) = numel of the returned index arrays; dtype-neutral (only arr.shape is read)",
    "triu_indices": "numel(output) = numel of the returned index arrays; dtype-neutral",
    "triu_indices_from": "numel(output) = numel of the returned index arrays; dtype-neutral (only arr.shape is read)",
    "diag_indices": "numel(output) = numel of the returned index arrays; dtype-neutral",
    "diag_indices_from": "numel(output) = numel of the returned index arrays; dtype-neutral (only arr.shape is read)",
    "unravel_index": "numel(output) = numel of the returned index arrays; dtype-neutral",
    "broadcast_shapes": "sum of len(shape) across input shape tuples (floor 1); dtype-neutral, no array operands",
    "ix_": "numel(output)",
    "diagflat": "numel(v)",
    "fill_diagonal": "min(m,n)",
    "packbits": "pinned in OP_EXPECTATIONS (numel(input) weight 1.0)",
    "unpackbits": "8*n",
    "unwrap": "pinned in OP_EXPECTATIONS (13*numel(input) weight 1.0; steps 8/12 are 3-arg where selects, charged like the rest since Task 6's select-class rework)",
    # unstack left this dict in Task 5 -- it now bills 0 FLOPs (free-tier
    # view, NumPy 2.1+); see tests/test_triage_price_pins.py::test_unstack_stays_free
    # and tests/test_view_semantics_lock.py.
    "unique_all": "n*ceil(log2(n)); unique family",
    "unique_counts": "n*ceil(log2(n)); unique family",
    "unique_inverse": "n*ceil(log2(n)); unique family",
    "unique_values": "n*ceil(log2(n)); unique family",
    "vecdot": "2*N-1 per output elem; test_cost_formula_vs_code.py",
    "matvec": "m*(2k-1); counted_binary auto-classified",
    "vecmat": "m*(2k-1); counted_binary auto-classified",
    "linalg.tensordot": "delegates to fnp.tensordot; COVERED_ELSEWHERE (_FMA)",
    "linalg.vecdot": "delegates to fnp.vecdot; test_cost_formula_vs_code.py",
    "linalg.outer": "pinned in OP_EXPECTATIONS",
    "linalg.matrix_power": "pinned in OP_EXPECTATIONS",
    "linalg.multi_dot": "COVERED_ELSEWHERE (_FMA)",
    "linalg.matmul": "pinned in OP_EXPECTATIONS",
    "linalg.cross": "pinned in OP_EXPECTATIONS (delegates to fnp.cross; 3*numel(output))",
    "in1d": "same model as isin; deprecated (removed in numpy >=2.4)",
    "correlate": "same model as convolve; COVERED_ELSEWHERE (_FMA)",
    "cov": "pinned in OP_EXPECTATIONS",
    "corrcoef": "pinned in OP_EXPECTATIONS",
    "sort_complex": "pinned in OP_EXPECTATIONS",
    "linalg.trace": "pinned in OP_EXPECTATIONS",
    "trace": "pinned in OP_EXPECTATIONS",
    # ---- newly re-registered from blacklist (audit-2 fix): no billing change --
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_conformance():
    """Each OP_EXPECTATIONS entry: charged == documented formula."""
    failures = []
    for op, (fn, expected) in OP_EXPECTATIONS.items():
        actual = _cost(fn)
        if actual != expected:
            failures.append(f"  {op}: got {actual}, expected {expected}")
    if failures:
        pytest.fail("Cost convention violations:\n" + "\n".join(failures))


def test_family_defaults_elementwise():
    """Elementwise unary and binary: flop_cost = numel(output)."""
    v = fnp.asarray(_rng.standard_normal(50))
    w = fnp.asarray(_rng.standard_normal(50))
    # Unary (numel=50 for all)
    assert _cost(lambda: fnp.abs(v)) == 50
    assert _cost(lambda: fnp.negative(v)) == 50
    assert _cost(lambda: fnp.ceil(v)) == 50
    assert _cost(lambda: fnp.floor(v)) == 50
    assert _cost(lambda: fnp.sign(v)) == 50
    assert _cost(lambda: fnp.exp(v)) == 50
    # log(abs(v)) = 2 ops × 50 = 100
    assert _cost(lambda: fnp.log(fnp.abs(v))) == 100
    # Binary (numel=50)
    assert _cost(lambda: fnp.add(v, w)) == 50
    assert _cost(lambda: fnp.subtract(v, w)) == 50
    assert _cost(lambda: fnp.multiply(v, w)) == 50
    assert _cost(lambda: fnp.maximum(v, w)) == 50
    assert _cost(lambda: fnp.greater(v, w)) == 50


def test_family_defaults_reduction():
    """Reduction: flop_cost = numel(input) - M (full) or similar."""
    v = fnp.asarray(_rng.standard_normal(100))
    # Pre-compute bool to avoid charging comparison op inside lambda
    vbool = fnp.asarray(_rng.standard_normal(100) > 0)
    # Full reduction: M=1 → cost = 100-1 = 99
    assert _cost(lambda: fnp.sum(v)) == 99
    assert _cost(lambda: fnp.prod(v)) == 99
    assert _cost(lambda: fnp.any(vbool)) == 99
    assert _cost(lambda: fnp.all(vbool)) == 99
    assert _cost(lambda: fnp.cumsum(v)) == 99  # cumulative: numel - M
    # Partial reduction along axis=1: (50,2) → M=50
    a = fnp.asarray(_rng.standard_normal((50, 2)))
    assert _cost(lambda: fnp.sum(a, axis=1)) == 100 - 50


def test_family_defaults_free():
    """Free / view ops: cost = 0.

    reshape/ones/ones_like/fft.fftshift/fft.ifftshift moved out of this test
    in Task 4 -- they now bill numel(input)/numel(output) (weight 1.0); see
    OP_EXPECTATIONS / DEFERRED below and tests/test_triage_price_pins.py.
    """
    v = fnp.asarray(_rng.standard_normal(100))
    sq = fnp.asarray(_rng.standard_normal((4, 4)))
    assert _cost(lambda: fnp.transpose(sq)) == 0
    assert _cost(lambda: fnp.zeros(100)) == 0
    assert _cost(lambda: fnp.empty(100)) == 0
    assert _cost(lambda: fnp.zeros_like(v)) == 0
    assert _cost(lambda: fnp.linalg.matrix_transpose(sq)) == 0


def test_family_defaults_random_sampler():
    """Random samplers: flop_cost = numel(output) [weight varies per op]."""
    import flopscope.numpy.random as fnpr

    assert _cost(lambda: fnpr.rand(100)) == 100
    assert (
        _cost(lambda: fnpr.uniform(0.0, 1.0, 100)) == 3 * 100
    )  # affine exception (draw + low+(high-low)*U)
    assert _cost(lambda: fnpr.random(100)) == 100
    assert _cost(lambda: fnpr.randint(0, 100, 100)) == 100
    assert _cost(lambda: fnpr.exponential(1.0, 100)) == 100
    assert _cost(lambda: fnpr.poisson(1.0, 100)) == 100


def test_completeness():
    """Every billed op in the registry must be classified in exactly one of:
    - OP_EXPECTATIONS (op-specific probe with exact expected value),
    - _UNARY_FAMILY / _BINARY_FAMILY / _REDUCTION_FAMILY (category-level rule),
    - DEFERRED (documented reason),
    - COVERED_ELSEWHERE (exact probes in another test file).
    Zero unclassified are allowed.
    """
    BILLED = frozenset(
        op
        for op, e in REGISTRY.items()
        if e["category"]
        in {
            "counted_unary",
            "counted_binary",
            "counted_reduction",
            "counted_custom",
            "counted_random_method",
        }
    )

    classified: set[str] = set()
    classified.update(OP_EXPECTATIONS)
    classified.update(_UNARY_FAMILY)
    classified.update(_BINARY_FAMILY)
    classified.update(_REDUCTION_FAMILY)
    classified.update(DEFERRED)
    classified.update(COVERED_ELSEWHERE)

    unclassified = BILLED - classified
    if unclassified:
        pytest.fail(
            f"UNCLASSIFIED billed ops ({len(unclassified)}) — "
            f"add each to OP_EXPECTATIONS, a FAMILY set, "
            f"or DEFERRED with a reason:\n"
            + "\n".join(f"  {op}" for op in sorted(unclassified))
        )
