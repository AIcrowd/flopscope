"""Compute-dtype conformance: billed rate covers numpy's actual compute dtype.

Every CHARGED registry op is either probed here or listed in SKIPPED with a
reason. The oracle: the rate of the billed (resolved) dtype must be >= both
the rate of the NEP-50-promoted input dtypes and the rate of the actual
result's dtype. Ops whose result dtype overstates their arithmetic (index
outputs) are exempted per-op below; ops that declare dtype-neutrality
(movement/shuffle family, resolved_dtype None) are skipped as a category —
width-independence is their documented cost model.

int32 probes discriminate: rate(int32) == 1.0, while a kernel that really
computes in float64/complex128/int64 must bill rate 2.0.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
import flopscope.stats as fstats
from flopscope._dtype_billing import rate_for
from flopscope._registry import REGISTRY
from flopscope._weights import get_weight, load_weights, reset_weights
from flopscope.errors import UnsupportedFunctionError

# --- probe inputs: int32 (the discriminating dtype), built at module import,
# --- outside any BudgetContext.
V = np.arange(1, 9, dtype=np.int32)  # 1-D, len 8, strictly positive
V01 = (np.arange(8) % 2).astype(np.int32)  # zeros and ones
M = np.eye(4, dtype=np.int32) * 3 + 1  # 4x4, nonsingular
COND = np.array([True, False] * 4)

# Extra shared probe inputs for calls the seed set didn't need.
V3 = V[:3]  # length-3 (cross-product domain)
V4 = V[:4]  # matches M's contracted dimension
SHIFT = (V01 % 2).astype(np.int32) + 1  # 1s and 2s: safe int shift amounts

# bool probe: the dtype whose unary-ufunc size-mapping target (float16) diverges
# most from what several composites actually compute in (angle(bool_) -> float64).
VB = (np.arange(8) % 2).astype(bool)

# float64 fixtures for the dtype-resolving gather/assembly ops (choose/block/
# bmat). They read/assemble from these arrays, so a float64 probe drives the
# result-dtype floor to 2.0 -- a passing billed rate proves they resolve the
# promoted operand dtype, not the neutral 1.0 they billed before set_dtypes().
CHOOSE_IDX = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.intp)  # 8 gather indices
F64_8 = np.arange(1, 9, dtype=np.float64)  # float64 len-8
BLK64 = F64_8.reshape(2, 4)  # 2x4 float64 block
XF = V.astype(np.float32)  # generic real-domain float32 probe
Q01 = XF / 10.0  # 0.1..0.8: valid probability domain (ppf)

# Two on-disk fixtures for the from{file,string} family (module-scoped, built
# once outside any BudgetContext; cleaned up isn't required for CI workers).
_PROBE_DIR = tempfile.mkdtemp(prefix="flopscope_dtype_sweep_")
_FROMFILE_PATH = os.path.join(_PROBE_DIR, "probe.bin")
V.tofile(_FROMFILE_PATH)


def _unary(name: str) -> Callable[[], Any]:
    return lambda: getattr(fnp, name)(V)


def _unary_bool(name: str) -> Callable[[], Any]:
    return lambda: getattr(fnp, name)(VB)


def _binary(name: str) -> Callable[[], Any]:
    return lambda: getattr(fnp, name)(V, V)


def _reduction(name: str) -> Callable[[], Any]:
    return lambda: getattr(fnp, name)(V)


def _shift(name: str) -> Callable[[], Any]:
    return lambda: getattr(fnp, name)(V, SHIFT)


def _stats_call(
    dist_module: Any, method: str, arg: np.ndarray, kwargs: dict
) -> Callable[[], Any]:
    return lambda: getattr(dist_module, method)(arg, **kwargs)


# op name -> zero-arg callable executing the op on int32 probes.
PROBES: dict[str, Callable[[], Any]] = {
    # elementwise (Task 6 wired these — the sweep locks them)
    "add": lambda: fnp.add(V, V),
    "multiply": lambda: fnp.multiply(V, V),
    "divide": lambda: fnp.divide(V, V),
    "exp": lambda: fnp.exp(V),
    "sqrt": lambda: fnp.sqrt(V),
    "float_power": lambda: fnp.float_power(V, V),
    # reductions
    "sum": lambda: fnp.sum(V),
    "mean": lambda: fnp.mean(V),
    "var": lambda: fnp.var(V),
    "std": lambda: fnp.std(V),
    # fft / linalg
    "fft.fft": lambda: fnp.fft.fft(V),
    "linalg.solve": lambda: fnp.linalg.solve(M, V4),
    "linalg.inv": lambda: fnp.linalg.inv(M),
    # composites known-affected (fixed in this task)
    "polyfit": lambda: fnp.polyfit(V, V, 2),
    "roots": lambda: fnp.roots(np.array([1, -3, 2], dtype=np.int32)),
    "gradient": lambda: fnp.gradient(V),
    "average": lambda: fnp.average(V),
    "median": lambda: fnp.median(V),
    "vander": lambda: fnp.vander(V),
}

# ---------------------------------------------------------------------------
# counted_unary — the whole family shares the (V,) -> fnp.<name>(V) shape.
# Domain warnings (e.g. arccos/arcsin/arctanh outside [-1, 1]) are cosmetic:
# the dtype the call resolves to does not depend on producing finite values.
# ---------------------------------------------------------------------------
_UNARY_OPS = [
    "abs",
    "absolute",
    "acos",
    "acosh",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "around",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "i0",
    "invert",
    "iscomplex",
    "isneginf",
    "isposinf",
    "isreal",
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
    "real_if_close",
    "reciprocal",
    "rint",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "spacing",
    "square",
    "tan",
    "tanh",
    "trunc",
]
for _name in _UNARY_OPS:
    PROBES[_name] = _unary(_name)
# isclose is registry-categorized counted_unary but takes (a, b).
PROBES["isclose"] = lambda: fnp.isclose(V, V)
# modf/frexp are multi-output unary ufuncs (_counted_unary_multi).
PROBES["modf"] = lambda: fnp.modf(V)
PROBES["frexp"] = lambda: fnp.frexp(V)

# ---------------------------------------------------------------------------
# counted_binary — the whole family shares the (V, V) -> fnp.<name>(V, V)
# shape except for the shift family (needs a small positive exponent/shift
# operand) and the matrix-vector contraction trio.
# ---------------------------------------------------------------------------
_BINARY_OPS = [
    "arctan2",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "copysign",
    "divmod",
    "equal",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "lcm",
    "ldexp",
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
    "nextafter",
    "not_equal",
    "pow",
    "power",
    "remainder",
    "subtract",
    "true_divide",
]
for _name in _BINARY_OPS:
    PROBES[_name] = _binary(_name)
for _name in (
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
):
    PROBES[_name] = _shift(_name)
PROBES["matvec"] = lambda: fnp.matvec(M, V4)
PROBES["vecmat"] = lambda: fnp.vecmat(V4, M)
PROBES["vecdot"] = lambda: fnp.vecdot(V4, V4)

# ---------------------------------------------------------------------------
# counted_reduction — the whole family shares the (V,) -> fnp.<name>(V) shape
# except the percentile/quantile quartet (needs a q argument).
# ---------------------------------------------------------------------------
_REDUCTION_OPS = [
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "cumulative_prod",
    "cumulative_sum",
    "max",
    "min",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
    "prod",
    "ptp",
]
for _name in _REDUCTION_OPS:
    PROBES[_name] = _reduction(_name)
PROBES["percentile"] = lambda: fnp.percentile(V, 50)
PROBES["nanpercentile"] = lambda: fnp.nanpercentile(V, 50)
PROBES["quantile"] = lambda: fnp.quantile(V, 0.5)
PROBES["nanquantile"] = lambda: fnp.nanquantile(V, 0.5)

# ---------------------------------------------------------------------------
# fft — every transform, 1-D (V) or 2-D (V.reshape(2, 4)) as required.
# ---------------------------------------------------------------------------
_V2D = V.reshape(2, 4)
PROBES.update(
    {
        "fft.fft2": lambda: fnp.fft.fft2(_V2D),
        "fft.fftfreq": lambda: fnp.fft.fftfreq(8),
        "fft.fftn": lambda: fnp.fft.fftn(_V2D),
        "fft.hfft": lambda: fnp.fft.hfft(V),
        "fft.ifft": lambda: fnp.fft.ifft(V),
        "fft.ifft2": lambda: fnp.fft.ifft2(_V2D),
        "fft.ifftn": lambda: fnp.fft.ifftn(_V2D),
        "fft.ihfft": lambda: fnp.fft.ihfft(V),
        "fft.irfft": lambda: fnp.fft.irfft(V),
        "fft.irfft2": lambda: fnp.fft.irfft2(_V2D),
        "fft.irfftn": lambda: fnp.fft.irfftn(_V2D),
        "fft.rfft": lambda: fnp.fft.rfft(V),
        "fft.rfft2": lambda: fnp.fft.rfft2(_V2D),
        "fft.rfftfreq": lambda: fnp.fft.rfftfreq(8),
        "fft.rfftn": lambda: fnp.fft.rfftn(_V2D),
    }
)

# ---------------------------------------------------------------------------
# gather/scatter (explicit-indexing weight flip, cost-model triage Task 2):
# take/take_along_axis (gather tier) and put/place/putmask/put_along_axis/
# fill_diagonal/extract/compress (scatter tier) all declare dtypes=() on the
# array being read from or written to (NOT the index/condition operand), so
# billing scales with that array's own dtype. The mutators copy V/M first --
# each lambda makes its own fresh copy on every call, so no cross-probe
# state leaks. (choose bills its promoted result dtype via set_dtypes -- see
# the assembly/gather-resolving block just below.)
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "take": lambda: fnp.take(V, V01),
        "take_along_axis": lambda: fnp.take_along_axis(V, V01, axis=0),
        "put": lambda: fnp.put(
            V.copy(),
            np.array([0, 1], dtype=np.int32),
            np.array([9, 10], dtype=np.int32),
        ),
        "place": lambda: fnp.place(V.copy(), COND, np.array([9], dtype=np.int32)),
        "putmask": lambda: fnp.putmask(V.copy(), COND, np.array([9], dtype=np.int32)),
        "put_along_axis": lambda: fnp.put_along_axis(
            V.copy(), V01, np.full(8, 9, dtype=np.int32), 0
        ),
        "fill_diagonal": lambda: fnp.fill_diagonal(M.copy(), 9),
        "extract": lambda: fnp.extract(COND, V),
        "compress": lambda: fnp.compress(COND, V),
        # getitem (Task 11, new billed surface): FlopscopeArray.__getitem__
        # declares dtypes=(self.dtype,) -- the array being gathered from, not
        # the fancy-index operand -- same convention as take/compress above.
        "getitem": lambda: fnp.asarray(V)[fnp.asarray(V01)],
    }
)

# ---------------------------------------------------------------------------
# choose / block / bmat -- deduct_after ops that declare their billed dtype
# from the assembled result via set_dtypes((result.dtype,)). result.dtype is
# numpy's promotion of the operands (np.result_type of the choices / blocks),
# so they bill the widest operand's rate, matching take's dtype-awareness.
# Probed with float64 operands (rate floor 2.0) so a passing rate proves the
# promotion is billed, not the neutral 1.0 they discounted to before the fix.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "choose": lambda: fnp.choose(CHOOSE_IDX, [F64_8, F64_8 * 2]),
        "block": lambda: fnp.block([[BLK64, BLK64]]),
        "bmat": lambda: fnp.bmat([[BLK64, BLK64]]),
    }
)

# ---------------------------------------------------------------------------
# linalg — LAPACK-backed decompositions/solvers on M (4x4, nonsingular).
# linalg.matmul/outer/tensordot/vecdot are pure aliases with no deduct() of
# their own (see SKIPPED) so they are not probed here.
# ---------------------------------------------------------------------------
_MPSD = M @ M.T + np.eye(4, dtype=np.int32) * 20  # symmetric positive-definite
PROBES.update(
    {
        "linalg.cholesky": lambda: fnp.linalg.cholesky(_MPSD),
        "linalg.cond": lambda: fnp.linalg.cond(M),
        "linalg.cross": lambda: fnp.linalg.cross(V3, V3),
        "linalg.det": lambda: fnp.linalg.det(M),
        "linalg.eig": lambda: fnp.linalg.eig(M),
        "linalg.eigh": lambda: fnp.linalg.eigh(_MPSD),
        "linalg.eigvals": lambda: fnp.linalg.eigvals(M),
        "linalg.eigvalsh": lambda: fnp.linalg.eigvalsh(_MPSD),
        "linalg.lstsq": lambda: fnp.linalg.lstsq(M, V4, rcond=None),
        "linalg.matrix_norm": lambda: fnp.linalg.matrix_norm(M),
        "linalg.matrix_power": lambda: fnp.linalg.matrix_power(M, 2),
        "linalg.matrix_rank": lambda: fnp.linalg.matrix_rank(M),
        "linalg.multi_dot": lambda: fnp.linalg.multi_dot([M, M, V4]),
        "linalg.norm": lambda: fnp.linalg.norm(V),
        "linalg.pinv": lambda: fnp.linalg.pinv(M),
        "linalg.qr": lambda: fnp.linalg.qr(M),
        "linalg.slogdet": lambda: fnp.linalg.slogdet(M),
        "linalg.svd": lambda: fnp.linalg.svd(M),
        "linalg.svdvals": lambda: fnp.linalg.svdvals(M),
        "linalg.tensorinv": lambda: fnp.linalg.tensorinv(M.reshape(2, 2, 2, 2)),
        "linalg.tensorsolve": lambda: fnp.linalg.tensorsolve(
            M.reshape(2, 2, 2, 2), V4.reshape(2, 2)
        ),
        "linalg.trace": lambda: fnp.linalg.trace(M),
        "linalg.vector_norm": lambda: fnp.linalg.vector_norm(V),
    }
)

# ---------------------------------------------------------------------------
# polynomial family (flopscope._polynomial) — polyfit/roots/vander already
# seeded above; the rest of the family fixed/verified in this task.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "poly": lambda: fnp.poly(np.array([1, 2, 3], dtype=np.int32)),
        "polyadd": lambda: fnp.polyadd(V, V),
        "polyder": lambda: fnp.polyder(V),
        "polydiv": lambda: fnp.polydiv(V, np.array([1, -1], dtype=np.int32)),
        "polyint": lambda: fnp.polyint(V),
        "polymul": lambda: fnp.polymul(V, V),
        "polysub": lambda: fnp.polysub(V, V),
        "polyval": lambda: fnp.polyval(V, V),
    }
)

# ---------------------------------------------------------------------------
# symmetric family (flopscope top-level namespace, not flopscope.numpy).
# symmetrize fixed in this task; is_symmetric/as_symmetric already correct.
# ---------------------------------------------------------------------------
_SYM_GROUP = f.SymmetryGroup.symmetric(axes=(0, 1))
_SYMMETRIZED = f.symmetrize(M, symmetry=_SYM_GROUP)
PROBES.update(
    {
        "symmetrize": lambda: f.symmetrize(M, symmetry=_SYM_GROUP),
        "is_symmetric": lambda: f.is_symmetric(_SYMMETRIZED, symmetry=_SYM_GROUP),
        "as_symmetric": lambda: f.as_symmetric(_SYMMETRIZED, symmetry=_SYM_GROUP),
    }
)

# ---------------------------------------------------------------------------
# stats.<dist>.{cdf,pdf,ppf} — pdf/cdf accept any real x; ppf needs a
# probability in (0, 1), so it uses the scaled float32 probe (Q01).
# ---------------------------------------------------------------------------
_STATS_DISTS = [
    "cauchy",
    "expon",
    "laplace",
    "logistic",
    "norm",
    "truncnorm",
    "uniform",
]
for _dist in _STATS_DISTS:
    _mod = getattr(fstats, _dist)
    _kwargs = {"a": -2.0, "b": 2.0} if _dist == "truncnorm" else {}
    PROBES[f"stats.{_dist}.cdf"] = _stats_call(_mod, "cdf", V, _kwargs)
    PROBES[f"stats.{_dist}.pdf"] = _stats_call(_mod, "pdf", V, _kwargs)
    PROBES[f"stats.{_dist}.ppf"] = _stats_call(_mod, "ppf", Q01, _kwargs)
# lognorm additionally requires the shape parameter `s`.
PROBES["stats.lognorm.cdf"] = lambda: fstats.lognorm.cdf(V, s=1.0)
PROBES["stats.lognorm.pdf"] = lambda: fstats.lognorm.pdf(V, s=1.0)
PROBES["stats.lognorm.ppf"] = lambda: fstats.lognorm.ppf(Q01, s=1.0)

# ---------------------------------------------------------------------------
# window functions (flopscope._window) — fixed float64 output; the int32
# argument is a count, not data, so these still discriminate correctly.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "bartlett": lambda: fnp.bartlett(8),
        "blackman": lambda: fnp.blackman(8),
        "hamming": lambda: fnp.hamming(8),
        "hanning": lambda: fnp.hanning(8),
        "kaiser": lambda: fnp.kaiser(8, 2.0),
    }
)

# ---------------------------------------------------------------------------
# histogram / bincount / sort_complex / unwrap — fixed in this task.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "bincount": lambda: fnp.bincount(V01),
        "histogram": lambda: fnp.histogram(V),
        "histogram2d": lambda: fnp.histogram2d(V, V),
        "histogramdd": lambda: fnp.histogramdd(V.reshape(-1, 1)),
        "histogram_bin_edges": lambda: fnp.histogram_bin_edges(V),
        "sort_complex": lambda: fnp.sort_complex(V),
        "unwrap": lambda: fnp.unwrap(V),
    }
)

# ---------------------------------------------------------------------------
# cov/corrcoef/interp/trapezoid/trapz — already dtype-aware before this task
# (unconditional float64 floor for cov/corrcoef/interp; kind-conditional for
# trapezoid/trapz); locked here so a future regression is caught.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "cov": lambda: fnp.cov(M),
        "corrcoef": lambda: fnp.corrcoef(M),
        "interp": lambda: fnp.interp(XF, XF, V),
        "trapezoid": lambda: fnp.trapezoid(V),
        "trapz": lambda: fnp.trapz(V),
    }
)

# ---------------------------------------------------------------------------
# Remote-callback ops (arbitrary user Python code) — apply_along_axis/
# apply_over_axes fixed in this task to bill on the callback's OWN
# demonstrated result dtype (not just the input's); fromfunction already did.
# piecewise's result dtype is architecturally forced to match its input's
# (numpy preallocates y = zeros_like(x) and assigns into it), so its
# callback's wider dtype cannot leak through even before this task's
# (defensive, no-op-in-practice) hardening.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "apply_along_axis": lambda: fnp.apply_along_axis(
            lambda a: a.astype(np.int64).sum(), 0, M
        ),
        "apply_over_axes": lambda: fnp.apply_over_axes(
            lambda a, ax: a.astype(np.int64).sum(axis=ax, keepdims=True), M, [0]
        ),
        "piecewise": lambda: fnp.piecewise(
            V, [V > 4], [lambda a: a.astype(np.int64) * 2, 0]
        ),
        "fromfunction": lambda: fnp.fromfunction(
            lambda i, j: i + j, (4, 4), dtype=np.int32
        ),
    }
)

# ---------------------------------------------------------------------------
# from{file,string} — construction ops that bill the actual constructed
# array's own dtype (dtypes=(result.dtype,)); fromregex needs a structured
# (void-kind) dtype instead, see SKIPPED.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "fromfile": lambda: fnp.fromfile(_FROMFILE_PATH, dtype=np.int32),
        "fromstring": lambda: fnp.fromstring(
            "1 2 3 4 5 6 7 8", dtype=np.int32, sep=" "
        ),
    }
)

# ---------------------------------------------------------------------------
# save/savez/savez_compressed (cost-model triage Task 10): became dtype-aware
# charged ops (4*size, billed at the saved array's own dtype -- dtypes=
# (base.dtype,) for save, dtypes=tuple(v.dtype for saved arrays) for
# savez/savez_compressed). Like from{file,string} above, exercised via temp
# files under _PROBE_DIR; each write returns None so the compute-dtype oracle's
# result-dtype floor stays at the default 1.0 (there is no result to inspect).
# ---------------------------------------------------------------------------
_SAVE_PATH = os.path.join(_PROBE_DIR, "probe_save.npy")
_SAVEZ_PATH = os.path.join(_PROBE_DIR, "probe_savez.npz")
_SAVEZ_COMPRESSED_PATH = os.path.join(_PROBE_DIR, "probe_savez_compressed.npz")
PROBES.update(
    {
        "save": lambda: fnp.save(_SAVE_PATH, V),
        "savez": lambda: fnp.savez(_SAVEZ_PATH, a=V),
        "savez_compressed": lambda: fnp.savez_compressed(_SAVEZ_COMPRESSED_PATH, a=V),
    }
)

# isnat moved to SKIPPED below: its entire valid-input domain (datetime64/
# timedelta64) is now refused by the numeric-allowlist dtype ban, so it can
# never log a probe record to conform.

# ---------------------------------------------------------------------------
# astype/asarray (Option B billing fix: both now weight 1.0 in
# default_weights.json -- a real cast/copy bills like `copy`, numel at the
# heavier of source/dest rate via heavier_billing_dtype). Cast int32 -> a
# heavier dtype so the oracle actually discriminates: numpy's real output
# dtype is the cast destination, and the billed dtype must cover its rate.
# ---------------------------------------------------------------------------
PROBES["astype"] = lambda: fnp.astype(V, np.float64)
PROBES["asarray"] = lambda: fnp.asarray(V, dtype=np.float64)

# ---------------------------------------------------------------------------
# Everything else: simple composites/manipulation ops that already resolve
# their billing dtype correctly from their own array input(s). Each call is
# the smallest valid int32 (or, where numpy requires it, float) invocation.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "allclose": lambda: fnp.allclose(V, V),
        "arange": lambda: fnp.arange(8, dtype=np.int32),
        "argpartition": lambda: fnp.argpartition(V, 2),
        "argsort": lambda: fnp.argsort(V),
        "argwhere": lambda: fnp.argwhere(V > 3),
        "array": lambda: fnp.array(V),
        "array_equal": lambda: fnp.array_equal(V, V),
        "array_equiv": lambda: fnp.array_equiv(V, V),
        "asarray_chkfinite": lambda: fnp.asarray_chkfinite(V),
        "clip": lambda: fnp.clip(V, 2, 6),
        "convolve": lambda: fnp.convolve(V, V3),
        "copyto": lambda: _copyto_probe(),
        "correlate": lambda: fnp.correlate(V, V3),
        "cross": lambda: fnp.cross(V3, V3),
        "diff": lambda: fnp.diff(V),
        "digitize": lambda: fnp.digitize(V, [2, 4, 6]),
        "dot": lambda: fnp.dot(M, M),
        "ediff1d": lambda: fnp.ediff1d(V),
        "einsum": lambda: fnp.einsum("ij,jk->ik", M, M),
        "flatnonzero": lambda: fnp.flatnonzero(V - 4),
        "geomspace": lambda: fnp.geomspace(1, 8, 4),
        "in1d": lambda: fnp.in1d(V, V3),
        "indices": lambda: fnp.indices((2, 2)),
        "intersect1d": lambda: fnp.intersect1d(V, V3),
        # Boolean input charges ix_'s nonzero scan; its index output is intp.
        "ix_": lambda: fnp.ix_(COND),
        "isfinite": lambda: fnp.isfinite(V),
        "isin": lambda: fnp.isin(V, V3),
        "isinf": lambda: fnp.isinf(V),
        "isnan": lambda: fnp.isnan(V),
        "kron": lambda: fnp.kron(V3, V3),
        "lexsort": lambda: fnp.lexsort((V, V)),
        "linspace": lambda: fnp.linspace(0, 8, 5, dtype=np.int32),
        "logspace": lambda: fnp.logspace(0, 2, 4),
        "matmul": lambda: fnp.matmul(M, M),
        "nonzero": lambda: fnp.nonzero(V - 4),
        "outer": lambda: fnp.outer(V, V),
        "packbits": lambda: fnp.packbits(V01.astype(np.uint8)),
        "pad": lambda: fnp.pad(V, 1),
        "partition": lambda: fnp.partition(V, 2),
        "searchsorted": lambda: fnp.searchsorted(V, 4),
        "setdiff1d": lambda: fnp.setdiff1d(V, V3),
        "setxor1d": lambda: fnp.setxor1d(V, V3),
        "sort": lambda: fnp.sort(V),
        "tensordot": lambda: fnp.tensordot(M, M),
        "trace": lambda: fnp.trace(M),
        "trim_zeros": lambda: fnp.trim_zeros(V),
        "union1d": lambda: fnp.union1d(V, V3),
        "unique": lambda: fnp.unique(V),
        "unique_all": lambda: fnp.unique_all(V),
        "unique_counts": lambda: fnp.unique_counts(V),
        "unique_inverse": lambda: fnp.unique_inverse(V),
        "unique_values": lambda: fnp.unique_values(V),
        "unpackbits": lambda: fnp.unpackbits(V01.astype(np.uint8)),
        "vdot": lambda: fnp.vdot(V, V),
        "inner": lambda: fnp.inner(V3, V3),
        # 3-arg branch: bills the operands' own dtype (dtypes=(x.dtype, y.dtype)),
        # no longer dtype-neutral since the select-class rework. The 1-arg
        # branch aliases to nonzero's own deduct() call and is covered by the
        # "nonzero" probe above (identical formula, same op-name at runtime).
        "where": lambda: fnp.where(COND, V, V),
        # select: newly charged (weight 0.0 -> 1.0) by the select-class
        # rework, bills the choicelist arrays' own dtype (dtypes=_select_dtypes).
        "select": lambda: fnp.select([V > 4], [V]),
    }
)

# ---------------------------------------------------------------------------
# array assembly & replication family (weight-flipped from free -> 1.0):
# each resolves its billing dtype from its own array input(s), same as the
# "everything else" group above. `insert` is a documented exception: its
# scalar `values=9` argument widens to platform int64 via `_np.asarray(9)`
# before `result_type` combines it with V's int32, so it bills int64 (rate
# 2.0) even though the actual OUTPUT stays int32 -- an over-bill, not an
# under-bill, so it still satisfies the `billed_rate >= floor` oracle.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "concatenate": lambda: fnp.concatenate([V, V3]),
        "concat": lambda: fnp.concat([V, V3]),
        "stack": lambda: fnp.stack([V, V]),
        "vstack": lambda: fnp.vstack([V, V]),
        "hstack": lambda: fnp.hstack([V, V3]),
        "dstack": lambda: fnp.dstack([V, V]),
        "column_stack": lambda: fnp.column_stack([V, V]),
        "tile": lambda: fnp.tile(V, 2),
        "repeat": lambda: fnp.repeat(V, 2),
        "roll": lambda: fnp.roll(V, 1),
        "resize": lambda: fnp.resize(V, (2, 8)),
        "delete": lambda: fnp.delete(V, 0),
        "insert": lambda: fnp.insert(V, 0, 9),
        "append": lambda: fnp.append(V, V3),
        "fromiter": lambda: fnp.fromiter(range(8), dtype=np.int32),
        "full": lambda: fnp.full((2, 2), 3, dtype=np.int32),
        "full_like": lambda: fnp.full_like(V, 3),
        "meshgrid": lambda: fnp.meshgrid(V3, V3),
    }
)

# ---------------------------------------------------------------------------
# value-writing creation & layout copies (weight-flipped from free -> 1.0,
# cost-model triage Task 4): ones/ones_like/eye/identity declare the
# constructor's own dtype= parameter (or the input's, for *_like); copy/
# reshape/ravel/require/fft.fftshift/fft.ifftshift declare the input array's
# dtype. All resolve int32 here -- none of these ops widen.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "ones": lambda: fnp.ones(8, dtype=np.int32),
        "ones_like": lambda: fnp.ones_like(V),
        "eye": lambda: fnp.eye(4, dtype=np.int32),
        "identity": lambda: fnp.identity(4, dtype=np.int32),
        "copy": lambda: fnp.copy(V),
        "reshape": lambda: fnp.reshape(V, (2, 4)),
        "ravel": lambda: fnp.ravel(M),
        "require": lambda: fnp.require(V, requirements=["C"]),
        "fft.fftshift": lambda: fnp.fft.fftshift(V),
        "fft.ifftshift": lambda: fnp.fft.ifftshift(V),
    }
)

# ---------------------------------------------------------------------------
# diag family + triangular constructors (weight-flipped from free -> 1.0,
# cost-model triage Task 7): diag's 1-D construct branch, diagflat, triu,
# and tril all declare the input array's own dtype; none of them widen.
# ---------------------------------------------------------------------------
PROBES.update(
    {
        "diag": lambda: fnp.diag(V),
        "diagflat": lambda: fnp.diagflat(V),
        "triu": lambda: fnp.triu(M),
        "tril": lambda: fnp.tril(M),
    }
)

# ---------------------------------------------------------------------------
# tri (weight-flipped from free -> 1.0, cost-model triage Task 8): unlike its
# tril_indices/triu_indices/... index-generator siblings (all dtype-neutral,
# see the SKIPPED block below), tri constructs a real matrix and declares its
# own dtype= parameter (mirrors ones/eye/identity above) -- it does not widen.
# ---------------------------------------------------------------------------
PROBES["tri"] = lambda: fnp.tri(3, dtype=np.int32)


def _copyto_probe() -> np.ndarray:
    dst = np.zeros(8, dtype=np.int32)
    fnp.copyto(dst, V)
    return dst


# op name -> reason. Categories, each reason states why no probe is needed.
SKIPPED: dict[str, str] = {}

# --- samplers: synthesize values from a distribution with no array data
# input (array/int arguments configure the distribution, e.g. multinomial's
# `pvals`, they are not data being transformed); output dtype is fixed by
# the distribution family, not derived from any operand's arithmetic width.
# Verified individually (module import + BudgetContext probe) that every one
# of these resolves a real, non-None dtype matching that fixed family width
# (float64 for continuous families, int64 for count families) regardless of
# argument dtype -- covered by test_random_cost_formulas.py, test_dtype_rates.py,
# and test_rng_classes_integration.py, not by this composite-arithmetic sweep.
_SAMPLER_REASON = (
    "sampler synthesizes output from a distribution; output dtype is fixed "
    "by the distribution family (verified real, non-None, width-independent "
    "of any operand's own dtype) -- covered by the RNG/sampler dtype test "
    "suite (test_random_cost_formulas.py, test_dtype_rates.py, "
    "test_rng_classes_integration.py), not a numpy-widens-the-input case "
    "this sweep targets."
)
_GENERATOR_SAMPLERS = [
    "beta",
    "binomial",
    "chisquare",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "integers",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_hypergeometric",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "poisson",
    "power",
    "random",
    "rayleigh",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
]
SKIPPED.update({f"random.Generator.{m}": _SAMPLER_REASON for m in _GENERATOR_SAMPLERS})
_RANDOMSTATE_SAMPLERS = [
    "beta",
    "binomial",
    "chisquare",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random",
    "random_integers",
    "random_sample",
    "rayleigh",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "tomaxint",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
]
SKIPPED.update(
    {f"random.RandomState.{m}": _SAMPLER_REASON for m in _RANDOMSTATE_SAMPLERS}
)
_TOPLEVEL_SAMPLERS = [
    "beta",
    "binomial",
    "chisquare",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "ranf",
    "rayleigh",
    "sample",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
]
SKIPPED.update({f"random.{m}": _SAMPLER_REASON for m in _TOPLEVEL_SAMPLERS})
SKIPPED["random.symmetric"] = (
    "sampler synthesizes a symmetric random tensor; output is always "
    "float64 by construction (Reynolds-projected standard-normal draws), "
    "with exact cost-formula probes in tests/test_symmetric_cost.py, not "
    "this composite-arithmetic sweep."
)

# --- dtype-neutral movement/shuffle family: rearranges, selects, or emits
# raw bytes without arithmetic. Verified individually that each of these
# resolves resolved_dtype=None (declared dtype-neutral) for every one of
# Generator/RandomState/top-level -- the oracle itself would reject probing
# a None-dtype op ("bills dtype-neutrally"), so these must be skipped, not
# exempted via INDEX_OUTPUT_OPS (which only relaxes the RESULT-dtype floor,
# not the "billed dtype exists" requirement).
_MOVEMENT_REASON = (
    "dtype-neutral movement: rearranges/selects existing elements or emits "
    "raw bytes without arithmetic (resolved_dtype=None by design, verified "
    "individually) -- matches the movement-family skip category."
)
for _m in ("bytes", "choice", "permutation", "permuted", "shuffle"):
    SKIPPED[f"random.Generator.{_m}"] = _MOVEMENT_REASON
for _m in ("bytes", "choice", "permutation", "shuffle"):
    SKIPPED[f"random.RandomState.{_m}"] = _MOVEMENT_REASON
for _m in ("bytes", "choice", "permutation", "shuffle"):
    SKIPPED[f"random.{_m}"] = _MOVEMENT_REASON
SKIPPED["random.random_integers"] = (
    "deprecated numpy alias, intentionally unsupported: raises AttributeError "
    "on access (verified), never reaches deduct()."
)

# --- dtype-neutral index bookkeeping (cost-model triage Task 8): the
# returned index arrays are pure position bookkeeping (which cells to
# select), independent of any input's value dtype -- same convention as
# random.permutation above (dtypes=() in each op's own deduct(), verified,
# resolved_dtype=None). tril_indices_from/triu_indices_from/diag_indices_from
# take an array argument, but only its .shape is read, never its dtype/
# values. mask_indices and ravel_multi_index were PROBES entries before
# Task 8 flipped their formula to this same dtype-neutral convention; moved
# here (their old probes would now fail the "bills dtype-neutrally" check).
_INDEX_BOOKKEEPING_REASON = (
    "dtype-neutral index bookkeeping: the returned index arrays are pure "
    "position bookkeeping, independent of any input's value dtype "
    "(resolved_dtype=None by design, verified) -- same convention as "
    "random.permutation."
)
for _name in (
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
    "diag_indices",
    "diag_indices_from",
    "unravel_index",
    "mask_indices",
    "ravel_multi_index",
):
    SKIPPED[_name] = _INDEX_BOOKKEEPING_REASON
# broadcast_shapes: no array operands at all (its args are shape tuples/
# ints), so there is no operand dtype to resolve in the first place --
# dtype-neutral by construction, not merely by convention.
SKIPPED["broadcast_shapes"] = (
    "dtype-neutral: takes shape tuples/ints, not arrays -- there is no "
    "operand dtype to resolve (dtypes=() in its own deduct(), verified, "
    "resolved_dtype=None)."
)

# --- free_random_method: pure attribute/state accessors ("no math" per
# their own registry notes) that never call deduct() at all.
SKIPPED["random.Generator.bit_generator"] = (
    "attribute access, no math (registry note); never calls deduct(), so "
    "there is no billed dtype to conform."
)
SKIPPED["random.Generator.spawn"] = (
    "returns child Generators via a subclass override wrapping them as "
    "_CountedGenerator; no math of its own (registry note), never calls "
    "deduct()."
)

# --- free (0-FLOP) I/O: "Cost: 0 FLOPs" per its own registry note; verified
# it never calls deduct() (src/flopscope/_io.py), so no billed dtype exists
# to check. save/savez/savez_compressed became dtype-aware charged ops (4*size)
# in the cost-model triage (Task 10) -- moved to PROBES below (see
# "save/savez/savez_compressed" block near the from{file,string} probes).
SKIPPED["load"] = (
    "0-FLOP data movement by design (registry note); never calls "
    "deduct() (src/flopscope/_io.py) -- no billed dtype to conform. "
    "Exercised by tests/test_io.py."
)

# --- linalg.{matmul,outer,tensordot,vecdot}: pure Python aliases with NO
# deduct() of their own -- attach_docstring literally documents each as
# "0 FLOPs (delegates to flopscope.<name>)" (src/flopscope/numpy/linalg/
# _aliases.py). Calling them logs a record under the top-level op_name
# instead (verified), so a probe here would always fail with "did not log
# an op record under that name"; the top-level probe already covers the
# arithmetic. (linalg.cross and linalg.diagonal/matrix_transpose are NOT in
# this set: cross has its own independent deduct(); diagonal/matrix_transpose
# are free/weight-0, not in the charged set at all.)
for _name in ("matmul", "outer", "tensordot", "vecdot"):
    SKIPPED[f"linalg.{_name}"] = (
        f"pure alias, 0 FLOPs of its own (attach_docstring: 'delegates to "
        f"flopscope.{_name}'); billing happens under the top-level "
        f"'{_name}' op_name (verified), whose own probe above covers this "
        f"arithmetic."
    )
# `row_stack` is a bare `return vstack(tup)` alias (pure data movement) with
# NO deduct() of its own: calling it logs an op_log record under `vstack`,
# not `row_stack`, so a probe here would always fail with "did not log an op
# record under that name" -- same shape as the linalg aliases above. Its own
# weight in default_weights.json is 1.0 (matches its vstack/column_stack/
# dstack/hstack siblings, for consistency), but that key is inert: billing
# always happens under vstack's op_name and vstack's own probe covers it.
SKIPPED["row_stack"] = (
    "bare `return vstack(tup)` alias, no deduct() of its own (verified); "
    "billing happens under the top-level 'vstack' op_name, whose own probe "
    "above covers this arithmetic -- same shape as the linalg.{matmul,...} "
    "aliases above."
)

# --- dtype-neutral by design, non-random: verified individually.
# `where`'s 3-arg branch used to live here (declared dtype-neutral,
# resolved_dtype=None) but the select-class rework made it bill the
# operands' own dtype -- it now has a direct PROBES entry above instead.
SKIPPED["einsum_path"] = (
    "returns a contraction PLAN, not a value (registry note: 'no numeric "
    "output'); verified 0 FLOPs / resolved_dtype=None -- a planning "
    "utility, not an arithmetic op this sweep targets."
)
SKIPPED["isnat"] = (
    "isnat's only valid input kind is datetime64/timedelta64, both outside "
    "the numeric allowlist the dtype ban enforces -- every real call is "
    "refused before deduct() charges anything, so there is no billed dtype "
    "left for this sweep to conform."
)
# --- blacklisted category, unreachable: raises AttributeError on access
# (verified individually via flopscope.numpy.__getattr__'s "does not
# provide" guard) -- never reaches deduct().
_UNREACHABLE_REASON = (
    "blacklisted: not exposed via flopscope.numpy (raises AttributeError on "
    "access, verified) -- no deduct() call is possible."
)
for _name in (
    "array2string",
    "array_repr",
    "array_str",
    "asmatrix",
    "busday_count",
    "busday_offset",
    "datetime_as_string",
    "datetime_data",
    "format_float_positional",
    "format_float_scientific",
    "frompyfunc",
    "genfromtxt",
    "get_include",
    "getbufsize",
    "geterrcall",
    "is_busday",
    "loadtxt",
    "nested_iters",
    "savetxt",
    "setbufsize",
    "seterrcall",
    "show_config",
    "show_runtime",
):
    SKIPPED[_name] = _UNREACHABLE_REASON

# --- blacklisted category, reachable but never call deduct(): pure
# introspection/state/iterator utilities (verified individually: 0 op_log
# records after calling each).
_NO_DEDUCT_REASON = (
    "introspection/state/iterator utility (registry note: 'not a "
    "remote-compute value op' / 'global state, not remote'); verified it "
    "never calls deduct() -- no billed dtype to conform."
)
for _name in (
    "broadcast",
    "errstate",
    "finfo",
    "get_printoptions",
    "geterr",
    "iinfo",
    "ndenumerate",
    "ndindex",
    "nditer",
    "printoptions",
    "set_printoptions",
    "seterr",
):
    SKIPPED[_name] = _NO_DEDUCT_REASON

# --- blacklisted category, reachable and charged, but dtype-neutral by
# design: billed on the OUTPUT STRING's length (dtypes=(), verified), not on
# any array dtype -- there is no numeric arithmetic width to conform.
for _name in ("base_repr", "binary_repr"):
    SKIPPED[_name] = (
        "billed on the output STRING's length (registry note: 'Cost: "
        "len(output string)'); declares dtypes=() (dtype-neutral, "
        "verified) -- no array arithmetic width to conform."
    )

SKIPPED["fromregex"] = (
    "numpy.fromregex requires a STRUCTURED dtype (each regex capture group "
    "maps to a struct field, verified: a non-structured dtype raises "
    "numpy's own TypeError). Structured/void (kind 'V') is outside the "
    "numeric allowlist unconditionally, regardless of field types, so every "
    "real fromregex call is now refused before deduct() charges anything -- "
    "no billed dtype is representable through this op at all. Its sibling "
    "from{file,string} (plain numeric dtypes) are probed above."
)

# Ops whose RESULT dtype overstates the arithmetic: int64 index outputs from
# comparisons at the input's width. The input side of the oracle still
# applies. lexsort returns sort-permutation indices (int64) like argsort.
# unique_all/unique_counts/unique_inverse return tuples whose non-values
# members (indices/inverse-indices/counts) are int64; their VALUES member
# shares unique/unique_values' own dtype-resolution code path (both PASS as
# direct probes above), so exempting the tuple-as-a-whole here does not hide
# an unverified path.
INDEX_OUTPUT_OPS = {
    "argmax",
    "argmin",
    "nanargmax",
    "nanargmin",
    "argsort",
    "argpartition",
    "nonzero",
    "flatnonzero",
    "argwhere",
    "searchsorted",
    "count_nonzero",
    "digitize",
    "lexsort",
    "unique_all",
    "unique_counts",
    "unique_inverse",
}


def _charged_ops() -> set[str]:
    load_weights()
    try:
        return {name for name in REGISTRY if get_weight(name) > 0}
    finally:
        reset_weights()


def test_probe_accounting():
    charged = _charged_ops()
    covered = set(PROBES) | set(SKIPPED)
    missing = sorted(charged - covered)
    assert not missing, (
        f"{len(missing)} charged ops have neither a probe nor a documented "
        f"skip: {missing[:40]}{' ...' if len(missing) > 40 else ''}"
    )
    # Every probe/skip entry must reference a currently-CHARGED op. A stale
    # entry for an op whose weight later drops to 0 (moving it out of the
    # charged set — e.g. reclassified into the data-movement free tier) is a
    # correctness signal, not harmless dead code: it means the sweep is
    # asserting coverage for something it no longer needs to, and hides that
    # the op silently stopped billing. Checking against `charged` rather than
    # the full `REGISTRY` makes that a hard failure (a REGISTRY-only check
    # would pass, since a de-charged op stays registered).
    non_charged = sorted(covered - charged)
    assert not non_charged, (
        f"probe/skip entries for ops that are no longer charged (weight 0) — "
        f"remove them; the op is now free-tier and outside the sweep: "
        f"{non_charged}"
    )
    stale = sorted(covered - set(REGISTRY))
    assert not stale, f"probe/skip entries for unknown ops: {stale}"


def _result_dtypes(result: Any):
    if isinstance(result, (tuple, list)):
        return [r.dtype for r in result if isinstance(r, np.ndarray)]
    if isinstance(result, np.ndarray):
        return [result.dtype]
    if isinstance(result, np.generic):
        return [result.dtype]
    return []


def _assert_billed_rate_covers_compute_dtype(op: str, call: Callable[[], Any]) -> None:
    load_weights()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
            try:
                result = call()
            except UnsupportedFunctionError as e:
                # Version-gated op absent from the RUNNING numpy (e.g. matvec
                # before 2.2); the matrix cells whose numpy has it still sweep
                # it, and the degradation contract itself is pinned by
                # test_numpy_version_support.py.
                pytest.skip(f"probe op unavailable on this numpy: {e}")
            records = [r for r in b.op_log if r.op_name == op]
    assert records, f"probe for {op!r} did not log an op record under that name"
    rec = records[-1]
    if rec.resolved_dtype is None:
        pytest.fail(
            f"{op!r} bills dtype-neutrally; if that is by design move it to "
            f"SKIPPED with the width-independence reason"
        )
    billed_rate = rate_for(np.dtype(rec.resolved_dtype))
    floor = 1.0  # every probe input here is int32 (rate 1.0)
    result_rates = [rate_for(dt) for dt in _result_dtypes(result)]
    if op not in INDEX_OUTPUT_OPS and result_rates:
        floor = max(floor, max(result_rates))
    assert billed_rate >= floor, (
        f"{op!r} billed dtype {rec.resolved_dtype} (rate {billed_rate}) but "
        f"numpy produced {_result_dtypes(result)} (needs rate >= {floor})"
    )


@pytest.mark.parametrize("op", sorted(PROBES))
def test_billed_rate_covers_compute_dtype(op: str):
    _assert_billed_rate_covers_compute_dtype(op, PROBES[op])


# Ops in _UNARY_OPS that genuinely reject a bool operand (numpy itself
# raises), excluded BY NAME with a one-line reason -- never a blanket
# try/except, which would also swallow a real regression on an op that
# should accept bool but silently stopped.
_BOOL_UNSUPPORTED_UNARY: dict[str, str] = {
    "negative": "numpy: boolean negative is not supported, use logical_not",
    "positive": "numpy: ufunc 'positive' has no bool loop",
    "sign": "numpy: ufunc 'sign' has no bool loop",
}


@pytest.mark.parametrize(
    "op", [n for n in _UNARY_OPS if n not in _BOOL_UNSUPPORTED_UNARY]
)
def test_billed_rate_covers_compute_dtype_bool(op: str):
    """Same floor as the int32 sweep, driven by a bool operand.

    int32 alone cannot see the size-mapped unary float loop's bool anomaly
    (angle(bool_) computes float64, not the float16 the itemsize-based
    mapping predicts) -- both int32 and its size-mapped float loop already
    land at float64, so the divergence only shows up when the operand is
    actually bool.
    """
    _assert_billed_rate_covers_compute_dtype(op, _unary_bool(op))


# ---------------------------------------------------------------------------
# The floor that replaces the result-dtype floor for INDEX_OUTPUT_OPS.
#
# Exempting those 16 ops from the RESULT-dtype floor is correct: they return
# int64 index arrays whatever the operand's width, so the result dtype says
# nothing about the arithmetic. But the sweep above replaces the exemption with
# nothing -- `floor` stays 1.0, and no dtype rates below 1.0, so for those ops
# `billed_rate >= floor` is not a relaxed oracle, it is a switched-off one.
#
# The replacement is an OPERAND-dtype floor, probed at a dtype whose rate is
# above the minimum so the assertion can actually fail.
# ---------------------------------------------------------------------------
IDX_WIDE = np.arange(1, 9, dtype=np.float64)  # rate 2.0, unlike the int32 sweep
IDX_WIDE_CENTERED = IDX_WIDE - 4.0  # spans zero, for the nonzero family

# Narrow dtypes spanning the bottom of the rate table. Their minimum is the
# lowest rate any resolvable dtype can carry; a floor at that value is
# unfalsifiable by construction, which is what went wrong here the first time.
# Read under load_weights() -- the autouse conftest fixture resets to unit
# weights, where every rate is 1.0.
NARROW_DTYPES = ("bool", "int8", "int16", "int32", "uint8", "float16", "float32")

# op -> (operand probed, call taking that operand). The operand travels with
# the probe so the floor and the failure message both read the dtype actually
# exercised. A free-floating "the operand dtype is float64" constant can drift
# away from the probes: narrowing one probe would then be reported against the
# constant's dtype, naming an operand the run never used.
INDEX_OUTPUT_PROBES: dict[str, tuple[Any, Callable[[Any], Any]]] = {
    "argmax": (IDX_WIDE, lambda a: fnp.argmax(a)),
    "argmin": (IDX_WIDE, lambda a: fnp.argmin(a)),
    "nanargmax": (IDX_WIDE, lambda a: fnp.nanargmax(a)),
    "nanargmin": (IDX_WIDE, lambda a: fnp.nanargmin(a)),
    "argsort": (IDX_WIDE, lambda a: fnp.argsort(a)),
    "argpartition": (IDX_WIDE, lambda a: fnp.argpartition(a, 2)),
    "nonzero": (IDX_WIDE_CENTERED, lambda a: fnp.nonzero(a)),
    "flatnonzero": (IDX_WIDE_CENTERED, lambda a: fnp.flatnonzero(a)),
    "argwhere": (IDX_WIDE_CENTERED, lambda a: fnp.argwhere(a)),
    "searchsorted": (IDX_WIDE, lambda a: fnp.searchsorted(a, 4.0)),
    "count_nonzero": (IDX_WIDE, lambda a: fnp.count_nonzero(a)),
    "digitize": (IDX_WIDE, lambda a: fnp.digitize(a, [2.0, 4.0, 6.0])),
    "lexsort": (IDX_WIDE, lambda a: fnp.lexsort((a, a))),
    "unique_all": (IDX_WIDE, lambda a: fnp.unique_all(a)),
    "unique_counts": (IDX_WIDE, lambda a: fnp.unique_counts(a)),
    "unique_inverse": (IDX_WIDE, lambda a: fnp.unique_inverse(a)),
}


def _probe_operand_dtype(op: str) -> np.dtype[Any]:
    """The dtype of the array the probe for ``op`` actually feeds in."""
    return np.dtype(INDEX_OUTPUT_PROBES[op][0].dtype)


def test_index_output_probe_accounting():
    """Every result-dtype-exempt op needs an operand-dtype probe, and no more."""
    assert set(INDEX_OUTPUT_PROBES) == INDEX_OUTPUT_OPS, (
        "ops exempted from the result-dtype floor without an operand-dtype "
        f"probe: {sorted(INDEX_OUTPUT_OPS - set(INDEX_OUTPUT_PROBES))}; "
        f"stale probes: {sorted(set(INDEX_OUTPUT_PROBES) - INDEX_OUTPUT_OPS)}"
    )


def test_index_output_floor_is_falsifiable():
    """Every probe's operand must rate above the lowest rate any dtype resolves to.

    This is the guard on the guard. A floor equal to the minimum rate cannot be
    violated by any billed dtype, so the per-op assertion below would pass
    unconditionally -- exactly the defect that left the result-dtype exemption
    asserting nothing. Narrowing any float64 probe to int32 would silently
    reintroduce it for that op; this fails instead. It reads each probe's own
    operand, so it covers all sixteen rather than one shared constant.
    """
    load_weights()
    min_rate = min(rate_for(np.dtype(name)) for name in NARROW_DTYPES)
    weak = sorted(
        (op, str(_probe_operand_dtype(op)), rate_for(_probe_operand_dtype(op)))
        for op in INDEX_OUTPUT_PROBES
        if rate_for(_probe_operand_dtype(op)) <= min_rate
    )
    assert not weak, (
        "these probe operands rate at or below the minimum rate any dtype can "
        f"resolve to ({min_rate}), so the index-output floor below cannot fail "
        f"for them and is asserting nothing: {weak}"
    )


def _billed_dtype_for_probe(op: str, operand: Any) -> np.dtype[Any]:
    """Run ``op``'s probe on ``operand`` and return the dtype it billed at."""
    _, probe = INDEX_OUTPUT_PROBES[op]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
            try:
                probe(operand)
            except UnsupportedFunctionError as e:
                # Version-gated op absent from the RUNNING numpy, matching the
                # int32 sweep's handling above.
                pytest.skip(f"probe op unavailable on this numpy: {e}")
            records = [r for r in b.op_log if r.op_name == op]
    assert records, f"probe for {op!r} did not log an op record under that name"
    rec = records[-1]
    assert rec.resolved_dtype is not None, (
        f"{op!r} bills dtype-neutrally; if that is by design move it out of "
        f"INDEX_OUTPUT_OPS and into SKIPPED with the width-independence reason"
    )
    return np.dtype(rec.resolved_dtype)


@pytest.mark.parametrize("op", sorted(INDEX_OUTPUT_OPS))
def test_index_output_ops_bill_operand_dtype(op: str):
    """Index-output ops are exempt from the RESULT-dtype floor (they return
    int64 indices at any input width) but must still bill at least the
    OPERAND's rate. Probed at float64 (rate 2.0) so this can actually fail --
    an int32 probe makes the floor 1.0, which no resolvable dtype is below.
    """
    load_weights()
    operand, _ = INDEX_OUTPUT_PROBES[op]
    billed = _billed_dtype_for_probe(op, operand)
    billed_rate = rate_for(billed)
    operand_dtype = _probe_operand_dtype(op)
    floor = rate_for(operand_dtype)
    assert billed_rate >= floor, (
        f"{op!r} billed dtype {billed} (rate {billed_rate}) on a "
        f"{operand_dtype} operand (needs rate >= {floor}): the index output "
        f"is int64 either way, but the comparison work is done at the operand's "
        f"width and must be billed there"
    )


# The floor alone cannot separate "bills the operand" from "bills the int64
# index result": float64 and int64 both rate 2.0 on the shipped table, so an op
# that priced its indices instead of its comparisons would satisfy it. Narrowing
# the operand splits them -- operand billing follows down to rate 1.0, index
# billing stays at 2.0.
IDX_NARROW_DTYPE = np.dtype("float32")

# searchsorted's needle and digitize's bin edges are float64 in their probes, so
# NEP-50 promotes the call even when the array narrows. They bill above the
# operand, which over-bills rather than under-bills -- the safe direction -- so
# they are exempt from tracking down and pinned to over-resolving instead, which
# keeps the exemption from rotting into a blanket skip.
NARROWING_EXEMPT = frozenset({"searchsorted", "digitize"})


@pytest.mark.parametrize("op", sorted(INDEX_OUTPUT_OPS))
def test_index_output_ops_track_operand_width_down(op: str):
    """Billing must follow the operand down, not sit on the index result's width."""
    load_weights()
    operand, _ = INDEX_OUTPUT_PROBES[op]
    wide_rate = rate_for(_probe_operand_dtype(op))
    narrow_rate = rate_for(IDX_NARROW_DTYPE)
    assert narrow_rate < wide_rate, (
        f"{IDX_NARROW_DTYPE} rates {narrow_rate} and the probe operand rates "
        f"{wide_rate}, so narrowing separates nothing and this test asserts "
        "nothing -- pick a narrower probe dtype"
    )

    billed = _billed_dtype_for_probe(op, operand.astype(IDX_NARROW_DTYPE))
    billed_rate = rate_for(billed)
    if op in NARROWING_EXEMPT:
        assert billed_rate > narrow_rate, (
            f"{op!r} now tracks its operand down to {billed} -- it no longer "
            "over-resolves, so drop it from NARROWING_EXEMPT and let it be "
            "held to the same rule as the rest"
        )
        return
    assert billed_rate <= narrow_rate, (
        f"{op!r} billed dtype {billed} (rate {billed_rate}) on a narrowed "
        f"{IDX_NARROW_DTYPE} operand (needs rate <= {narrow_rate}): billing "
        "that does not follow the operand down is pricing the int64 index "
        "output rather than the comparison work"
    )


def test_fixed_composites_do_not_overbill_f32():
    """Composites fixed in this task that numpy computes f32-in-f32-out must
    still bill float32, not fold unconditionally to float64.

    Excludes composites that numpy itself widens even from float32 input
    (polyfit, polyint, symmetrize, cov/corrcoef/interp, histogram/histogram2d/
    histogramdd's counts, sort_complex, stats.*, vander) -- those correctly
    bill float64 (or wider) for an f32 probe too, verified separately above.
    i0/sinc only widen INTEGER/bool input unconditionally; float32 stays
    float32 (integer_to_float64_min_dtype leaves float/complex kinds unchanged),
    so they ARE included below.
    """
    load_weights()
    g = np.linspace(1.0, 2.0, 8, dtype=np.float32)
    p32 = np.array([1.0, -3.0, 2.0], dtype=np.float32)
    cases = [
        lambda: fnp.gradient(g),
        lambda: fnp.median(g),
        lambda: fnp.average(g),
        lambda: fnp.nanmedian(g),
        lambda: fnp.percentile(g, 50),
        lambda: fnp.nanpercentile(g, 50),
        lambda: fnp.trapezoid(g),
        # numpy 2.4 removed trapz; its wrapper then raises
        # UnsupportedFunctionError (pinned by test_numpy_version_support).
        *([lambda: fnp.trapz(g)] if hasattr(np, "trapz") else []),
        lambda: fnp.unwrap(g),
        lambda: fnp.histogram_bin_edges(g),
        lambda: fnp.angle(g),
        lambda: fnp.i0(g),
        lambda: fnp.sinc(g),
        lambda: fnp.roots(p32),
        lambda: fnp.polydiv(g, np.array([1.0, -1.0], dtype=np.float32)),
        lambda: fnp.poly(p32),
        lambda: fnp.polyval(g, g),
        lambda: fnp.polyadd(g, g),
        lambda: fnp.polymul(g, g),
        lambda: fnp.frexp(g),
        lambda: fnp.modf(g),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for call in cases:
            with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
                call()
                assert b.op_log[-1].resolved_dtype == "float32", b.op_log[-1]


def test_sort_complex_does_not_overbill_narrow_ints():
    """sort_complex's fix must not blanket-fold to complex128.

    numpy's own hardcoded dtype.char table (see _sort_complex_billing_dtype)
    routes int8/int16/uint8/uint16 to the narrower complex64 loop, not
    complex128 -- billing complex128 unconditionally would overcount these
    by 2x. Complex input is returned unchanged, not widened either.
    """
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.sort_complex(np.array([1, 2, 3], dtype=np.int8))
        assert b.op_log[-1].resolved_dtype == "complex64", b.op_log[-1]
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.sort_complex(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
        assert b.op_log[-1].resolved_dtype == "complex64", b.op_log[-1]
