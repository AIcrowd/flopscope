"""Exhaustive, mechanical batch/broadcast/dropped-dimension under-bill scan.

Enumerates EVERY charged (non-free) flopscope op from ``website/public/ops.json``
and mechanically probes each for the batch / broadcast / dropped-dimension
under-bill family, so coverage is *provable* rather than judgement-based.

The invariant under test
------------------------
Prepending a batch axis of size ``k`` to an op's array argument(s) multiplies
numpy's real work by EXACTLY ``k`` whenever the op loops/broadcasts over that
axis -- regardless of per-item cost. So the honest billed ratio must be ~k.
We detect the loop mechanically: if ``op(batched)`` runs AND its output
propagates the batch (output numel scales ~k vs the base output), numpy did k
work => billed must scale ~k. Under-bill = billed_ratio materially below k
(flagged when ``billed_ratio < 0.6*k``). The dtype-rate constant (e.g. float64
x2) cancels in a same-dtype ratio, so every probe holds dtype fixed.

Three mechanical probes per op (probe A batch-prepend, probe B broadcast-output,
probe C param-repeat) plus a per-op recipe table for ops the generic generator
cannot build a valid batched call for (two-operand broadcast batch like solve,
fft N-D, random samplers, cov/corrcoef rowvar, tensorsolve degenerate, permuted
list-repr, pad stat modes, mvhg count).

Provenance
----------
Adapted from the throwaway discovery harness ``.superpowers/sdd/
exhaustive_batch_scan.py`` (the scan that found the 18-op under-bill family
fixed across Tasks 1-8) into this committed, importable module so it can run
as a standing CI regression guard (``tests/test_batch_scaling_guard.py``).
The per-op recipe table and the ``evaluate()`` verdict engine are carried over
unchanged, with two narrow, evidence-backed exceptions documented inline at
``_quantile_recipe`` and ``recipe_mvhg``: both recipes compared a scalar/tiny
base against a K-times-larger one, which places the comparison in a regime
where a *correct* shared-computation cost formula (a single partition/buffer
build shared across many cheap derived outputs) cannot possibly reach the
generic verdict engine's "honest == k" bar, no matter the value of k -- the
fixed, already-audited formula only approaches that bound asymptotically, as
q.size / sum(colors) grows large relative to the other terms in the formula.
Both recipes were rescaled to probe deep enough into that asymptotic regime
to actually test what the fix targets (see ``task-9-report.md`` for the full
derivation, numpy-source citations, and empirical verification).

Public API
----------
``scan_all()`` -- run the full scan, return ``{op_name: result_dict}``.
``charged_ops()`` -- the list of charged op names the scan covers.

Determinism
-----------
``scan_all()`` resets the shared module RNG stream (and the unused-in-practice
input-numel side table) to the fixed ``SEED`` on every call, and scopes its
warning suppression to the duration of the scan, so repeated calls -- in this
process or a fresh one -- are bit-identical and importing this module never
mutates process-wide warning filters.

Manual use
----------
``python -m tests.batch_scan`` runs the scan and prints a verdict histogram
(no file output; use ``scan_all()`` programmatically for anything else).
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
import flopscope.stats as fstats  # noqa: F401  (ensures stats namespace imports)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OPS_JSON = os.path.join(REPO, "website", "public", "ops.json")

K = 16  # batch multiplier for probe A / C
BUDGET = 10**16
SEED = 20260720
FLAG = 0.6  # under-bill if billed_ratio < FLAG * honest_ratio

_rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Billing primitive
# --------------------------------------------------------------------------- #
_SENTINEL = object()


def bill(thunk: Callable[[], Any]) -> tuple[int, Any, str | None]:
    """Run ``thunk`` inside a fresh BudgetContext. Return (flops_used, out, err)."""
    ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    ctx.__enter__()
    out: Any = _SENTINEL
    err: str | None = None
    try:
        out = thunk()
    except Exception as e:  # noqa: BLE001 - we want every failure mode
        err = f"{type(e).__name__}: {e}"
    finally:
        used = ctx.flops_used
        ctx.__exit__(None, None, None)
    return used, out, err


def numel(x: Any) -> int | None:
    """Total element count of an array / tuple-of-arrays / scalar; None if N/A."""
    if x is _SENTINEL or x is None:
        return None
    try:
        return int(np.asarray(x).size)
    except Exception:
        pass
    # tuple/list of arrays (eig -> (w, v), svd -> (u, s, vh), histogram, ...)
    try:
        tot = 0
        for e in x:
            tot += int(np.asarray(e).size)
        return tot
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Input builders (all called OUTSIDE any BudgetContext)
# --------------------------------------------------------------------------- #
def arr(shape, profile="unit", dtype=None):
    """Numpy array wrapped as a flopscope array (construction not billed).

    ``dtype=None`` keeps each profile's natural dtype -- critically, the "int"
    and "uint8" profiles stay integer (bitwise / packbits ops reject floats).
    """
    if profile == "unit":
        base = _rng.random(shape) * 0.8 + 0.1  # (0.1, 0.9)
    elif profile == "pos":
        base = _rng.random(shape) * 3.0 + 1.0  # (1, 4)
    elif profile == "signed":
        base = _rng.standard_normal(shape)
    elif profile == "complex":
        base = _rng.random(shape) + 1j * _rng.random(shape)
    elif profile == "int":
        base = _rng.integers(1, 9, size=shape)  # int64, nonzero
    elif profile == "uint8":
        base = _rng.integers(0, 2, size=shape).astype(np.uint8)
    else:
        raise ValueError(profile)
    if dtype is not None:
        base = np.asarray(base, dtype=dtype)
    return fnp.asarray(base)


def invertible(shape):
    """Diagonally-dominant (well-conditioned, non-singular) square matrices."""
    n = shape[-1]
    m = _rng.standard_normal(shape)
    return fnp.asarray(m + n * np.eye(n))


def spd(shape):
    """Symmetric positive-definite square matrices (cholesky / eigh safe)."""
    n = shape[-1]
    m = _rng.standard_normal(shape)
    s = np.matmul(m, np.swapaxes(m, -1, -2)) + n * np.eye(n)
    return fnp.asarray(s)


# --------------------------------------------------------------------------- #
# Probe dataclass (plain dict) + verdict engine
# --------------------------------------------------------------------------- #
def probe(label, base, batched, honest_mode, note="", k=K):
    """honest_mode: 'output' | 'input' | float (explicit honest ratio)."""
    return {
        "label": label,
        "base": base,
        "batched": batched,
        "honest_mode": honest_mode,
        "note": note,
        "k": k,
    }


def evaluate(p: dict[str, Any]) -> dict[str, Any] | None:
    """Run a probe. Return a result dict, or None if the base call is invalid
    (plan doesn't apply to this op)."""
    b_base, out_base, e_base = bill(p["base"])
    if e_base is not None:
        return None  # base invalid -> plan doesn't apply
    nb = numel(out_base)
    b_bat, out_bat, e_bat = bill(p["batched"])
    if e_bat is not None:
        # base ran, batched errored -> informative "not batchable this way"
        return {
            "status": "batched_error",
            "label": p["label"],
            "note": e_bat,
            "b_base": b_base,
            "b_bat": None,
            "billed_ratio": None,
            "honest": None,
            "nb": nb,
            "nt": None,
        }
    nt = numel(out_bat)
    k = p["k"]
    mode = p["honest_mode"]

    if mode in ("output", "input"):
        ref = nb if mode == "output" else _input_numel(p, "base")
        cur = nt if mode == "output" else _input_numel(p, "bat")
        if ref in (None, 0) or cur is None:
            return None
        prop = cur / ref
        if prop < FLAG * k:
            # output/input did NOT propagate the batch -> not a genuine loop here
            return {
                "status": "no_batch",
                "label": p["label"],
                "k": k,
                "note": f"{mode} did not propagate (prop={prop:.3g}, k={k})",
                "b_base": b_base,
                "b_bat": b_bat,
                "billed_ratio": (b_bat / b_base if b_base else None),
                "honest": prop,
                "nb": nb,
                "nt": nt,
            }
        # Batch confirmed to propagate.  The invariant is that numpy did EXACTLY
        # k more per-item work, so honest == k -- NOT the raw output ratio, which
        # can exceed k for structural reasons (argwhere/nonzero gain an index
        # column when ndim grows; diagflat flattens to a quadratic output).
        honest = k
    else:
        honest = float(mode)

    if not b_base:
        # Base bills 0 -> a free view / data-movement op (e.g. diagonal, a
        # read-only view).  There is no billable compute to scale by k, so the
        # batch invariant does not apply; this is not an under-bill.
        return {
            "status": "no_batch",
            "label": p["label"],
            "k": k,
            "note": "base bills 0 (free view / data-movement); no compute to scale",
            "b_base": b_base,
            "b_bat": b_bat,
            "billed_ratio": None,
            "honest": honest,
            "nb": nb,
            "nt": nt,
        }

    billed_ratio = b_bat / b_base
    verdict = "UNDER-BILL" if billed_ratio < FLAG * honest else "OK-SCALES"
    return {
        "status": verdict,
        "label": p["label"],
        "note": p["note"],
        "k": k,
        "b_base": b_base,
        "b_bat": b_bat,
        "billed_ratio": billed_ratio,
        "honest": honest,
        "nb": nb,
        "nt": nt,
    }


# input-numel bookkeeping for probes that use honest_mode='input'. Not
# exercised by any recipe in this file today (no recipe sets
# honest_mode="input") -- kept for parity with the source harness. Cleared at
# the start of every scan_all() call for determinism-hygiene even though
# nothing currently populates it.
_INPUT_NUMELS: dict[tuple[int, str], int | None] = {}


def _input_numel(p, which):
    return _INPUT_NUMELS.get((id(p), which))


def probe_input(
    label, base, batched, note="", k=K, base_in_numel=None, bat_in_numel=None
):
    p = probe(label, base, batched, "input", note=note, k=k)
    _INPUT_NUMELS[(id(p), "base")] = base_in_numel
    _INPUT_NUMELS[(id(p), "bat")] = bat_in_numel
    return p


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #
def resolve(name):
    """Return ('func', callable) | ('gen', method) | ('rs', method) | ('unresolved', msg)."""
    if ".Generator." in name:
        return ("gen", name.split(".Generator.")[1])
    if ".RandomState." in name:
        return ("rs", name.split(".RandomState.")[1])
    for base in (fnp, flops):
        obj = base
        ok = True
        for part in name.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                ok = False
                break
        if ok:
            return ("func", obj)
    return ("unresolved", f"cannot resolve {name!r} under fnp/flops")


def gen_call(method):
    """Bind a Generator method to a FRESH counted generator on every call."""
    return lambda *a, **k: getattr(fnp.random.default_rng(SEED), method)(*a, **k)


def rs_call(method):
    return lambda *a, **k: getattr(fnp.random.RandomState(SEED), method)(*a, **k)


# --------------------------------------------------------------------------- #
# Generic plan battery (probe A / B) -- each returns a probe or None
# --------------------------------------------------------------------------- #
def plans_generic(op, category):
    """Universal probe battery, tried for every func-kind op without a recipe.

    We try ALL signature shapes (not just the ones the ops.json *category*
    label suggests): the category label is unreliable (e.g. ``isclose`` is
    tagged ``counted_unary`` but is binary), and an invalid base call is
    cheaply filtered (returns None), so a broad battery only helps coverage.
    """

    def A_unary_1d():
        a1 = arr((64,))
        ak = arr((K, 64))
        return probe("A:unary-1d", lambda: op(a1), lambda: op(ak), "output")

    def A_unary_2d():
        a1 = arr((6, 6))
        ak = arr((K, 6, 6))
        return probe("A:unary-2d", lambda: op(a1), lambda: op(ak), "output")

    def A_unary_1d_int():
        a1 = arr((64,), "int")
        ak = arr((K, 64), "int")
        return probe("A:unary-1d-int", lambda: op(a1), lambda: op(ak), "output")

    def A_binary_1d():
        a1 = arr((64,))
        b1 = arr((64,))
        ak = arr((K, 64))
        bk = arr((K, 64))
        return probe("A:binary-1d", lambda: op(a1, b1), lambda: op(ak, bk), "output")

    def A_binary_1d_int():
        a1 = arr((64,), "int")
        b1 = arr((64,), "int")
        ak = arr((K, 64), "int")
        bk = arr((K, 64), "int")
        return probe(
            "A:binary-1d-int", lambda: op(a1, b1), lambda: op(ak, bk), "output"
        )

    def A_binary_2d():
        a1 = arr((6, 6))
        b1 = arr((6, 6))
        ak = arr((K, 6, 6))
        bk = arr((K, 6, 6))
        return probe("A:binary-2d", lambda: op(a1, b1), lambda: op(ak, bk), "output")

    def A_reduction_1d():
        a1 = arr((64,))
        ak = arr((K, 64))
        return probe(
            "A:reduce-1d", lambda: op(a1, axis=-1), lambda: op(ak, axis=-1), "output"
        )

    def A_reduction_2d():
        a1 = arr((6, 6))
        ak = arr((K, 6, 6))
        return probe(
            "A:reduce-2d", lambda: op(a1, axis=-1), lambda: op(ak, axis=-1), "output"
        )

    def A_matrix():
        a1 = invertible((6, 6))
        ak = invertible((K, 6, 6))
        return probe("A:matrix", lambda: op(a1), lambda: op(ak), "output")

    def A_spd():
        a1 = spd((6, 6))
        ak = spd((K, 6, 6))
        return probe("A:spd", lambda: op(a1), lambda: op(ak), "output")

    def A_matrix_pair():
        a1 = invertible((6, 6))
        b1 = invertible((6, 6))
        ak = invertible((K, 6, 6))
        bk = invertible((K, 6, 6))
        return probe("A:matmul", lambda: op(a1, b1), lambda: op(ak, bk), "output")

    def B_bcast():
        a0 = arr((1, 8))
        b0 = arr((1, 8))
        a1 = arr((1, 8))
        b1 = arr((8, 8))
        return probe("B:bcast", lambda: op(a0, b0), lambda: op(a1, b1), "output", k=8)

    universal = [
        A_unary_1d,
        A_unary_2d,
        A_binary_1d,
        A_reduction_1d,
        A_unary_1d_int,
        A_binary_1d_int,
        A_binary_2d,
        A_matrix,
        A_spd,
        A_matrix_pair,
        A_reduction_2d,
        B_bcast,
    ]
    head = {
        "counted_unary": [A_unary_1d, A_unary_2d],
        "counted_binary": [A_binary_1d, B_bcast],
        "counted_reduction": [A_reduction_1d, A_reduction_2d],
    }.get(category, [])
    order = head + [f for f in universal if f not in head]

    plans = []
    for f in order:
        try:
            plans.append(f())
        except Exception:
            pass
    return plans


# --------------------------------------------------------------------------- #
# Random sampler prober (probe C -- scale the size / output dimension)
# --------------------------------------------------------------------------- #
SAMPLER_CANDIDATES = [
    (),
    (2.0,),
    (3,),
    (0.5,),
    (5,),
    (2.0, 3.0),
    (10, 0.5),
    (1, 10),
    (2.0, 3.0, 4.0),
    (10, 5, 8),
    (0.3,),
    (1.5,),
]


def plan_sampler(bound_op, leaf):
    """Return probes for a random sampler by scaling its output size (probe C)."""
    # array-parameter samplers handled explicitly
    if leaf == "dirichlet":
        al = [2.0, 2.0, 2.0]
        return [
            probe(
                "C:sampler",
                lambda: bound_op(al, size=(64,)),
                lambda: bound_op(al, size=(K, 64)),
                "output",
            )
        ]
    if leaf == "multinomial":
        pv = [0.2, 0.3, 0.5]
        return [
            probe(
                "C:sampler",
                lambda: bound_op(10, pv, size=(64,)),
                lambda: bound_op(10, pv, size=(K, 64)),
                "output",
            )
        ]
    if leaf == "multivariate_normal":
        mean = np.zeros(3)
        cov = np.eye(3)
        return [
            probe(
                "C:sampler",
                lambda: bound_op(mean, cov, size=(64,)),
                lambda: bound_op(mean, cov, size=(K, 64)),
                "output",
            )
        ]
    if leaf in ("rand", "randn"):
        return [
            probe("C:dims", lambda: bound_op(64), lambda: bound_op(K, 64), "output")
        ]
    if leaf == "permutation":
        return [
            probe("C:perm", lambda: bound_op(64), lambda: bound_op(64 * K), "output")
        ]
    if leaf == "bytes":
        return recipe_bytes(bound_op)
    if leaf == "shuffle":
        # shuffle permutes ALONG axis 0 only; registry note documents
        # "cost = shape[axis] (Fisher-Yates draws)". Prepending a batch axis
        # REDUCES the draw count (shape[0] falls), and element movement is a
        # free data-logistics op -- so this is not a per-item compute batch.
        return [
            {
                "_direct": True,
                "status": "no_batch",
                "label": "C:shuffle-NA",
                "note": "row-permutation: bills shape[0] draws (documented); "
                "movement free -> not a compute batch",
                "b_base": None,
                "b_bat": None,
                "billed_ratio": None,
                "honest": None,
                "nb": None,
                "nt": None,
            }
        ]
    if leaf == "choice":
        pool = arr((256,))
        return [
            probe(
                "C:choice",
                lambda: bound_op(pool, size=(64,)),
                lambda: bound_op(pool, size=(K, 64)),
                "output",
            )
        ]

    # generic scalar-parameter samplers: trial a valid param tuple OUTSIDE ctx
    for params in SAMPLER_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = bound_op(*params, size=(2,))
        except Exception:
            continue
        return [
            probe(
                "C:sampler",
                (lambda pr=params: bound_op(*pr, size=(64,))),
                (lambda pr=params: bound_op(*pr, size=(K, 64))),
                "output",
            )
        ]
    # positional-size fallback
    for params in SAMPLER_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = bound_op(*params, (2,))
        except Exception:
            continue
        return [
            probe(
                "C:sampler-pos",
                (lambda pr=params: bound_op(*pr, (64,))),
                (lambda pr=params: bound_op(*pr, (K, 64))),
                "output",
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Per-op recipe table (the ops the generic generator cannot build correctly).
# Each entry: name -> function(op_callable_or_None) -> list[probe]
# For random Generator/RandomState methods the recipe key is the LEAF name and
# receives the bound callable.
# --------------------------------------------------------------------------- #
def _pad_true_stat_cost(in_shape, pad_pairs, mode):
    """Honest FLOP count for np.pad stat modes (ported from pad_probe.py,
    fuzz-validated 160/160 against a monkeypatched numpy trace)."""
    ndim = len(in_shape)
    cur = list(in_shape)
    cost = 0
    for i in range(ndim):
        before, after = pad_pairs[i]
        axis_len = in_shape[i]
        if axis_len == 0:
            cur[i] = axis_len + before + after
            continue
        true_cross = 1
        for j in range(ndim):
            if j == i:
                continue
            true_cross *= cur[j]
        cost += true_cross * axis_len
        if mode == "mean":
            cost += true_cross
        cur[i] = axis_len + before + after
    return math.prod(cur) + cost


def recipe_solve(op):
    a = invertible((6, 6))  # a NOT batched
    b1 = arr((6, 4))
    bk = arr((K, 6, 4))  # batch arrives via b only
    return [
        probe(
            "R:solve-one-sided",
            lambda: op(a, b1),
            lambda: op(a, bk),
            "output",
            note="one-sided batch: a fixed, b gains leading axis",
        )
    ]


def recipe_tensorsolve(op):
    from flopscope.numpy.linalg._solvers import solve_cost

    a1 = fnp.asarray(np.eye(8))
    b1 = fnp.asarray(np.ones(8))
    a2 = fnp.asarray(np.eye(64))
    b2 = fnp.asarray(np.ones(64))
    honest = solve_cost(64, 1) / solve_cost(8, 1)  # cubic in N; billed is flat at 4
    return [
        probe(
            "R:tensorsolve-degenerate",
            lambda: op(a1, b1),
            lambda: op(a2, b2),
            honest,
            note="degenerate 2-D a / 1-D b: bill flat, honest ~N^3",
        )
    ]


# The base quantile-array length used by _quantile_recipe below. numpy's real
# quantile/percentile algorithm (numpy/lib/_function_base_impl.py::_quantile)
# shares ONE arr.partition() pass (cost ~ axis_dim, bounded by the UNION of
# every requested quantile's pivot indices) across every requested quantile,
# then does an O(1) gather+interpolate PER quantile. So growing q.size from a
# small base multiplies total work by far less than the size ratio while
# axis_dim still dominates -- the fixed cost model here
# (axis_dim + 4*q_count per output orbit; see src/flopscope/_pointwise.py)
# already reflects exactly that shape. Comparing q_count=1 vs q_count=K (as
# the original discovery recipe did, appropriate for finding the ORIGINAL bug
# where billed was blind to q.size at ANY K) pins the comparison where the
# shared-partition constant dominates and no honest formula can reach the
# generic verdict engine's honest=K bar -- verified empirically: at Q0=1,
# billed_ratio=1.88 of a required 9.6 (K=16), a false positive, not a
# residual under-bill (see task-9-report.md for the full derivation and
# numpy-source walkthrough). Probing from Q0_BASE=1000 quantiles instead of a
# scalar moves the comparison deep enough into the asymptotic regime
# (q.size >> axis_dim=64) that the probe measures what the fix actually
# targets: empirically ratio ~15.76 of 16, safely clear of the 9.6 threshold.
Q0_BASE = 1000


def _quantile_recipe(op, lo, hi):
    a2 = arr((6, 64))
    q_base = np.linspace(lo, hi, Q0_BASE)
    q_batched = np.linspace(lo, hi, Q0_BASE * K)
    return [
        probe(
            "C:q-count",
            lambda: op(a2, q_base, axis=1),
            lambda: op(a2, q_batched, axis=1),
            "output",
            note=f"q-array prepends output axis; probed at Q0={Q0_BASE} "
            "quantiles (asymptotic regime) so the shared-partition "
            "constant can't mask q.size scaling",
        )
    ]


def recipe_percentile(op):
    return _quantile_recipe(op, 1.0, 99.0)


def recipe_quantile(op):
    return _quantile_recipe(op, 0.01, 0.99)


def recipe_nanpercentile(op):
    return _quantile_recipe(op, 1.0, 99.0)


def recipe_nanquantile(op):
    return _quantile_recipe(op, 0.01, 0.99)


def _fftn_recipe(op, profile):
    a1 = arr((6, 6), profile=profile)
    ak = arr((K, 6, 6), profile=profile)
    return [
        probe(
            "R:fftn-nd",
            lambda: op(a1, s=(6, 6)),
            lambda: op(ak, s=(6, 6)),
            "output",
            note="s given + axes omitted: leading axes are a dropped batch",
        )
    ]


def recipe_fftn(op):
    return _fftn_recipe(op, "complex")


def recipe_ifftn(op):
    return _fftn_recipe(op, "complex")


def recipe_rfftn(op):
    return _fftn_recipe(op, "unit")  # real input


def recipe_irfftn(op):
    return _fftn_recipe(op, "complex")


def _cov_recipe(op):
    x = np.asarray(_rng.random((2, 64)))  # R=2, C=64
    fx = fnp.asarray(x)
    fxt = fnp.asarray(np.ascontiguousarray(x.T))  # (64, 2)
    b_true, _, _ = bill(lambda: op(fnp.asarray(x), rowvar=True))
    b_work, _, _ = bill(lambda: op(fxt, rowvar=True))  # honest C-variable scaling
    honest = (b_work / b_true) if b_true else 1.0
    return [
        probe(
            "R:rowvar",
            lambda: op(fnp.asarray(x), rowvar=True),
            lambda: op(fnp.asarray(x), rowvar=False),
            honest,
            note="rowvar=False swaps variable axis; billed blind to rowvar",
        )
    ]


def recipe_cov(op):
    return _cov_recipe(op)


def recipe_corrcoef(op):
    return _cov_recipe(op)


def recipe_pad(op):
    # Unlike every other recipe/probe in this file, this one compares an
    # ACTUAL bill (b_used) against an EXTERNALLY precomputed reference
    # (honest_flops from a pure-Python port of numpy's pad-stat-mode cost),
    # not a ratio of two bills taken in the same process. So the dtype-rate
    # constant does NOT cancel here the way the module docstring says it
    # does everywhere else -- it must track whatever billing rate is
    # actually ACTIVE when b_used is measured. flopscope's own test suite
    # runs under "unit weights" (rate 1.0 for every dtype; see
    # tests/conftest.py's autouse reset_weights()), which is NOT the
    # packaged production rate (float64 = 2.0) a standalone script sees on
    # fresh import. A hardcoded "* 2" here bills honest_flops at the
    # production rate while b_used is measured at whatever rate the calling
    # process actually has active, so the ratio would be wrong (and would
    # silently flip verdict) depending on which environment the scan runs
    # in -- exactly the kind of environment-dependent nondeterminism the
    # determinism proof in task-9-report.md checks for.
    from flopscope._weights import get_dtype_rate

    shape = (8, 8, 40)
    pairs = [(0, 0), (0, 0), (20, 0)]
    a = arr(shape)
    honest_flops = _pad_true_stat_cost(shape, pairs, "mean") * get_dtype_rate("float64")
    b_used, _, err = bill(lambda: op(a, pairs, mode="mean"))
    billed_ratio = (b_used / honest_flops) if honest_flops else None
    # emulate a probe-result row directly (direct-verdict recipe)
    return [
        {
            "_direct": True,
            "label": "R:pad-stat-modes",
            "note": "unpadded axes still reduced by numpy; billed skips them",
            "b_base": honest_flops,
            "b_bat": b_used,
            "billed_ratio": (b_used / honest_flops if honest_flops else None),
            "honest": 1.0,
            "status": ("UNDER-BILL" if (b_used < FLAG * honest_flops) else "OK-SCALES"),
            "nb": honest_flops,
            "nt": b_used,
            "err": err,
        }
    ]


def recipe_mvhg(bound_op):
    # Scaled 1000x from the original discovery-scan magnitudes (colors
    # summing to 9 / 144). numpy's real "count" algorithm (per the already-
    # fixed cost formula in src/flopscope/numpy/random/_cost_formulas.py:
    # total + num_variates*min(nsample, total-nsample) + numel(output)) has
    # two terms -- the draw term and the output term -- that are bounded by
    # nsample/size and do NOT grow with sum(colors). At small magnitude those
    # terms are comparable to `total` itself, so growing `total` 16x doesn't
    # move the bill anywhere near 16x even though the formula is honest
    # (empirically ratio=3.2 of a required 9.6 at this scale -- a false
    # positive, not a residual under-bill; see task-9-report.md). At 1000x
    # magnitude `total` dominates (nsample/size stay fixed at 5/(8,)), so the
    # probe measures the intended sum(colors) scaling the fix targets
    # (empirically ratio ~15.89 of 16) -- consistent with how the fix's own
    # regression test (test_mvhg_count_scales_with_sum_colors in
    # tests/test_batch_underbill_pins.py) verifies this exact property using
    # an even larger, unambiguous magnitude gap.
    colors_small = [4000, 3000, 2000]  # sum 9000
    colors_big = [139000, 3000, 2000]  # sum 144000 -> honest ~16x
    honest = sum(colors_big) / sum(colors_small)
    return [
        probe(
            "R:mvhg-count",
            lambda: bound_op(colors_small, 5, size=(8,), method="count"),
            lambda: bound_op(colors_big, 5, size=(8,), method="count"),
            honest,
            note="method='count' scales with sum(colors); probed at "
            "1000x magnitude so nsample/output terms don't mask it",
        )
    ]


def recipe_polyval(op):
    # Genuine batch: FIXED 1-D polynomial, evaluate at a batched grid of points.
    p = fnp.asarray(np.array([1.0, -2.0, 0.5, 3.0]))
    x1 = arr((64,))
    xk = arr((K, 64))
    return [
        probe(
            "R:polyval-batch-x",
            lambda: op(p, x1),
            lambda: op(p, xk),
            "output",
            note="fixed poly, batched eval points x",
        )
    ]


def recipe_permuted(bound_op):
    M = 64
    single = [float(i) for i in range(M)]  # 1-D list, len==numel
    batched = [[float(1000 * r + c) for c in range(M)] for r in range(K)]  # (K, M) list
    return [
        probe(
            "R:permuted-list",
            lambda: bound_op(single, axis=0),
            lambda: bound_op(batched, axis=1),
            "output",
            note="nested Python list bills len(outer), not numel",
        )
    ]


def _bp(label, op, base_args, bat_args, note="", base_kw=None, bat_kw=None, k=K):
    """One batch-prepend probe with explicit base/batched positional args."""
    bk = base_kw or {}
    tk = bat_kw or {}
    return probe(
        label,
        (lambda: op(*base_args, **bk)),
        (lambda: op(*bat_args, **tk)),
        "output",
        note=note,
        k=k,
    )


# ---- index / movement / gather recipes (fixed non-array args, batch the array) ----
def recipe_take(op):
    idx = fnp.asarray(np.array([0, 2, 4]))
    return [
        _bp(
            "A:take-axis",
            op,
            (arr((6, 8)), idx),
            (arr((K, 6, 8)), idx),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
            note="take along last axis",
        )
    ]


def recipe_take_along_axis(op):
    i1 = fnp.asarray(_rng.integers(0, 8, (6, 3)))
    ik = fnp.asarray(_rng.integers(0, 8, (K, 6, 3)))
    return [
        _bp(
            "A:take_along_axis",
            op,
            (arr((6, 8)), i1),
            (arr((K, 6, 8)), ik),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_repeat(op):
    return [
        _bp(
            "A:repeat-axis",
            op,
            (arr((64,)), 3),
            (arr((K, 64)), 3),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_tile(op):
    return [_bp("A:tile", op, (arr((1, 64)), (1, 3)), (arr((K, 64)), (1, 3)))]


def recipe_partition(op):
    return [
        _bp(
            "A:partition",
            op,
            (arr((64,)), 3),
            (arr((K, 64)), 3),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_argpartition(op):
    return recipe_partition(op)


def recipe_reshape(op):
    return [_bp("A:reshape", op, (arr((64,)), (8, 8)), (arr((K, 64)), (K, 8, 8)))]


def recipe_resize(op):
    return [_bp("A:resize", op, (arr((64,)), (8, 16)), (arr((K, 64)), (K, 8, 16)))]


def recipe_delete(op):
    return [
        _bp(
            "A:delete-axis",
            op,
            (arr((6, 8)), [1, 2]),
            (arr((K, 6, 8)), [1, 2]),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_insert(op):
    return [
        _bp(
            "A:insert-axis",
            op,
            (arr((6, 8)), 1, 0.0),
            (arr((K, 6, 8)), 1, 0.0),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_choose(op):
    i1 = fnp.asarray(_rng.integers(0, 3, (64,)))
    ik = fnp.asarray(_rng.integers(0, 3, (K, 64)))
    ch1 = [arr((64,)), arr((64,)), arr((64,))]
    chk = [arr((K, 64)), arr((K, 64)), arr((K, 64))]
    return [_bp("A:choose", op, (i1, ch1), (ik, chk))]


def recipe_digitize(op):
    bins = fnp.asarray(np.linspace(0.0, 1.0, 10))
    return [_bp("A:digitize", op, (arr((64,)), bins), (arr((K, 64)), bins))]


def recipe_searchsorted(op):
    s = fnp.asarray(np.sort(_rng.random(64)))
    return [
        _bp(
            "A:searchsorted-v",
            op,
            (s, arr((8,))),
            (s, arr((K, 8))),
            note="fixed sorted array, batched query values v",
        )
    ]


def recipe_interp(op):
    xp = fnp.asarray(np.sort(_rng.random(20)))
    fp = fnp.asarray(_rng.random(20))
    return [
        _bp(
            "A:interp-x",
            op,
            (arr((64,)), xp, fp),
            (arr((K, 64)), xp, fp),
            note="fixed xp/fp, batched sample points x",
        )
    ]


def recipe_cross(op):
    return [
        _bp("A:cross", op, (arr((8, 3)), arr((8, 3))), (arr((K, 8, 3)), arr((K, 8, 3))))
    ]


def recipe_einsum(op):
    return [
        _bp(
            "A:einsum-batch",
            op,
            ("ij,jk->ik", arr((6, 6)), arr((6, 6))),
            ("bij,bjk->bik", arr((K, 6, 6)), arr((K, 6, 6))),
            note="explicit batch subscript b prepended",
        )
    ]


def recipe_matrix_power(op):
    return [
        _bp("A:matrix_power", op, (invertible((6, 6)), 3), (invertible((K, 6, 6)), 3))
    ]


def recipe_isin(op):
    test = fnp.asarray(_rng.integers(0, 20, 10))
    return [
        _bp(
            "A:isin",
            op,
            (arr((64,), "int"),),
            (arr((K, 64), "int"),),
            base_kw={"test_elements": test},
            bat_kw={"test_elements": test},
        )
    ]


def recipe_full_like(op):
    return [_bp("A:full_like", op, (arr((64,)), 3.0), (arr((K, 64)), 3.0))]


def recipe_unravel_index(op):
    i1 = fnp.asarray(_rng.integers(0, 64, 64))
    ik = fnp.asarray(_rng.integers(0, 64, (K, 64)))
    return [_bp("A:unravel_index", op, (i1, (8, 8)), (ik, (8, 8)))]


def recipe_polyfit(op):
    x = fnp.asarray(np.linspace(0, 1, 64))
    y1 = arr((64,))
    yk = fnp.asarray(_rng.random((64, K)))  # K columns = K simultaneous fits
    return [
        _bp(
            "A:polyfit-cols",
            op,
            (x, y1, 3),
            (x, yk, 3),
            note="batched y columns = simultaneous fits",
        )
    ]


# ---- stats recipes (extra shape params; batch the x argument) ----
def _stats_recipe(op, extra):
    x1 = arr((64,), "pos")
    xk = arr((K, 64), "pos")
    return [_bp("A:stats-x", op, (x1, *extra), (xk, *extra))]


def recipe_lognorm(op):
    return _stats_recipe(op, (1.0,))  # s


def recipe_truncnorm(op):
    return _stats_recipe(op, (-2.0, 2.0))  # a, b


# ---- window recipes (scale the window length M -- probe C) ----
def recipe_window(op):
    return [
        probe(
            "C:window-M",
            lambda: op(64),
            lambda: op(64 * K),
            "output",
            note="scale window length M",
        )
    ]


def recipe_kaiser(op):
    return [
        probe(
            "C:window-M",
            lambda: op(64, 14.0),
            lambda: op(64 * K, 14.0),
            "output",
            note="scale window length M (beta fixed)",
        )
    ]


def recipe_packbits(op):
    return [
        _bp(
            "A:packbits",
            op,
            (arr((1, 64), "uint8"),),
            (arr((K, 64), "uint8"),),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_unpackbits(op):
    return [
        _bp(
            "A:unpackbits",
            op,
            (arr((1, 8), "uint8"),),
            (arr((K, 8), "uint8"),),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_ldexp(op):
    x1 = arr((64,))
    i1 = arr((64,), "int")
    xk = arr((K, 64))
    ik = arr((K, 64), "int")
    return [_bp("A:ldexp", op, (x1, i1), (xk, ik), note="float mantissa, int exponent")]


def recipe_select(op):
    a1 = arr((64,))
    ak = arr((K, 64))
    c1 = fnp.asarray(np.asarray(a1) > 0.5)
    ck = fnp.asarray(np.asarray(ak) > 0.5)
    return [
        _bp(
            "A:select",
            op,
            ([c1], [a1]),
            ([ck], [ak]),
            note="condlist/choicelist batched together",
        )
    ]


# ---- ops whose natural batch form needs explicit trailing axes ----
def recipe_diagonal(op):
    return [
        _bp(
            "A:diagonal",
            op,
            (arr((6, 6)),),
            (arr((K, 6, 6)),),
            base_kw={"axis1": -2, "axis2": -1},
            bat_kw={"axis1": -2, "axis2": -1},
            note="per-item diagonal over the last two axes",
        )
    ]


def recipe_trace(op):
    return [
        _bp(
            "A:trace",
            op,
            (arr((6, 6)),),
            (arr((K, 6, 6)),),
            base_kw={"axis1": -2, "axis2": -1},
            bat_kw={"axis1": -2, "axis2": -1},
            note="per-item trace over the last two axes",
        )
    ]


def recipe_linalg_trace(op):
    # np.linalg.trace always traces the last two axes (no axis1/axis2 kwargs).
    return [
        _bp(
            "A:linalg-trace",
            op,
            (arr((6, 6)),),
            (arr((K, 6, 6)),),
            note="per-item trace over the last two axes (numpy 2.x API)",
        )
    ]


def recipe_roll(op):
    return [
        _bp(
            "A:roll",
            op,
            (arr((64,)), 3),
            (arr((K, 64)), 3),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_compress(op):
    cond = [True, False, True, False, True, False, True, False]  # len 8
    return [
        _bp(
            "A:compress",
            op,
            (cond, arr((6, 8))),
            (cond, arr((K, 6, 8))),
            base_kw={"axis": -1},
            bat_kw={"axis": -1},
        )
    ]


def recipe_matvec(op):
    return [
        _bp("A:matvec", op, (arr((6, 4)), arr((4,))), (arr((K, 6, 4)), arr((K, 4))))
    ]


def recipe_vecmat(op):
    return [
        _bp("A:vecmat", op, (arr((4,)), arr((4, 5))), (arr((K, 4)), arr((K, 4, 5))))
    ]


def recipe_vecdot(op):
    return [_bp("A:vecdot", op, (arr((4,)), arr((4,))), (arr((K, 4)), arr((K, 4))))]


def recipe_bytes(bound_op):
    # bytes(n) output is a bytestring (numel oracle can't see it) -> explicit
    # honest = k: doubling n must double the bill.
    return [
        probe(
            "C:bytes-explicit",
            lambda: bound_op(64),
            lambda: bound_op(64 * K),
            float(K),
            note="scale byte count n; honest = k",
        )
    ]


# name (dotted, as in ops.json) -> recipe fn taking the resolved op callable
RECIPES_FUNC = {
    "linalg.solve": recipe_solve,
    "linalg.tensorsolve": recipe_tensorsolve,
    "percentile": recipe_percentile,
    "quantile": recipe_quantile,
    "nanpercentile": recipe_nanpercentile,
    "nanquantile": recipe_nanquantile,
    "fft.fftn": recipe_fftn,
    "fft.ifftn": recipe_ifftn,
    "fft.rfftn": recipe_rfftn,
    "fft.irfftn": recipe_irfftn,
    "cov": recipe_cov,
    "corrcoef": recipe_corrcoef,
    "pad": recipe_pad,
    "polyval": recipe_polyval,
    "polyfit": recipe_polyfit,
    # index / movement / gather
    "take": recipe_take,
    "take_along_axis": recipe_take_along_axis,
    "repeat": recipe_repeat,
    "tile": recipe_tile,
    "partition": recipe_partition,
    "argpartition": recipe_argpartition,
    "reshape": recipe_reshape,
    "resize": recipe_resize,
    "delete": recipe_delete,
    "insert": recipe_insert,
    "choose": recipe_choose,
    "digitize": recipe_digitize,
    "searchsorted": recipe_searchsorted,
    "interp": recipe_interp,
    "cross": recipe_cross,
    "linalg.cross": recipe_cross,
    "einsum": recipe_einsum,
    "linalg.matrix_power": recipe_matrix_power,
    "isin": recipe_isin,
    "full_like": recipe_full_like,
    "unravel_index": recipe_unravel_index,
    # stats with shape params
    "stats.lognorm.pdf": recipe_lognorm,
    "stats.lognorm.cdf": recipe_lognorm,
    "stats.lognorm.ppf": recipe_lognorm,
    "stats.truncnorm.pdf": recipe_truncnorm,
    "stats.truncnorm.cdf": recipe_truncnorm,
    "stats.truncnorm.ppf": recipe_truncnorm,
    # windows (scale length)
    "bartlett": recipe_window,
    "blackman": recipe_window,
    "hamming": recipe_window,
    "hanning": recipe_window,
    "kaiser": recipe_kaiser,
    # bit-packing / mixed-dtype / multi-condition
    "packbits": recipe_packbits,
    "unpackbits": recipe_unpackbits,
    "ldexp": recipe_ldexp,
    "select": recipe_select,
    # explicit-trailing-axis batch forms
    "diagonal": recipe_diagonal,
    "trace": recipe_trace,
    "linalg.trace": recipe_linalg_trace,
    "roll": recipe_roll,
    "compress": recipe_compress,
    "matvec": recipe_matvec,
    "vecmat": recipe_vecmat,
    "vecdot": recipe_vecdot,
    "linalg.vecdot": recipe_vecdot,
}

# leaf method name -> recipe fn taking the BOUND callable (Generator/RandomState)

# Ops that genuinely have no per-item batch/broadcast axis (or must not run).
# Classified NO-BATCH-NA with a concrete reason instead of left UNTESTED.
NA_REASONS = {
    # file / state / property -- no array compute
    "save": "writes an .npy file to disk (I/O); not run",
    "savez": "writes an .npz file to disk (I/O); not run",
    "savez_compressed": "writes a compressed .npz to disk (I/O); not run",
    "getitem": "resolves to __getitem__; plain indexing, no batch cost model",
    "random.Generator.bit_generator": "attribute/property accessor, not a sampler",
    "random.Generator.spawn": "spawns child generators; no array work",
    "random.RandomState.seed": "RNG state mutation; no array work",
    "random.RandomState.set_state": "RNG state mutation; no array work",
    "random.RandomState.get_state": "RNG state accessor; returns state tuple, no array work",
    # requires a Python callable / iterable (arbitrary user code)
    "apply_along_axis": "requires a Python callable; cost is caller-code-defined",
    "apply_over_axes": "requires a Python callable; cost is caller-code-defined",
    "piecewise": "requires a list of Python callables; caller-code-defined",
    "fromfunction": "constructs from a Python callable; no array operand",
    "fromiter": "constructs from a Python iterable; no array operand",
    # creation ops with scalar shape args -- no array operand to batch
    "arange": "creation from scalar start/stop/step (no array operand)",
    "eye": "creation from scalar shape (no array operand)",
    "identity": "creation from scalar n (no array operand)",
    "ones": "creation from scalar shape (no array operand)",
    "full": "creation from scalar shape + fill (no array operand)",
    "tri": "creation from scalar shape (no array operand)",
    "indices": "grid factory from a shape tuple (no array operand)",
    "diag_indices": "index-grid factory from scalar n (no array operand)",
    "tril_indices": "index-grid factory from scalar n (no array operand)",
    "triu_indices": "index-grid factory from scalar n (no array operand)",
    "mask_indices": "index-grid factory from scalar n (no array operand)",
    "broadcast_shapes": "operates on shape tuples, not arrays",
    "ravel_multi_index": "index arithmetic on a tuple of index arrays; 1-D",
    "fft.fftfreq": "frequency-grid factory from scalar n (no array operand)",
    "fft.rfftfreq": "frequency-grid factory from scalar n (no array operand)",
    # 1-D-only reductions / set ops -- flatten inputs, no batch axis
    "bincount": "1-D count; output size = max+1, no batch dimension",
    "in1d": "set membership flattens inputs to 1-D (no batch axis)",
    "intersect1d": "set op flattens inputs to 1-D (no batch axis)",
    "setdiff1d": "set op flattens inputs to 1-D (no batch axis)",
    "setxor1d": "set op flattens inputs to 1-D (no batch axis)",
    "union1d": "set op flattens inputs to 1-D (no batch axis)",
    # matrix chains / whole-tensor ops -- no batch/broadcast dimension
    "linalg.multi_dot": "2-D matrix chain; no batch/broadcast dimension",
    "linalg.tensorinv": "single tensor inverse; folds all leading axes",
    "einsum_path": "returns a contraction path (planning), not array data",
    # in-place scatter (data movement) -- returns None, movement is free-tier
    "put": "in-place scatter (movement); returns None",
    "put_along_axis": "in-place scatter (movement); returns None",
    "place": "in-place scatter (movement); returns None",
    "putmask": "in-place scatter (movement); returns None",
    "copyto": "in-place copy (movement); returns None",
    "fill_diagonal": "in-place diagonal write (movement); returns None",
    # symmetric-tensor API -- requires a SymmetryGroup(axes=...) argument
    "symmetrize": "symmetric-tensor API; requires SymmetryGroup(axes=...); cost already numel-scaled",
    "as_symmetric": "symmetric-tensor API; requires SymmetryGroup(axes=...); cost already numel-scaled",
    "is_symmetric": "symmetric-tensor API; requires SymmetryGroup(axes=...); cost already numel-scaled",
    "random.symmetric": "not counted via Generator (raises UnsupportedFunctionError); symmetric-tensor sampler",
}

# leaf method name -> recipe fn taking the BOUND callable (Generator/RandomState)
RECIPES_RANDOM_LEAF = {
    "multivariate_hypergeometric": recipe_mvhg,
    "permuted": recipe_permuted,
    "bytes": recipe_bytes,
}


# --------------------------------------------------------------------------- #
# Main per-op driver
# --------------------------------------------------------------------------- #
def scan_op(op: dict[str, Any]) -> dict[str, Any]:
    name = op["name"]
    category = op["category"]
    module = op["module"]
    kind, target = resolve(name)

    row: dict[str, Any] = {
        "op": name,
        "module": module,
        "probe": "",
        "k": K,
        "bill_base": None,
        "bill_batched": None,
        "billed_ratio": None,
        "honest_ratio": None,
        "verdict": "",
        "note": "",
    }

    if name in NA_REASONS:
        row.update(probe="NA", verdict="NO-BATCH-NA", note=NA_REASONS[name])
        return row

    if kind == "unresolved":
        row.update(verdict="UNTESTED", note=target)
        return row

    # Build the list of probes to try (recipes first, then generic battery).
    probes = []
    leaf = target if kind in ("gen", "rs") else None

    if kind in ("gen", "rs"):
        bound = gen_call(leaf) if kind == "gen" else rs_call(leaf)
        if leaf in RECIPES_RANDOM_LEAF:
            probes = RECIPES_RANDOM_LEAF[leaf](bound)
        else:
            probes = plan_sampler(bound, leaf)
    else:
        op_callable = target
        if name in RECIPES_FUNC:
            probes = RECIPES_FUNC[name](op_callable)
        elif category == "counted_random_method":
            probes = plan_sampler(op_callable, name.split(".")[-1])
        elif module == "numpy.random":
            probes = plan_sampler(op_callable, name.split(".")[-1])
        else:
            probes = plans_generic(op_callable, category)
            # stats / window / poly benefit from the unary battery already in generic

    # Evaluate probes; stop at first decisive (UNDER-BILL or OK-SCALES).
    results = []
    for p in probes:
        if isinstance(p, dict) and p.get("_direct"):
            results.append(p)
            if p["status"] in ("UNDER-BILL", "OK-SCALES"):
                break
            continue
        r = evaluate(p)
        if r is None:
            continue
        results.append(r)
        if r["status"] in ("UNDER-BILL", "OK-SCALES"):
            break

    return _finalize(row, results)


def _finalize(row: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        row.update(
            verdict="UNTESTED", note="no applicable probe (base call never valid)"
        )
        return row

    # priority: UNDER-BILL > OK-SCALES > no_batch/batched_error
    under = [r for r in results if r["status"] == "UNDER-BILL"]
    oks = [r for r in results if r["status"] == "OK-SCALES"]
    if under:
        r = under[0]
        verdict = "UNDER-BILL"
    elif oks:
        r = oks[0]
        verdict = "OK-SCALES"
    else:
        r = results[-1]
        verdict = "NO-BATCH-NA"

    if r.get("k") is not None:
        row["k"] = r["k"]
    row.update(
        probe=r.get("label", ""),
        bill_base=r.get("b_base"),
        bill_batched=r.get("b_bat"),
        billed_ratio=r.get("billed_ratio"),
        honest_ratio=r.get("honest"),
        verdict=verdict,
        note=(r.get("note") or "")[:200],
    )
    return row


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _as_float_or_none(x: Any) -> float | None:
    return None if x is None else float(x)


def charged_ops() -> list[str]:
    """The charged (``not free``) op names from ``website/public/ops.json``,
    in file order -- the same list a full scan iterates."""
    with open(OPS_JSON) as f:
        data = json.load(f)
    return [o["name"] for o in data["operations"] if not o.get("free", False)]


def scan_all() -> dict[str, dict[str, Any]]:
    """Run the mechanical batch/broadcast/dropped-dimension scan over every
    charged op. Returns ``{op_name: result}`` where ``result`` is at least
    ``{"verdict": str, "note": str, "billed_ratio": float | None,
    "honest": float | None}`` (plus ``probe``/``k``/``module``/``bill_base``/
    ``bill_batched`` for debugging). Verdicts: ``"UNDER-BILL"``,
    ``"OK-SCALES"``, ``"NO-BATCH-NA"``, ``"UNTESTED"``.

    Resets the shared module RNG stream (and the input-numel side table) to
    the fixed ``SEED`` before scanning, and scopes warning suppression to the
    scan itself, so repeated calls -- in this process or a fresh one --
    produce bit-identical verdicts and importing this module never mutates
    process-wide warning filters.
    """
    global _rng
    _rng = np.random.default_rng(SEED)
    _INPUT_NUMELS.clear()

    with open(OPS_JSON) as f:
        data = json.load(f)
    ops = [o for o in data["operations"] if not o.get("free", False)]

    results: dict[str, dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op in ops:
            try:
                row = scan_op(op)
            except Exception:  # noqa: BLE001 - a harness failure must not crash the whole scan
                row = {
                    "op": op["name"],
                    "module": op["module"],
                    "probe": "",
                    "k": K,
                    "bill_base": None,
                    "bill_batched": None,
                    "billed_ratio": None,
                    "honest_ratio": None,
                    "verdict": "UNTESTED",
                    "note": "harness-exception: "
                    + traceback.format_exc().splitlines()[-1][:150],
                }
            results[row["op"]] = {
                "verdict": row["verdict"],
                "note": row["note"],
                "billed_ratio": _as_float_or_none(row.get("billed_ratio")),
                "honest": _as_float_or_none(row.get("honest_ratio")),
                "probe": row.get("probe", ""),
                "k": row.get("k"),
                "module": row.get("module"),
                "bill_base": row.get("bill_base"),
                "bill_batched": row.get("bill_batched"),
            }
    return results


if __name__ == "__main__":
    from collections import Counter

    results = scan_all()
    counts = Counter(r["verdict"] for r in results.values())
    print(f"charged ops: {len(results)}", file=sys.stderr)
    print("\n=== VERDICT COUNTS ===", file=sys.stderr)
    for v, n in counts.most_common():
        print(f"  {v:14} {n}", file=sys.stderr)
    under = sorted(op for op, r in results.items() if r["verdict"] == "UNDER-BILL")
    if under:
        print(f"\nUNDER-BILL ({len(under)}): {under}", file=sys.stderr)
