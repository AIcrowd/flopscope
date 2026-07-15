"""Canonical worked-example inputs for charged registry ops.

``resolve(op, entry)`` gives every charged op a runnable, deterministic input
via three layers, first hit wins:

1. ``CANONICAL_INPUTS`` -- explicit per-op seeds for shaped/constrained ops.
2. ``harvest_from_tests`` -- reuse test-suite fixtures (filled by the later
   curation pass; returns None until then).
3. ``category_default`` -- a generic input keyed on the registry category.

Inputs are deterministic (np.ones / seeded default_rng(0)) and are built at
``make()`` time, OUTSIDE the returned zero-arg callable, so the callable runs
only the op under measurement: ``capture_cost_site`` records the first deduct,
and building arrays inside the callable would attribute every op's cost site
to ``asarray``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import flopscope.numpy as fnp


@dataclass
class InputSpec:
    describe: str
    make: Callable[..., Any]  # make(dtype, scale=1) -> zero-arg callable
    scalable: bool
    # Dtype for the RAW/cost-site measurement passes. float32 for most ops;
    # integer-only ops (bitwise/shift/gcd) must measure at an integer dtype.
    # The four billed columns always honor their own dtypes regardless.
    raw_dtype: str = "float32"


def _arr(n: int, dt: Any) -> Any:
    return fnp.asarray(np.ones(n, dtype=dt))


# --- Layer 3: category defaults ---------------------------------------------

# Fresh seeded rng per make() call keeps repeated measurements deterministic.
_RNG_FACTORIES: dict[str, Callable[[], Any]] = {
    "Generator": lambda: fnp.random.default_rng(0),
    "RandomState": lambda: fnp.random.RandomState(0),
}


def _random_default(op: str, entry: dict) -> InputSpec | None:
    # random.<Class>.<dist> -> call <dist> on a seeded counted rng of <Class>.
    parts = op.split(".")
    if len(parts) != 3 or parts[0] != "random":
        return None
    cls_name, dist = parts[1], parts[2]
    factory = _RNG_FACTORIES.get(cls_name)
    if factory is None:
        return None
    try:
        # The counted rng raises UnsupportedFunctionError (not AttributeError)
        # for unwrapped methods, so probe with try/except rather than hasattr.
        getattr(factory(), dist)
    except Exception:
        return None

    def make(dt: Any, s: int = 1) -> Callable[[], Any]:
        sampler = getattr(factory(), dist)  # rng built here, not in the callable
        return lambda: sampler(size=1000 * s)

    return InputSpec("size=1000", make, True)


def category_default(op: str, entry: dict) -> InputSpec | None:
    cat = entry.get("category")
    if cat == "counted_random_method":
        return _random_default(op, entry)
    fn = getattr(fnp, op, None)
    if fn is None:
        return None
    if cat == "counted_unary":

        def make_unary(dt: Any, s: int = 1) -> Callable[[], Any]:
            a = _arr(1000 * s, dt)
            return lambda: fn(a)

        return InputSpec("(1000,)", make_unary, True)
    if cat == "counted_binary":

        def make_binary(dt: Any, s: int = 1) -> Callable[[], Any]:
            a = _arr(1000 * s, dt)
            b = _arr(1000 * s, dt)
            return lambda: fn(a, b)

        return InputSpec("(1000,)+(1000,)", make_binary, True)
    if cat == "counted_reduction":

        def make_reduction(dt: Any, s: int = 1) -> Callable[[], Any]:
            a = _arr(1000 * s, dt)
            return lambda: fn(a)

        return InputSpec("reduce (1000,)", make_reduction, True)
    return None


# --- Layer 2: harvest from test fixtures ------------------------------------


def harvest_from_tests(op: str) -> InputSpec | None:
    # Reuse OP_EXPECTATIONS-style fixtures. Implemented in the curation pass
    # where a fixture exists; returns None until then.
    return None


# --- Layer 1: explicit overrides for shaped/constrained ops -----------------
# Seed the obvious families now; the long tail is filled by the curation pass.


def _psd(n: int, dt: Any) -> Any:
    # Gram matrix of all-ones rows (== n*ones) plus n*I: positive definite and
    # exact in every dtype. Built directly rather than via a @ a.T -- BLAS
    # matmul leaks spurious FP-status warnings on some platforms.
    return fnp.asarray(n * np.ones((n, n), dtype=dt) + n * np.eye(n, dtype=dt))


def _make_matmul(dt: Any, s: int = 1) -> Callable[[], Any]:
    a = fnp.asarray(np.ones((256 * s, 256), dtype=dt))
    b = fnp.asarray(np.ones((256, 256 * s), dtype=dt))
    return lambda: fnp.matmul(a, b)


def _make_svd(dt: Any, s: int = 1) -> Callable[[], Any]:
    a = fnp.asarray(np.ones((64 * s, 64), dtype=dt))
    return lambda: fnp.linalg.svd(a)


def _make_cholesky(dt: Any, s: int = 1) -> Callable[[], Any]:
    a = _psd(64 * s, dt)
    return lambda: fnp.linalg.cholesky(a)


def _make_einsum(dt: Any, s: int = 1) -> Callable[[], Any]:
    a = fnp.asarray(np.ones((64 * s, 64), dtype=dt))
    b = fnp.asarray(np.ones((64, 64), dtype=dt))
    return lambda: fnp.einsum("ij,jk->ik", a, b)


CANONICAL_INPUTS: dict[str, InputSpec] = {
    "matmul": InputSpec("(256,256)@(256,256)", _make_matmul, True),
    "linalg.svd": InputSpec("(64,64)", _make_svd, True),
    "linalg.cholesky": InputSpec("(64,64) PSD", _make_cholesky, True),
    "einsum": InputSpec("'ij,jk->ik' (64,64)", _make_einsum, True),
    # ... long tail filled by the curation pass ...
}


# --- Resolver ----------------------------------------------------------------


def resolve(op: str, entry: dict) -> InputSpec | None:
    if op in CANONICAL_INPUTS:
        return CANONICAL_INPUTS[op]
    h = harvest_from_tests(op)
    if h is not None:
        return h
    return category_default(op, entry)
