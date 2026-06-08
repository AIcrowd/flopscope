"""Registry-driven client↔server round-trip conformance.

Every op the client registry advertises as proxyable (``iter_proxyable()``) is
*optimistically forwarded* to the server — the client never checks whether the
result can come back. The server packs results with a type switch
(``_request_handler._pack_result``); a return type it doesn't handle used to
fall through to a silent ``str()`` and then fail opaquely inside
``msgpack.packb`` ("failed to serialize response"). That is exactly how
``random.default_rng`` (which returns a ``numpy.random.Generator``) broke for
participants until flopscope 0.4.2 (PR #109).

This module turns "does the client↔server model actually support this call?"
into a directly-enforced, registry-driven property:

* ``test_op_round_trips`` — drives a representative call for every *exercised*
  op through the real ``RequestHandler`` and fails on the serialization-error
  class (``UnsupportedReturnType`` / ``FlopscopeServerError``). Argument-domain
  errors (``ValueError`` / ``TypeError``) are skipped — they mean the example
  args were wrong, not that the boundary is broken.
* ``test_every_proxyable_op_is_classified`` — the coverage guard: every
  proxyable op must be either *exercised* (category default or
  ``EXAMPLE_OVERRIDES``) or explicitly listed in ``PENDING_EXAMPLES``. A newly
  generated op is neither, so it turns this test red until someone gives it an
  example — it can't ship unexercised.

The in-process harness (Session + RequestHandler) mirrors
``tests/test_serialization_parity.py``. The 94 random-*method* ops
(``counted_random_method`` / ``free_random_method``) are ``RemoteGenerator``
methods, not module-level proxies, so they're outside ``iter_proxyable()`` and
this test's scope.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SERVER_SRC = str(_ROOT / "flopscope-server" / "src")
_CLIENT_SRC = str(_ROOT / "flopscope-client" / "src")

if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)

from flopscope_server._request_handler import (  # pyright: ignore[reportMissingImports]
    RequestHandler,  # noqa: E402
)
from flopscope_server._session import (  # pyright: ignore[reportMissingImports]
    Session,  # noqa: E402
)


def _load_client_module(rel_path: str, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, Path(_CLIENT_SRC) / rel_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The proxyable surface lives in the client's generated ``_registry_data.py``
# (a pure ``FUNCTION_CATEGORIES`` dict). Load it under a unique alias so it
# never collides with the core ``flopscope._registry`` the server imports, then
# replicate the trivial proxy-category logic from
# ``flopscope-client/src/flopscope/_registry.py``.
_registry_data = _load_client_module(
    "flopscope/_registry_data.py", "_conformance_client_registry_data"
)
FUNCTION_CATEGORIES: dict[str, str] = _registry_data.FUNCTION_CATEGORIES

COUNTED_UNARY = "counted_unary"
COUNTED_BINARY = "counted_binary"
COUNTED_REDUCTION = "counted_reduction"
COUNTED_CUSTOM = "counted_custom"
FREE = "free"
_PROXY_CATEGORIES = frozenset(
    {COUNTED_UNARY, COUNTED_BINARY, COUNTED_REDUCTION, COUNTED_CUSTOM, FREE}
)


def iter_proxyable() -> list[str]:
    return sorted(n for n, c in FUNCTION_CATEGORIES.items() if c in _PROXY_CATEGORIES)


def get_category(name: str) -> str | None:
    return FUNCTION_CATEGORIES.get(name)


PROXYABLE: list[str] = iter_proxyable()
SERIALIZATION_ERRORS = {"UnsupportedReturnType", "FlopscopeServerError"}

# --------------------------------------------------------------------------
# Example inputs
# --------------------------------------------------------------------------
# Shared, well-conditioned constants so a generic call works across many ops.
_A = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])  # 3x3 SPD
_B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])  # invertible
_v = np.array([1.0, 2.0, 3.0])
_w = np.array([0.5, 1.5, 2.5])
_p = np.array([0.2, 0.5, 0.8])  # in (0, 1) — valid for ppf
_six = np.arange(6.0)
_pos = np.array([[1.2, 1.5, 1.8], [2.1, 2.4, 2.7], [3.0, 3.3, 3.6]])  # >1, positive
_pos2 = _pos + 0.5
_iA = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # int 3x3
_iB = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])  # int 3x3
_ishift = np.array([[1, 1, 2], [2, 1, 1], [1, 2, 1]])  # small int shift amounts
_iv = np.array([1, 2, 3])  # int vector

# Per-op (args, kwargs) for the irregular `free` / `counted_custom` ops that a
# category default can't synthesize. ndarray args are auto-encoded as handles.
EXAMPLE_OVERRIDES: dict[str, tuple[list, dict]] = {
    # RNG (the historically-broken surface)
    "random.default_rng": ([0], {}),
    "random.seed": ([0], {}),
    "random.random": ([3], {}),
    "random.standard_normal": ([3], {}),
    "random.randn": ([3], {}),
    "random.normal": ([], {"size": 3}),
    "random.uniform": ([], {"size": 3}),
    "random.randint": ([0, 10], {"size": 3}),
    "random.choice": ([_v], {}),
    "random.permutation": ([5], {}),
    # creation
    "zeros": ([[3, 3]], {}),
    "ones": ([[3, 3]], {}),
    "eye": ([3], {}),
    "empty": ([[3, 3]], {}),
    "identity": ([3], {}),
    "tri": ([3], {}),
    "arange": ([6], {}),
    "linspace": ([0.0, 1.0, 5], {}),
    "logspace": ([0.0, 1.0, 5], {}),
    "geomspace": ([1.0, 8.0, 4], {}),
    "full": ([[2, 2], 1.0], {}),
    # shape / manipulation
    "reshape": ([_six, [2, 3]], {}),
    "transpose": ([_B], {}),
    "squeeze": ([_v], {}),
    "ravel": ([_B], {}),
    "moveaxis": ([_B, 0, 1], {}),
    "swapaxes": ([_B, 0, 1], {}),
    "flip": ([_B], {}),
    "expand_dims": ([_v, 0], {}),
    "atleast_2d": ([_v], {}),
    "asarray": ([_v], {}),
    "copy": ([_B], {}),
    "concatenate": ([[_v, _w]], {}),
    "stack": ([[_v, _w]], {}),
    "vstack": ([[_v, _w]], {}),
    "hstack": ([[_v, _w]], {}),
    "sort": ([_B], {}),
    "argsort": ([_B], {}),
    "clip": ([_B, 0.0, 1.0], {}),
    "unique": ([_B], {}),
    "diff": ([_v], {}),
    "where": ([_B > 5.0, _A, _B], {}),
    # contraction
    "dot": ([_A, _B], {}),
    "matmul": ([_A, _B], {}),
    "inner": ([_v, _w], {}),
    "outer": ([_v, _w], {}),
    "vdot": ([_v, _w], {}),
    "kron": ([_A, _B], {}),
    "tensordot": ([_A, _B], {}),
    "einsum": (["ij,jk->ik", _A, _B], {}),
    "cross": ([_v, _w], {}),
    # linalg
    "linalg.inv": ([_B], {}),
    "linalg.svd": ([_B], {}),
    "linalg.qr": ([_B], {}),
    "linalg.det": ([_B], {}),
    "linalg.solve": ([_B, _v], {}),
    "linalg.eig": ([_B], {}),
    "linalg.eigh": ([_A], {}),
    "linalg.norm": ([_B], {}),
    "linalg.pinv": ([_B], {}),
    "linalg.matrix_rank": ([_B], {}),
    "linalg.trace": ([_B], {}),
    "linalg.cholesky": ([_A], {}),
    # fft
    "fft.fft": ([_v], {}),
    "fft.ifft": ([_v], {}),
    "fft.rfft": ([_v], {}),
    "fft.fft2": ([_B], {}),
    # stats
    "stats.norm.pdf": ([_v], {}),
    "stats.norm.cdf": ([_v], {}),
    "stats.norm.ppf": ([_p], {}),
    # integer-only ufuncs (the category default feeds floats)
    "bitwise_and": ([_iA, _iB], {}),
    "bitwise_or": ([_iA, _iB], {}),
    "bitwise_xor": ([_iA, _iB], {}),
    "bitwise_not": ([_iA], {}),
    "bitwise_invert": ([_iA], {}),
    "invert": ([_iA], {}),
    "bitwise_count": ([_iA], {}),
    "bitwise_left_shift": ([_iA, _ishift], {}),
    "bitwise_right_shift": ([_iA, _ishift], {}),
    "left_shift": ([_iA, _ishift], {}),
    "right_shift": ([_iA, _ishift], {}),
    "gcd": ([_iA, _iB], {}),
    "lcm": ([_iA, _iB], {}),
    "ldexp": ([_v, _iv], {}),  # (float, int)
    # reductions that need a percentile/quantile value or a 1-D input
    "percentile": ([_B, 50], {}),
    "quantile": ([_B, 0.5], {}),
    "nanpercentile": ([_B], {"q": 50}),
    "nanquantile": ([_B], {"q": 0.5}),
    "cumulative_sum": ([_v], {}),  # 1-D: no axis required
    "cumulative_prod": ([_v], {}),
    # isclose is registry-categorized counted_unary but takes (a, b)
    "isclose": ([_v, _w], {}),
}

# Proxyable ops not yet exercised. The coverage guard requires every proxyable
# op to be here or exercised; shrinking this set (by adding EXAMPLE_OVERRIDES)
# is follow-up work. Populated from the guard's own failure output.
PENDING_EXAMPLES: frozenset[str] = frozenset(
    {
        "allclose",
        "append",
        "apply_along_axis",
        "apply_over_axes",
        "argpartition",
        "argwhere",
        "array",
        "array_equal",
        "array_equiv",
        "array_split",
        "asarray_chkfinite",
        "astype",
        "atleast_1d",
        "atleast_3d",
        "bartlett",
        "bincount",
        "blackman",
        "block",
        "bmat",
        "broadcast_arrays",
        "broadcast_shapes",
        "broadcast_to",
        "can_cast",
        "choose",
        "column_stack",
        "common_type",
        "compress",
        "concat",
        "convolve",
        "copyto",
        "corrcoef",
        "correlate",
        "cov",
        "delete",
        "diag",
        "diag_indices",
        "diag_indices_from",
        "diagflat",
        "diagonal",
        "digitize",
        "dsplit",
        "dstack",
        "ediff1d",
        "einsum_path",
        "empty_like",
        "extract",
        "fft.fftfreq",
        "fft.fftn",
        "fft.fftshift",
        "fft.hfft",
        "fft.ifft2",
        "fft.ifftn",
        "fft.ifftshift",
        "fft.ihfft",
        "fft.irfft",
        "fft.irfft2",
        "fft.irfftn",
        "fft.rfft2",
        "fft.rfftfreq",
        "fft.rfftn",
        "fill_diagonal",
        "flatnonzero",
        "fliplr",
        "flipud",
        "from_dlpack",
        "frombuffer",
        "fromfunction",
        "fromiter",
        "full_like",
        "gradient",
        "hamming",
        "hanning",
        "histogram",
        "histogram2d",
        "histogram_bin_edges",
        "histogramdd",
        "hsplit",
        "in1d",
        "indices",
        "insert",
        "interp",
        "intersect1d",
        "isdtype",
        "isfinite",
        "isfortran",
        "isin",
        "isinf",
        "isnan",
        "isscalar",
        "issubdtype",
        "iterable",
        "ix_",
        "kaiser",
        "lexsort",
        "linalg.cond",
        "linalg.cross",
        "linalg.diagonal",
        "linalg.eigvals",
        "linalg.eigvalsh",
        "linalg.lstsq",
        "linalg.matmul",
        "linalg.matrix_norm",
        "linalg.matrix_power",
        "linalg.matrix_transpose",
        "linalg.multi_dot",
        "linalg.outer",
        "linalg.slogdet",
        "linalg.svdvals",
        "linalg.tensordot",
        "linalg.tensorinv",
        "linalg.tensorsolve",
        "linalg.vecdot",
        "linalg.vector_norm",
        "load",  # file I/O: exercised by tests/test_io.py, not this array harness
        "mask_indices",
        "matrix_transpose",
        "may_share_memory",
        "meshgrid",
        "min_scalar_type",
        "mintypecode",
        "ndim",
        "nonzero",
        "ones_like",
        "packbits",
        "pad",
        "partition",
        "permute_dims",
        "piecewise",
        "place",
        "poly",
        "polyadd",
        "polyder",
        "polydiv",
        "polyfit",
        "polyint",
        "polymul",
        "polysub",
        "polyval",
        "promote_types",
        "put",
        "put_along_axis",
        "putmask",
        "random.beta",
        "random.binomial",
        "random.bytes",
        "random.chisquare",
        "random.dirichlet",
        "random.exponential",
        "random.f",
        "random.gamma",
        "random.geometric",
        "random.get_state",
        "random.gumbel",
        "random.hypergeometric",
        "random.laplace",
        "random.logistic",
        "random.lognormal",
        "random.logseries",
        "random.multinomial",
        "random.multivariate_normal",
        "random.negative_binomial",
        "random.noncentral_chisquare",
        "random.noncentral_f",
        "random.pareto",
        "random.poisson",
        "random.power",
        "random.rand",
        "random.random_integers",
        "random.random_sample",
        "random.ranf",
        "random.rayleigh",
        "random.sample",
        "random.set_state",
        "random.shuffle",
        "random.standard_cauchy",
        "random.standard_exponential",
        "random.standard_gamma",
        "random.standard_t",
        "random.triangular",
        "random.vonmises",
        "random.wald",
        "random.weibull",
        "random.zipf",
        "ravel_multi_index",
        "repeat",
        "require",
        "resize",
        "result_type",
        "roll",
        "rollaxis",
        "roots",
        "rot90",
        "row_stack",
        "save",  # file I/O: exercised by tests/test_io.py, not this array harness
        "savez",
        "savez_compressed",
        "searchsorted",
        "select",
        "setdiff1d",
        "setxor1d",
        "shape",
        "shares_memory",
        "size",
        "sort_complex",
        "split",
        "stats.cauchy.cdf",
        "stats.cauchy.pdf",
        "stats.cauchy.ppf",
        "stats.expon.cdf",
        "stats.expon.pdf",
        "stats.expon.ppf",
        "stats.laplace.cdf",
        "stats.laplace.pdf",
        "stats.laplace.ppf",
        "stats.logistic.cdf",
        "stats.logistic.pdf",
        "stats.logistic.ppf",
        "stats.lognorm.cdf",
        "stats.lognorm.pdf",
        "stats.lognorm.ppf",
        "stats.truncnorm.cdf",
        "stats.truncnorm.pdf",
        "stats.truncnorm.ppf",
        "stats.uniform.cdf",
        "stats.uniform.pdf",
        "stats.uniform.ppf",
        "take",
        "take_along_axis",
        "tile",
        "trace",
        "trapezoid",
        "trapz",
        "tril",
        "tril_indices",
        "tril_indices_from",
        "trim_zeros",
        "triu",
        "triu_indices",
        "triu_indices_from",
        "typename",
        "union1d",
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        "unpackbits",
        "unravel_index",
        "unstack",
        "unwrap",
        "vander",
        "vsplit",
        "zeros_like",
    }
)


def _has_example(op: str) -> bool:
    return op in EXAMPLE_OVERRIDES or get_category(op) in (
        COUNTED_UNARY,
        COUNTED_BINARY,
        COUNTED_REDUCTION,
    )


def _example(op: str) -> tuple[list, dict]:
    if op in EXAMPLE_OVERRIDES:
        return EXAMPLE_OVERRIDES[op]
    cat = get_category(op)
    if cat == COUNTED_UNARY:
        return [_pos], {}
    if cat == COUNTED_BINARY:
        return [_pos, _pos2], {}
    if cat == COUNTED_REDUCTION:
        return [_pos], {}
    raise AssertionError(f"no example for {op!r}")  # pragma: no cover


EXERCISED: list[str] = sorted(op for op in PROXYABLE if _has_example(op))


def _encode(arg, session: Session):
    """Encode an argument the way the client would: arrays become handles."""
    if isinstance(arg, np.ndarray):
        return {"__handle__": session.store_array(arg)}
    if isinstance(arg, (list, tuple)):
        return [_encode(item, session) for item in arg]
    return arg


@pytest.fixture()
def handler_session():
    session = Session(flop_budget=10**12)
    handler = RequestHandler(session)
    yield session, handler
    if session.is_open:
        session.close()


@pytest.mark.parametrize("op", EXERCISED)
def test_op_round_trips(op, handler_session):
    """Every exercised op must round-trip without a serialization-class error."""
    session, handler = handler_session
    args, kwargs = _example(op)
    request = {
        "op": op,
        "args": [_encode(a, session) for a in args],
        "kwargs": {k: _encode(v, session) for k, v in kwargs.items()},
    }
    resp = handler.handle(request)

    if resp.get("status") == "error":
        etype = resp.get("error_type")
        if etype in SERIALIZATION_ERRORS:
            pytest.fail(
                f"{op}: client/server boundary cannot round-trip this op — "
                f"{etype}: {resp.get('message')}"
            )
        pytest.skip(
            f"{op}: example args rejected ({etype}: {resp.get('message')}) — "
            f"not a serialization divergence; refine EXAMPLE_OVERRIDES"
        )
    assert resp.get("status") == "ok", f"{op}: unexpected response {resp!r}"


def test_default_rng_round_trips(handler_session):
    """Regression for PR #109: the exact op that broke must round-trip."""
    session, handler = handler_session
    resp = handler.handle({"op": "random.default_rng", "args": [0], "kwargs": {}})
    assert resp["status"] == "ok", resp
    assert "gen_id" in resp["result"], resp


def test_every_proxyable_op_is_classified():
    """Coverage guard: no proxyable op may be silently un-exercised."""
    proxyable = set(PROXYABLE)
    exercised = {op for op in proxyable if _has_example(op)}
    unclassified = proxyable - exercised - PENDING_EXAMPLES
    assert not unclassified, (
        f"{len(unclassified)} proxyable op(s) are neither exercised nor in "
        f"PENDING_EXAMPLES — add an EXAMPLE_OVERRIDES entry (preferred) or list "
        f"each in PENDING_EXAMPLES:\n  " + "\n  ".join(sorted(unclassified))
    )

    stale = (set(EXAMPLE_OVERRIDES) | PENDING_EXAMPLES) - proxyable
    assert not stale, f"stale (non-proxyable) entries — remove: {sorted(stale)}"

    overlap = set(EXAMPLE_OVERRIDES) & PENDING_EXAMPLES
    assert not overlap, f"ops both exercised and pending: {sorted(overlap)}"
