"""Exhaustive smoke test: every non-blacklisted function via client-server proxy.

Starts a real FlopscopeServer in a subprocess, then calls every proxyable
function from the registry with valid arguments.

Run with:
    PYTHONPATH=flopscope-client/src:flopscope-server/src:src \
        .venv/bin/python -m pytest flopscope-client/tests/test_exhaustive.py -v --timeout=120
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

import flopscope as flops
import flopscope.numpy as fnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
# Prefer the server's own venv (which has msgpack/pyzmq) for the server subprocess;
# fall back to the worktree root venv if it doesn't exist.
_SERVER_VENV_PYTHON = os.path.join(
    _WORKTREE, "flopscope-server", ".venv", "bin", "python"
)
_ROOT_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")
_VENV_PYTHON = (
    _SERVER_VENV_PYTHON if os.path.exists(_SERVER_VENV_PYTHON) else _ROOT_VENV_PYTHON
)

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15557"

_SERVER_SCRIPT = f"""
import sys, os
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    import flopscope._budget as _bmod

    def _force_cleanup():
        """Force-reset client state and try to close server session."""
        _bmod._active_context = None
        # Always reset connection first (kills any stuck ZMQ socket)
        reset_connection()
        # Now create a fresh connection and try to close any server session
        try:
            from flopscope._connection import get_connection
            from flopscope._protocol import encode_budget_close

            conn = get_connection()
            conn.send_recv(encode_budget_close())
        except Exception:
            pass
        # Reset again to clear the connection used for cleanup
        reset_connection()

    _force_cleanup()
    yield
    _force_cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_we():
    import flopscope as we

    return we


# ---------------------------------------------------------------------------
# Test data (created fresh in a budget context)
# ---------------------------------------------------------------------------


class _TestData:
    """Lazily-created test data for all tests."""

    _instance = None

    @classmethod
    def get(cls, we):
        # We recreate each time since handles may expire between budget contexts
        d = type("D", (), {})()
        d.SMALL_POS = we.array([0.5, 1.0, 1.5, 2.0])
        d.SMALL_UNIT = we.array([0.1, 0.3, 0.5, 0.7])
        d.SMALL_NEG = we.array([-2.0, -1.0, 1.0, 2.0])
        d.SMALL_INT = we.array([1.0, 2.0, 3.0, 4.0])
        d.PAIR_A = we.array([1.0, 2.0, 3.0])
        d.PAIR_B = we.array([4.0, 5.0, 6.0])
        d.MATRIX = we.array([[1.0, 2.0], [3.0, 4.0]])
        d.MATRIX3x2 = we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        d.VEC2 = we.array([1.0, 2.0])
        d.VEC3 = we.array([1.0, 2.0, 3.0])
        d.BOOL_ARR = we.greater(d.PAIR_A, we.array([2.0, 2.0, 2.0]))
        d.SMALL_GE1 = we.array([1.0, 1.5, 2.0, 3.0])
        d.INT_ARR = we.array([1, 2, 3, 4], dtype="int64")
        d.INT_PAIR_A = we.array([6, 12, 15], dtype="int64")
        d.INT_PAIR_B = we.array([4, 8, 10], dtype="int64")
        d.COMPLEX_ARR = we.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
        d.ZERO_TRIMMED = we.array([0.0, 1.0, 2.0, 0.0])
        return d


# ---------------------------------------------------------------------------
# Counted Unary
# ---------------------------------------------------------------------------

_UNARY_GENERAL = [
    "exp",
    "exp2",
    "expm1",
    "sqrt",
    "square",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arctan",
    "arcsinh",
    "sign",
    "ceil",
    "floor",
    "abs",
    "absolute",
    "fabs",
    "negative",
    "positive",
    "rint",
    "round",
    "around",
    "fix",
    "trunc",
    "deg2rad",
    "degrees",
    "rad2deg",
    "radians",
    "log",
    "log2",
    "log10",
    "log1p",
    "reciprocal",
    "signbit",
    "spacing",
    "sinc",
    "i0",
    "nan_to_num",
    "real",
    "imag",
    "conj",
    "conjugate",
    "iscomplex",
    "isreal",
    "real_if_close",
    "logical_not",
]


@pytest.mark.parametrize("op_name", _UNARY_GENERAL)
def test_unary_general(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.SMALL_POS)
        assert result is not None


_UNARY_UNIT = ["arcsin", "arccos", "asin", "acos"]


@pytest.mark.parametrize("op_name", _UNARY_UNIT)
def test_unary_unit(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_ARCTANH = ["arctanh", "atanh", "atan"]


@pytest.mark.parametrize("op_name", _UNARY_ARCTANH)
def test_unary_arctanh(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_GE1 = ["acosh", "arccosh"]


@pytest.mark.parametrize("op_name", _UNARY_GE1)
def test_unary_ge1(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.SMALL_GE1)
        assert result is not None


def test_modf():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.modf(d.SMALL_POS)
        assert result is not None


def test_frexp():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.frexp(d.SMALL_POS)
        assert result is not None


def test_sort_complex():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.sort_complex(d.COMPLEX_ARR)
        assert result is not None


def test_isclose():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.isclose(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_angle():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.angle(d.COMPLEX_ARR)
        assert result is not None


def test_iscomplexobj():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.iscomplexobj(d.SMALL_POS)
        assert result is not None


def test_isrealobj():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.isrealobj(d.SMALL_POS)
        assert result is not None


_BITWISE_UNARY = ["bitwise_invert", "bitwise_not", "bitwise_count", "invert"]


@pytest.mark.parametrize("op_name", _BITWISE_UNARY)
def test_bitwise_unary(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.INT_ARR)
        assert result is not None


def test_isneginf():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.isneginf(d.SMALL_NEG)
        assert result is not None


def test_isposinf():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.isposinf(d.SMALL_POS)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Binary
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "power",
    "pow",
    "float_power",
    "mod",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logaddexp",
    "logaddexp2",
    "arctan2",
    "atan2",
    "hypot",
    "copysign",
    "nextafter",
    "heaviside",
]


@pytest.mark.parametrize("op_name", _BINARY_OPS)
def test_binary(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_ldexp():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.ldexp(d.PAIR_A, we.array([1, 2, 3], dtype="int64"))
        assert result is not None


_BITWISE_BINARY = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
]


@pytest.mark.parametrize("op_name", _BITWISE_BINARY)
def test_bitwise_binary(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.INT_PAIR_A, we.array([1, 2, 1], dtype="int64"))
        assert result is not None


def test_gcd():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.gcd(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_lcm():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.lcm(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_divmod():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.divmod(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vecdot():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.vecdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Reduction
# ---------------------------------------------------------------------------

_REDUCTION_OPS = [
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "max",
    "min",
    "amax",
    "amin",
    "all",
    "any",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "count_nonzero",
    "median",
    "nansum",
    "nanprod",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmax",
    "nanmin",
    "nanmedian",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "ptp",
]


@pytest.mark.parametrize("op_name", _REDUCTION_OPS)
def test_reduction(op_name):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        fn = getattr(we, op_name)
        result = fn(d.SMALL_INT)
        assert result is not None


def test_average():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.average(d.SMALL_INT)
        assert result is not None


def test_percentile():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        # percentile server-side is wrapped as reduction(a, axis=None, **kw)
        # so q must be passed as keyword
        result = we.percentile(d.SMALL_INT, q=50)
        assert result is not None


def test_quantile():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.quantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_nanpercentile():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.nanpercentile(d.SMALL_INT, q=50)
        assert result is not None


def test_nanquantile():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.nanquantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_cumulative_sum():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.cumulative_sum(d.SMALL_INT, axis=0)
        assert result is not None


def test_cumulative_prod():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.cumulative_prod(d.SMALL_INT, axis=0)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Custom
# ---------------------------------------------------------------------------


def test_clip():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.clip(d.SMALL_INT, 1.5, 3.5)
        assert result is not None


def test_dot():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.dot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_matmul():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.matmul(d.MATRIX, d.VEC2)
        assert result is not None


def test_inner():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.inner(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_outer():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.outer(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vdot():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.vdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_tensordot():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.tensordot(d.MATRIX, d.MATRIX, 1)
        assert result is not None


def test_kron():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.kron(d.VEC2, d.VEC3)
        assert result is not None


def test_cross():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.cross(d.VEC3, we.array([4.0, 5.0, 6.0]))
        assert result is not None


def test_diff():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.diff(d.SMALL_INT)
        assert result is not None


def test_ediff1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.ediff1d(d.SMALL_INT)
        assert result is not None


def test_gradient():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.gradient(d.SMALL_INT)
        assert result is not None


def test_convolve():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.convolve(d.PAIR_A, d.VEC2)
        assert result is not None


def test_correlate():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.correlate(d.PAIR_A, d.VEC2)
        assert result is not None


def test_corrcoef():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.corrcoef(d.SMALL_INT)
        assert result is not None


def test_cov():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.cov(d.SMALL_INT)
        assert result is not None


def test_einsum():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.einsum("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


@pytest.mark.xfail(
    strict=True,
    reason="einsum_path returns a nested (path, repr-string) structure that is "
    "not serializable across the client/server proxy",
)
def test_einsum_path():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.einsum_path("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_trapezoid():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.trapezoid(d.SMALL_INT)
        assert result is not None


def test_trapz():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        result = we.trapz(d.SMALL_INT)
        assert result is not None


def test_interp():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        result = we.interp(
            we.array([1.5, 2.5]),
            we.array([1.0, 2.0, 3.0]),
            we.array([10.0, 20.0, 30.0]),
        )
        assert result is not None


def test_linalg_svd():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        U, S, Vt = we.linalg.svd(d.MATRIX3x2)
        assert U is not None
        assert S is not None
        assert Vt is not None


# ---------------------------------------------------------------------------
# Free Ops: Creation
# ---------------------------------------------------------------------------


def test_array_creation():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.array([1.0, 2.0, 3.0]) is not None


def test_zeros():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.zeros((3, 3)) is not None


def test_ones():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.ones((3, 3)) is not None


def test_full():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.full((3, 3), 7.0) is not None


def test_empty():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.empty((3, 3)) is not None


def test_eye():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.eye(3) is not None


def test_identity():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.identity(3) is not None


def test_arange():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.arange(0, 10, 2) is not None


def test_linspace():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.linspace(0, 1, 5) is not None


def test_logspace():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.logspace(0, 2, 5) is not None


def test_geomspace():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.geomspace(1, 100, 5) is not None


def test_zeros_like():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.zeros_like(d.SMALL_INT) is not None


def test_ones_like():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.ones_like(d.SMALL_INT) is not None


def test_full_like():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.full_like(d.SMALL_INT, 9.0) is not None


def test_empty_like():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.empty_like(d.SMALL_INT) is not None


def test_diag():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.diag(d.VEC3) is not None


def test_diagflat():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.diagflat(d.VEC2) is not None


def test_tri():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.tri(3, 3) is not None


def test_tril():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.tril(d.MATRIX) is not None


def test_triu():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.triu(d.MATRIX) is not None


def test_vander():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.vander(d.PAIR_A, 3) is not None


# ---------------------------------------------------------------------------
# Free Ops: Manipulation
# ---------------------------------------------------------------------------


def test_reshape():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.reshape(d.SMALL_INT, (2, 2)) is not None


def test_transpose():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.transpose(d.MATRIX) is not None


def test_swapaxes():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.swapaxes(d.MATRIX, 0, 1) is not None


def test_moveaxis():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.moveaxis(d.MATRIX, 0, 1) is not None


def test_concatenate():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.concatenate([d.PAIR_A, d.PAIR_B]) is not None


def test_stack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.stack([d.PAIR_A, d.PAIR_B]) is not None


def test_vstack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.vstack([d.PAIR_A, d.PAIR_B]) is not None


def test_hstack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.hstack([d.PAIR_A, d.PAIR_B]) is not None


def test_column_stack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.column_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_row_stack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.row_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_dstack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.dstack([d.PAIR_A, d.PAIR_B]) is not None


def test_split():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.split(we.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_hsplit():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.hsplit(we.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_vsplit():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.vsplit(we.array([[1.0, 2.0], [3.0, 4.0]]), 2) is not None


def test_dsplit():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.dsplit(we.array([[[1.0, 2.0], [3.0, 4.0]]]), 2) is not None


def test_array_split():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.array_split(d.SMALL_INT, 2) is not None


def test_squeeze():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.squeeze(we.array([[[1.0, 2.0]]])) is not None


def test_expand_dims():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.expand_dims(d.PAIR_A, 0) is not None


def test_ravel():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.ravel(d.MATRIX) is not None


def test_copy():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.copy(d.PAIR_A) is not None


def test_flip():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.flip(d.PAIR_A) is not None


def test_fliplr():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.fliplr(d.MATRIX) is not None


def test_flipud():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.flipud(d.MATRIX) is not None


def test_rot90():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.rot90(d.MATRIX) is not None


def test_roll():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.roll(d.PAIR_A, 1) is not None


def test_rollaxis():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.rollaxis(d.MATRIX, 1) is not None


def test_tile():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.tile(d.PAIR_A, 2) is not None


def test_repeat():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.repeat(d.PAIR_A, 2) is not None


def test_resize():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.resize(d.PAIR_A, (2, 3)) is not None


def test_append():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.append(d.PAIR_A, d.PAIR_B) is not None


def test_insert():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.insert(d.PAIR_A, 1, 99.0) is not None


def test_delete():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.delete(d.PAIR_A, 1) is not None


def test_unique():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_trim_zeros():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.trim_zeros(d.ZERO_TRIMMED) is not None


def test_sort():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.sort(we.array([3.0, 1.0, 2.0])) is not None


def test_argsort():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argsort(we.array([3.0, 1.0, 2.0])) is not None


def test_partition():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.partition(we.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_argpartition():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argpartition(we.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_take():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.take(d.PAIR_A, [0, 2]) is not None


def test_take_along_axis():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.take_along_axis(
                we.array([[1.0, 2.0], [3.0, 4.0]]),
                we.array([[0, 1], [1, 0]], dtype="int64"),
                axis=1,
            )
            is not None
        )


def test_compress():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.compress([True, False, True], d.PAIR_A) is not None


def test_extract():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.extract(d.BOOL_ARR, d.PAIR_A) is not None


def test_diagonal():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.diagonal(d.MATRIX) is not None


def test_trace():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.trace(d.MATRIX) is not None


def test_pad():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.pad(d.PAIR_A, 1) is not None


def test_searchsorted():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.searchsorted(we.array([1.0, 3.0, 5.0]), 2.0) is not None


def test_where():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.where(d.BOOL_ARR, d.PAIR_A, d.PAIR_B) is not None


def test_select():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.select([d.BOOL_ARR], [d.PAIR_A]) is not None


def test_nonzero():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.nonzero(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_argwhere():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argwhere(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_flatnonzero():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.flatnonzero(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_isin():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.isin(d.PAIR_A, [1.0, 3.0]) is not None


def test_in1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.in1d(d.PAIR_A, we.array([1.0, 3.0])) is not None


def test_intersect1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.intersect1d(d.PAIR_A, we.array([2.0, 3.0, 7.0])) is not None


def test_union1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.union1d(d.PAIR_A, we.array([2.0, 5.0])) is not None


def test_setdiff1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.setdiff1d(d.PAIR_A, we.array([2.0])) is not None


def test_setxor1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.setxor1d(d.PAIR_A, we.array([2.0, 5.0])) is not None


def test_allclose():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.allclose(d.PAIR_A, d.PAIR_A) is not None


def test_array_equal():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.array_equal(d.PAIR_A, d.PAIR_A) is not None


def test_array_equiv():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.array_equiv(d.PAIR_A, d.PAIR_A) is not None


def test_isfinite():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.isfinite(d.SMALL_POS) is not None


def test_isinf():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.isinf(d.SMALL_POS) is not None


def test_isnan():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.isnan(d.SMALL_POS) is not None


def test_isscalar():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.isscalar(5.0) is not None


def test_shape():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.shape(d.MATRIX) is not None


def test_ndim():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.ndim(d.MATRIX) is not None


def test_size():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.size(d.MATRIX) is not None


def test_broadcast_to():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.broadcast_to(d.PAIR_A, (2, 3)) is not None


def test_broadcast_arrays():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.broadcast_arrays(d.PAIR_A, d.PAIR_B) is not None


def test_broadcast_shapes():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.broadcast_shapes((3,), (3,)) is not None


def test_histogram():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.histogram(d.SMALL_INT, 3) is not None


def test_histogram_bin_edges():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.histogram_bin_edges(d.SMALL_INT, 3) is not None


def test_histogram2d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.histogram2d(
                we.array([1.0, 2.0, 3.0]),
                we.array([4.0, 5.0, 6.0]),
                3,
            )
            is not None
        )


def test_histogramdd():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.histogramdd(
                we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                3,
            )
            is not None
        )


def test_asarray():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.asarray(d.PAIR_A) is not None


def test_asarray_chkfinite():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.asarray_chkfinite(d.PAIR_A) is not None


def test_astype():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.astype(d.PAIR_A, "float32") is not None


def test_can_cast():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.can_cast("float32", "float64") is not None


def test_result_type():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.result_type("float32", "float64") is not None


def test_promote_types():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.promote_types("float32", "float64") is not None


def test_atleast_1d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.atleast_1d(d.PAIR_A) is not None


def test_atleast_2d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.atleast_2d(d.PAIR_A) is not None


def test_atleast_3d():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.atleast_3d(d.PAIR_A) is not None


def test_diag_indices():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.diag_indices(3) is not None


def test_diag_indices_from():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.diag_indices_from(d.MATRIX) is not None


def test_tril_indices():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.tril_indices(3) is not None


def test_triu_indices():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.triu_indices(3) is not None


def test_tril_indices_from():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.tril_indices_from(d.MATRIX) is not None


def test_triu_indices_from():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.triu_indices_from(d.MATRIX) is not None


def test_indices():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.indices((2, 3)) is not None


def test_ix_():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.ix_(
                we.array([0, 1], dtype="int64"),
                we.array([0, 1], dtype="int64"),
            )
            is not None
        )


def test_unravel_index():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unravel_index(5, (3, 3)) is not None


def test_ravel_multi_index():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.ravel_multi_index(
                (
                    we.array([0, 1, 2], dtype="int64"),
                    we.array([0, 1, 2], dtype="int64"),
                ),
                (3, 3),
            )
            is not None
        )


@pytest.mark.xfail(
    strict=True,
    reason="mask_indices requires a callable (mask function) argument, which "
    "cannot be serialized across the client/server proxy",
)
def test_mask_indices():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.mask_indices(3, we.triu) is not None


def test_meshgrid():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.meshgrid(we.array([1.0, 2.0]), we.array([3.0, 4.0])) is not None


def test_lexsort():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.lexsort((we.array([1.0, 2.0, 1.0]), we.array([3.0, 1.0, 2.0])))
            is not None
        )


def test_digitize():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.digitize(we.array([0.5, 1.5, 2.5]), we.array([1.0, 2.0, 3.0]))
            is not None
        )


def test_bincount():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.bincount(we.array([0, 1, 1, 2, 3, 3, 3], dtype="int64")) is not None


def test_packbits():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.packbits(we.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8")) is not None
        )


def test_unpackbits():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unpackbits(we.array([177], dtype="uint8")) is not None


def test_block():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.block([[d.PAIR_A, d.PAIR_B]]) is not None


def test_concat():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.concat([d.PAIR_A, d.PAIR_B]) is not None


def test_iterable():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.iterable(d.PAIR_A) is not None


def test_isfortran():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.isfortran(d.MATRIX) is not None


def test_typename():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        # numpy.typename expects a single-char type code (e.g. "d"), not a
        # dtype name. Returns a str description, which serializes fine.
        assert we.typename("d") is not None


def test_mintypecode():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.mintypecode(["f", "d"]) is not None


def test_base_repr_blacklisted():
    # base_repr is a string-formatting helper, intentionally blacklisted (not
    # proxyable); accessing it raises AttributeError before any dispatch.
    we = _import_we()
    with pytest.raises(AttributeError, match="not supported"):
        _ = we.base_repr


def test_binary_repr_blacklisted():
    # binary_repr is a string-formatting helper, intentionally blacklisted (not
    # proxyable); accessing it raises AttributeError before any dispatch.
    we = _import_we()
    with pytest.raises(AttributeError, match="not supported"):
        _ = we.binary_repr


def test_put():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.put(arr, [0, 2], [99.0, 88.0])


def test_place():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.place(arr, we.array([True, False, True, False], dtype="bool"), [99.0, 88.0])


def test_putmask():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.putmask(arr, we.array([True, False, True, False], dtype="bool"), 0.0)


def test_fill_diagonal():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        mat = we.array([[1.0, 2.0], [3.0, 4.0]])
        we.fill_diagonal(mat, 0.0)


def test_copyto():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0])
        we.copyto(arr, we.array([9.0, 8.0, 7.0]))


def test_choose():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.choose(
                we.array([0, 1, 0], dtype="int64"),
                [we.array([10.0, 20.0, 30.0]), we.array([40.0, 50.0, 60.0])],
            )
            is not None
        )


def test_require():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.require(d.PAIR_A) is not None


def test_matrix_transpose():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.matrix_transpose(we.array([[[1.0, 2.0], [3.0, 4.0]]])) is not None


def test_permute_dims():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.permute_dims(d.MATRIX, (1, 0)) is not None


def test_unique_all():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_all(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_counts():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_counts(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_inverse():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_inverse(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_values():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_values(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unstack():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.unstack(d.MATRIX) is not None


def test_shares_memory():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.shares_memory(d.PAIR_A, d.PAIR_B) is not None


def test_may_share_memory():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.may_share_memory(d.PAIR_A, d.PAIR_B) is not None


def test_min_scalar_type():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.min_scalar_type(10) is not None


def test_issubdtype():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        assert we.issubdtype("float64", "float64") is not None


@pytest.mark.xfail(
    strict=True,
    reason="common_type returns a Python type object, which is not "
    "serializable across the client/server proxy",
)
def test_common_type():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(we)
        assert we.common_type(d.PAIR_A) is not None


def test_put_along_axis():
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        we.put_along_axis(
            we.array([[1.0, 2.0], [3.0, 4.0]]),
            we.array([[0], [1]], dtype="int64"),
            we.array([[99.0], [88.0]]),
            axis=1,
        )


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

_RANDOM_SIMPLE = [
    ("random.rand", lambda we: we.random.rand(3, 3)),
    ("random.randn", lambda we: we.random.randn(3, 3)),
    ("random.random", lambda we: we.random.random((3,))),
    ("random.randint", lambda we: we.random.randint(0, 10, (3,))),
    ("random.uniform", lambda we: we.random.uniform(0.0, 1.0, (3,))),
    ("random.normal", lambda we: we.random.normal(0.0, 1.0, (3,))),
    ("random.seed", lambda we: we.random.seed(42)),
    ("random.choice", lambda we: we.random.choice(10, 3)),
    ("random.permutation", lambda we: we.random.permutation(5)),
    ("random.beta", lambda we: we.random.beta(2.0, 5.0, (3,))),
    ("random.binomial", lambda we: we.random.binomial(10, 0.5, (3,))),
    ("random.chisquare", lambda we: we.random.chisquare(2, (3,))),
    ("random.exponential", lambda we: we.random.exponential(1.0, (3,))),
    ("random.gamma", lambda we: we.random.gamma(2.0, 1.0, (3,))),
    ("random.geometric", lambda we: we.random.geometric(0.5, (3,))),
    ("random.gumbel", lambda we: we.random.gumbel(0.0, 1.0, (3,))),
    ("random.laplace", lambda we: we.random.laplace(0.0, 1.0, (3,))),
    ("random.logistic", lambda we: we.random.logistic(0.0, 1.0, (3,))),
    ("random.lognormal", lambda we: we.random.lognormal(0.0, 1.0, (3,))),
    ("random.logseries", lambda we: we.random.logseries(0.9, (3,))),
    ("random.multinomial", lambda we: we.random.multinomial(10, [0.5, 0.3, 0.2])),
    ("random.negative_binomial", lambda we: we.random.negative_binomial(5, 0.5, (3,))),
    (
        "random.noncentral_chisquare",
        lambda we: we.random.noncentral_chisquare(2, 1.0, (3,)),
    ),
    ("random.noncentral_f", lambda we: we.random.noncentral_f(5, 10, 1.0, (3,))),
    ("random.pareto", lambda we: we.random.pareto(2.0, (3,))),
    ("random.poisson", lambda we: we.random.poisson(5.0, (3,))),
    ("random.power", lambda we: we.random.power(2.0, (3,))),
    ("random.rayleigh", lambda we: we.random.rayleigh(1.0, (3,))),
    ("random.standard_cauchy", lambda we: we.random.standard_cauchy((3,))),
    ("random.standard_exponential", lambda we: we.random.standard_exponential((3,))),
    ("random.standard_gamma", lambda we: we.random.standard_gamma(2.0, (3,))),
    ("random.standard_normal", lambda we: we.random.standard_normal((3,))),
    ("random.standard_t", lambda we: we.random.standard_t(5.0, (3,))),
    ("random.triangular", lambda we: we.random.triangular(0.0, 0.5, 1.0, (3,))),
    ("random.vonmises", lambda we: we.random.vonmises(0.0, 1.0, (3,))),
    ("random.wald", lambda we: we.random.wald(1.0, 1.0, (3,))),
    ("random.weibull", lambda we: we.random.weibull(2.0, (3,))),
    ("random.zipf", lambda we: we.random.zipf(2.0, (3,))),
    ("random.dirichlet", lambda we: we.random.dirichlet([1.0, 1.0, 1.0])),
    (
        "random.multivariate_normal",
        lambda we: we.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    ),
    ("random.f", lambda we: we.random.f(5, 10, (3,))),
    ("random.hypergeometric", lambda we: we.random.hypergeometric(10, 5, 7, (3,))),
    ("random.random_sample", lambda we: we.random.random_sample((3,))),
    ("random.ranf", lambda we: we.random.ranf((3,))),
    ("random.sample", lambda we: we.random.sample((3,))),
    ("random.get_state", lambda we: we.random.get_state()),
    (
        "random.shuffle",
        lambda we: we.random.shuffle(we.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ),
]


@pytest.mark.parametrize(
    "name,call", _RANDOM_SIMPLE, ids=[n for n, _ in _RANDOM_SIMPLE]
)
def test_random(name, call):
    we = _import_we()
    with we.BudgetContext(flop_budget=10**9):
        call(we)
