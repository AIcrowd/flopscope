"""Adversarial test suite for flopscope client-server system.

Designed to find bugs by testing edge cases, data integrity,
operator correctness, and error handling.
"""

from __future__ import annotations

import math
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

    reset_connection()
    yield
    reset_connection()


# ===========================================================================
# Category 1: Data Integrity
# ===========================================================================


class TestDataIntegrity:
    def test_3d_nested_list_roundtrip(self):
        import flopscope as we

        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array(data)
            assert a.shape == (2, 2, 2)
            result = a.tolist()
            assert result == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    def test_float32_dtype_preserved(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1.0, 2.0, 3.0], dtype="float32")
            assert a.dtype == "float32"
            values = a.tolist()
            assert values == [1.0, 2.0, 3.0]

    def test_int64_dtype_inferred(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1, 2, 3])
            assert a.dtype == "int64"
            assert a.tolist() == [1, 2, 3]

    def test_very_small_numbers(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1e-300, 2e-300, 3e-300])
            values = a.tolist()
            assert abs(values[0] - 1e-300) < 1e-310
            assert abs(values[1] - 2e-300) < 1e-310
            assert abs(values[2] - 3e-300) < 1e-310

    def test_very_large_numbers(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([1e300, 2e300, 3e300])
            values = a.tolist()
            assert values[0] == 1e300
            assert values[1] == 2e300
            assert values[2] == 3e300

    def test_inf_and_nan_survive(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([float("inf"), float("-inf"), float("nan")])
            values = a.tolist()
            assert values[0] == float("inf")
            assert values[1] == float("-inf")
            assert math.isnan(values[2])

    def test_negative_zero(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([-0.0])
            values = a.tolist()
            assert values[0] == 0.0
            # Check it's actually negative zero
            assert math.copysign(1.0, values[0]) == -1.0

    def test_0d_scalar_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array(42.0)
            assert a.shape == ()
            assert a.ndim == 0
            result = a.tolist()
            assert result == 42.0

    def test_empty_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            a = we.array([])
            assert a.shape == (0,)
            result = a.tolist()
            assert result == []


# ===========================================================================
# Category 2: Operation Chains
# ===========================================================================


class TestOperationChains:
    def test_exp_log_roundtrip(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.ones((100,))
            y = we.exp(we.log(x))
            values = y.tolist()
            for v in values:
                assert abs(v - 1.0) < 1e-10, f"exp(log(1)) = {v}, expected 1.0"

    def test_sum_axis_0(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1, 2], [3, 4]])
            y = we.sum(x, axis=0)
            assert y.shape == (2,)
            assert y.tolist() == [4, 6]

    def test_sum_axis_1(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1, 2], [3, 4]])
            y = we.sum(x, axis=1)
            assert y.shape == (2,)
            assert y.tolist() == [3, 7]

    def test_einsum_matvec(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            W = we.array([[1, 0], [0, 1]])
            x = we.array([3, 4])
            y = we.einsum("ij,j->i", W, x)
            assert y.tolist() == [3, 4]

    def test_chain_10_operations(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=10_000_000):
            x = we.ones((10,))
            x = we.exp(x)  # e^1
            x = we.log(x)  # 1
            x = we.abs(x)  # 1
            x = we.negative(x)  # -1
            x = we.negative(x)  # 1
            x = we.exp(x)  # e
            x = we.log(x)  # 1
            x = we.abs(x)  # 1
            s = we.sum(x)  # 10
            assert abs(float(s) - 10.0) < 1e-6


# ===========================================================================
# Category 3: Operators
# ===========================================================================


class TestOperators:
    def test_add_two_remote_arrays(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            y = we.array([10.0, 20.0, 30.0])
            z = x + y
            assert z.tolist() == [11.0, 22.0, 33.0]

    def test_scalar_left_multiply(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            z = 2.0 * x
            assert z.tolist() == [2.0, 4.0, 6.0]

    def test_scalar_right_multiply(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            z = x * 2.0
            assert z.tolist() == [2.0, 4.0, 6.0]

    def test_radd(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            z = 2.0 + x
            assert z.tolist() == [3.0, 4.0, 5.0]

    def test_sub_differs_from_rsub(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            a = x - 2.0
            b = 2.0 - x
            assert a.tolist() == [-1.0, 0.0, 1.0]
            assert b.tolist() == [1.0, 0.0, -1.0]

    def test_pow_differs_from_rpow(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            a = x**2
            b = 2**x
            assert a.tolist() == [1.0, 4.0, 9.0]
            b_vals = b.tolist()
            assert abs(b_vals[0] - 2.0) < 1e-10
            assert abs(b_vals[1] - 4.0) < 1e-10
            assert abs(b_vals[2] - 8.0) < 1e-10

    def test_floordiv_and_rfloordiv(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([5.0, 7.0, 9.0])
            a = x // 2
            b = 2 // x
            # 5//2=2, 7//2=3, 9//2=4
            assert a.tolist() == [2.0, 3.0, 4.0]
            # 2//5=0, 2//7=0, 2//9=0
            assert b.tolist() == [0.0, 0.0, 0.0]

    def test_neg_and_abs(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, -2.0, 3.0])
            neg = -x
            assert neg.tolist() == [-1.0, 2.0, -3.0]
            absneg = abs(-x)
            assert absneg.tolist() == [1.0, 2.0, 3.0]

    def test_matmul_operator(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            A = we.array([[1.0, 2.0], [3.0, 4.0]])
            B = we.array([[5.0, 6.0], [7.0, 8.0]])
            C = A @ B
            # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
            # [[19, 22], [43, 50]]
            assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]

    def test_comparison_returns_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([0.1, 0.6, 0.3, 0.8])
            mask = x > 0.5
            # Should be a RemoteArray with bool values
            from flopscope import RemoteArray

            assert isinstance(mask, RemoteArray)
            values = mask.tolist()
            assert values == [False, True, False, True]

    def test_chained_arithmetic(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0])
            y = we.array([3.0, 4.0])
            z = we.array([2.0, 2.0])
            result = (x + y) * z
            assert result.tolist() == [8.0, 12.0]


# ===========================================================================
# Category 4: Methods on RemoteArray
# ===========================================================================


class TestRemoteArrayMethods:
    def test_shape_dtype_ndim_size_nbytes(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.zeros((3, 4))
            assert x.shape == (3, 4)
            assert x.dtype == "float64"
            assert x.ndim == 2
            assert x.size == 12
            assert x.nbytes == 96  # 12 * 8 bytes

    def test_transpose_2d(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            t = x.T
            assert t.shape == (3, 2)
            assert t.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_reshape_with_args(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            y = x.reshape(2, 3)
            assert y.shape == (2, 3)
            assert y.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_reshape_with_tuple(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            y = x.reshape((2, 3))
            assert y.shape == (2, 3)
            assert y.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_sum_no_axis(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            s = x.sum()
            assert float(s) == 10.0

    def test_sum_with_axis(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            s = x.sum(axis=0)
            assert s.tolist() == [4.0, 6.0]

    def test_mean(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0, 4.0])
            m = x.mean()
            assert float(m) == 2.5

    def test_max(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([3.0, 1.0, 4.0, 1.0, 5.0])
            assert float(x.max()) == 5.0

    def test_min(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([3.0, 1.0, 4.0, 1.0, 5.0])
            assert float(x.min()) == 1.0

    def test_flatten(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            f = x.flatten()
            assert f.shape == (4,)
            assert f.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_copy_different_handle(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            y = x.copy()
            assert y.handle_id != x.handle_id
            assert y.tolist() == x.tolist()

    def test_astype(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            y = x.astype("float32")
            assert y.dtype == "float32"
            assert y.tolist() == [1.0, 2.0, 3.0]

    def test_tolist_nested(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            result = x.tolist()
            assert isinstance(result, list)
            assert isinstance(result[0], list)
            assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_dot(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            y = we.array([4.0, 5.0, 6.0])
            d = x.dot(y)
            # 1*4 + 2*5 + 3*6 = 32
            assert float(d) == 32.0


# ===========================================================================
# Category 5: Indexing
# ===========================================================================


class TestIndexing:
    def test_1d_integer_index_returns_scalar(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            val = x[0]
            # Should be usable as a float
            assert float(val) == 10.0

    def test_2d_integer_index_returns_1d(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            row = x[0]
            from flopscope import RemoteArray

            assert isinstance(row, RemoteArray)
            assert row.tolist() == [1.0, 2.0]

    def test_slice_returns_remote_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0, 40.0, 50.0])
            sliced = x[1:3]
            from flopscope import RemoteArray

            assert isinstance(sliced, RemoteArray)
            assert sliced.shape == (2,)
            assert sliced.tolist() == [20.0, 30.0]

    def test_negative_indexing(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            val = x[-1]
            assert float(val) == 30.0

    def test_tuple_indexing_2d(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            val = x[0, 1]
            assert float(val) == 2.0

    def test_fancy_indexing_argsort(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([3.0, 1.0, 2.0])
            idx = we.argsort(x)
            y = x[idx]
            assert y.tolist() == [1.0, 2.0, 3.0]


# ===========================================================================
# Category 6: Iteration and Data Access
# ===========================================================================


class TestIterationAndDataAccess:
    def test_list_of_1d_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            values = list(x)
            assert len(values) == 3
            # Each value should be comparable to a float
            assert float(values[0]) == 10.0
            assert float(values[1]) == 20.0
            assert float(values[2]) == 30.0

    def test_comprehension_1d(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([10.0, 20.0, 30.0])
            values = list(x)
            assert len(values) == 3

    def test_iterate_2d_rows(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0], [3.0, 4.0]])
            rows = []
            for row in x:
                rows.append(row.tolist())
            assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_float_conversion_scalar_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([42.0])
            assert float(x) == 42.0

    def test_int_conversion_scalar_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([42.0])
            assert int(x) == 42

    def test_bool_conversion_scalar_array(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0])
            assert bool(x) is True
            y = we.array([0.0])
            assert bool(y) is False

    def test_print_does_not_crash(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            s = str(x)
            assert "1.0" in s

    def test_len_returns_first_dimension(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            assert len(x) == 2

    def test_fstring_works(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            s = f"value: {x}"
            assert "value:" in s


# ===========================================================================
# Category 7: Error Cases
# ===========================================================================


class TestErrorCases:
    def test_operation_outside_budget_context_raises(self):
        with pytest.raises((flops.NoBudgetContextError, flops.FlopscopeServerError)):
            fnp.ones((3,))

    def test_budget_exhaustion_raises(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1):
            a = we.ones((100,))
            with pytest.raises(we.BudgetExhaustedError):
                we.exp(a)

    def test_blacklisted_function_raises_attribute_error(self):
        import flopscope as we

        # save/load/savez are supported pickle-free I/O now; savetxt (text I/O)
        # remains blacklisted.
        with pytest.raises(AttributeError, match="blacklisted"):
            _ = we.savetxt

    def test_unknown_function_raises_attribute_error(self):
        import flopscope as we

        with pytest.raises(AttributeError):
            _ = we.nonexistent_function_xyz_12345

    def test_setitem_raises_type_error(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            with pytest.raises(TypeError, match="immutable"):
                x[0] = 5

    def test_bool_multi_element_raises_value_error(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000):
            x = we.array([1.0, 2.0, 3.0])
            with pytest.raises(ValueError):
                bool(x)


# ===========================================================================
# Category 8: Budget Tracking
# ===========================================================================


class TestBudgetTracking:
    def test_flops_used_increments(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=10_000_000) as ctx:
            a = we.ones((10,))
            _ = we.exp(a)
            ctx.summary()
            flops_after_exp = ctx.flops_used
            assert flops_after_exp > 0

            _ = we.exp(a)
            ctx.summary()
            flops_after_second = ctx.flops_used
            assert flops_after_second > flops_after_exp

    def test_flops_remaining_decreases(self):
        import flopscope as we

        budget = 10_000_000
        with we.BudgetContext(flop_budget=budget) as ctx:
            a = we.ones((10,))
            _ = we.exp(a)
            ctx.summary()
            assert ctx.flops_remaining < budget
            assert ctx.flops_remaining == budget - ctx.flops_used

    def test_summary_returns_string_with_numbers(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=1_000_000) as ctx:
            _ = we.ones((10,))
            _ = we.exp(we.ones((10,)))
            s = ctx.summary()
            assert isinstance(s, str)
            # Should contain some digits
            assert any(c.isdigit() for c in s)


# ===========================================================================
# Category 9: Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_large_array_without_fetch(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=10_000_000):
            z = we.zeros((10000,))
            assert z.shape == (10000,)
            # Don't fetch the data -- just verify the handle works
            s = we.sum(z)
            assert float(s) == 0.0

    def test_many_operations_without_fetch(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=100_000_000):
            x = we.ones((10,))
            for _ in range(50):
                x = we.exp(we.log(x))
            # After 50 roundtrips of exp(log(x)), should still be ~ones
            values = x.tolist()
            for v in values:
                assert abs(v - 1.0) < 1e-6, f"After 50 exp/log cycles: {v}"

    def test_create_many_arrays(self):
        import flopscope as we

        with we.BudgetContext(flop_budget=100_000_000):
            arrays = []
            for i in range(100):
                arrays.append(we.array([float(i)]))
            # Verify the last one
            assert float(arrays[99]) == 99.0
            # Verify the first one is still accessible
            assert float(arrays[0]) == 0.0
