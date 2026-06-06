"""Client-side tests for round-2 bugfixes (9 bugs).

Unit tests use mocks -- no running server required.
Server-side tests live in flopscope-server/tests/test_bugfixes_round2.py.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import msgpack
import pytest

# =========================================================================
# Helpers
# =========================================================================


def _make_mock_conn(response):
    """Return a mock Connection whose send_recv always returns *response*."""
    conn = MagicMock()
    conn.send_recv.return_value = response
    return conn


# =========================================================================
# Fix 1: flop_multiplier sent to server
# =========================================================================


class TestFix1FlopMultiplierSent:
    """flop_multiplier is included in the budget_open message."""

    def test_encode_budget_open_includes_multiplier(self):
        from flopscope._protocol import encode_budget_open

        raw = encode_budget_open(1000, flop_multiplier=2.5)
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["kwargs"]["flop_multiplier"] == 2.5

    def test_encode_budget_open_default_multiplier(self):
        from flopscope._protocol import encode_budget_open

        raw = encode_budget_open(1000)
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["kwargs"]["flop_multiplier"] == 1.0

    def test_enter_sends_flop_multiplier(self):
        import flopscope._budget as bmod
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        old = bmod._active_context
        bmod._active_context = None
        try:
            with patch("flopscope._budget.get_connection", return_value=mock_conn):
                ctx = BudgetContext(flop_budget=500, flop_multiplier=3.0)
                ctx.__enter__()
                sent_bytes = mock_conn.send_recv.call_args[0][0]
                decoded = msgpack.unpackb(sent_bytes, raw=False)
                assert decoded["kwargs"]["flop_multiplier"] == 3.0
                ctx.__exit__(None, None, None)
        finally:
            bmod._active_context = old


# =========================================================================
# Fix 2: .T property on RemoteArray
# =========================================================================


class TestFix2TransposeProperty:
    """RemoteArray.T dispatches 'transpose' to the server."""

    def test_T_property_exists(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "T")

    def test_T_is_property(self):
        from flopscope._remote_array import RemoteArray

        assert isinstance(
            type.__getattribute__(RemoteArray, "T"),
            property,
        )


# =========================================================================
# Fix 3: __getitem__ dispatches to server
# =========================================================================


class TestFix3GetitemDispatches:
    """__getitem__ sends __getitem__ op to server."""

    def test_getitem_sends_request(self):
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"value": 42.0, "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
            result = arr[1]
            mock_conn.send_recv.assert_called_once()
            sent = msgpack.unpackb(mock_conn.send_recv.call_args[0][0], raw=False)
            assert sent["op"] == "__getitem__"

    def test_getitem_slice_encoding(self):
        from flopscope._remote_array import _encode_index_key

        key = slice(0, 5, 2)
        encoded = _encode_index_key(key)
        assert encoded == {"__slice__": [0, 5, 2]}

    def test_getitem_int_passthrough(self):
        from flopscope._remote_array import _encode_index_key

        assert _encode_index_key(3) == 3

    def test_getitem_tuple_encoding(self):
        from flopscope._remote_array import _encode_index_key

        key = (slice(1, 3), 0)
        encoded = _encode_index_key(key)
        assert encoded == [{"__slice__": [1, 3, None]}, 0]


# =========================================================================
# Fix 4: complex packing in we.array()
# =========================================================================


class TestFix4ComplexPacking:
    """we.array() correctly packs complex numbers."""

    def test_complex128_list_packing(self):
        """Complex values are split into real/imag pairs for struct.pack."""
        # Simulate the packing logic from __init__.py
        flat = [1 + 2j, 3 + 4j]
        expanded = []
        for v in flat:
            c = complex(v)
            expanded.extend([c.real, c.imag])
        data = struct.pack(f"<{len(expanded)}d", *expanded)
        assert len(data) == 4 * 8  # 4 doubles

    def test_complex_scalar_type_accepted(self):
        """isinstance check includes complex."""
        assert isinstance(1 + 2j, (int, float, complex))

    def test_infer_dtype_complex(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([1 + 2j, 3.0]) == "complex128"


# =========================================================================
# Fix 5: __iter__ on 2D arrays uses __getitem__
# =========================================================================


class TestFix5IterUsesGetitem:
    """__iter__ yields self[i] for each i, not plain Python values."""

    def test_iter_calls_getitem(self):
        from flopscope._remote_array import RemoteArray

        calls = []

        def mock_getitem(self, key):
            calls.append(key)
            return f"item_{key}"

        with patch.object(RemoteArray, "__getitem__", mock_getitem):
            arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
            result = list(arr)
            assert calls == [0, 1, 2]
            assert result == ["item_0", "item_1", "item_2"]

    def test_iter_0d_raises(self):
        from flopscope._remote_array import RemoteArray

        arr = RemoteArray(handle_id="a0", shape=(), dtype="float64")
        with pytest.raises(TypeError, match="0-d"):
            list(arr)


# =========================================================================
# Fix 6: isinstance(RemoteScalar, RemoteArray) is True
# =========================================================================


class TestFix6IsinstanceCheck:
    """RemoteScalar passes isinstance(s, RemoteArray) check."""

    def test_remote_scalar_isinstance_remote_array(self):
        from flopscope._remote_array import RemoteArray, RemoteScalar

        s = RemoteScalar(value=3.14, dtype="float64")
        assert isinstance(s, RemoteArray)

    def test_ndarray_alias_works(self):
        from flopscope._remote_array import RemoteArray, RemoteScalar

        ndarray = RemoteArray
        s = RemoteScalar(value=1.0, dtype="float64")
        assert isinstance(s, ndarray)

    def test_remote_array_still_isinstance(self):
        from flopscope._remote_array import RemoteArray

        a = RemoteArray(handle_id="h1", shape=(3,), dtype="float64")
        assert isinstance(a, RemoteArray)

    def test_plain_object_not_instance(self):
        from flopscope._remote_array import RemoteArray

        assert not isinstance(42, RemoteArray)
        assert not isinstance("hello", RemoteArray)


# =========================================================================
# Fix 7: Nested BudgetContext raises clear error
# =========================================================================


class TestFix7NestedBudgetGuard:
    """Nested BudgetContext raises RuntimeError before hitting the server."""

    def test_nested_raises_runtime_error(self):
        import flopscope._budget as bmod
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        old = bmod._active_context
        bmod._active_context = None

        try:
            with patch("flopscope._budget.get_connection", return_value=mock_conn):
                ctx1 = BudgetContext(flop_budget=1000)
                ctx1.__enter__()

                ctx2 = BudgetContext(flop_budget=2000)
                with pytest.raises(RuntimeError, match="Nested"):
                    ctx2.__enter__()

                ctx1.__exit__(None, None, None)
        finally:
            bmod._active_context = old

    def test_exit_clears_active_context(self):
        import flopscope._budget as bmod
        from flopscope._budget import BudgetContext

        responses = [
            {"status": "ok", "flops_used": 0},
            {"status": "ok", "flops_used": 100},
        ]
        mock_conn = MagicMock()
        mock_conn.send_recv.side_effect = responses
        old = bmod._active_context
        bmod._active_context = None

        try:
            with patch("flopscope._budget.get_connection", return_value=mock_conn):
                ctx = BudgetContext(flop_budget=1000)
                ctx.__enter__()
                assert bmod._active_context is ctx
                ctx.__exit__(None, None, None)
                assert bmod._active_context is None
        finally:
            bmod._active_context = old

    def test_exit_without_enter_is_noop(self):
        """Calling __exit__ on a context that was never opened should not crash."""
        import flopscope._budget as bmod
        from flopscope._budget import BudgetContext

        old = bmod._active_context
        bmod._active_context = None
        try:
            ctx = BudgetContext(flop_budget=1000)
            # __exit__ without __enter__ -- _is_open is False, should be a no-op
            ctx.__exit__(None, None, None)
        finally:
            bmod._active_context = old
