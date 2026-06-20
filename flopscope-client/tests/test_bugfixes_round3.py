"""Client-side tests for round-3 bugfixes (8 bugs).

Unit tests use mocks -- no running server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

# =========================================================================
# Helpers
# =========================================================================


def _make_mock_conn(response):
    """Return a mock Connection whose send_recv always returns *response*."""
    conn = MagicMock()
    conn.send_recv.return_value = response
    return conn


# =========================================================================
# Fix 1: _encode_arg recursively encodes RemoteArrays in list/tuple args
# =========================================================================


class TestFix1EncodeArgRecursive:
    """_encode_arg recursively encodes RemoteArray/RemoteScalar in containers."""

    def test_encode_arg_remote_array(self):
        from flopscope._remote_array import RemoteArray, _encode_arg

        arr = RemoteArray(handle_id="a5", shape=(3,), dtype="float64")
        assert _encode_arg(arr) == {"__handle__": "a5"}

    def test_encode_arg_remote_scalar(self):
        from flopscope._remote_array import RemoteScalar, _encode_arg

        s = RemoteScalar(value=3.14, dtype="float64")
        assert _encode_arg(s) == 3.14

    def test_encode_arg_list_of_remote_arrays(self):
        from flopscope._remote_array import RemoteArray, _encode_arg

        a = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        b = RemoteArray(handle_id="a1", shape=(3,), dtype="float64")
        result = _encode_arg([a, b])
        assert result == [{"__handle__": "a0"}, {"__handle__": "a1"}]

    def test_encode_arg_tuple_of_remote_arrays(self):
        from flopscope._remote_array import RemoteArray, _encode_arg

        a = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        b = RemoteArray(handle_id="a1", shape=(3,), dtype="float64")
        result = _encode_arg((a, b))
        assert result == [{"__handle__": "a0"}, {"__handle__": "a1"}]

    def test_encode_arg_nested_list(self):
        from flopscope._remote_array import RemoteArray, _encode_arg

        a = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        result = _encode_arg([[a], [a]])
        assert result == [[{"__handle__": "a0"}], [{"__handle__": "a0"}]]

    def test_encode_arg_plain_value(self):
        from flopscope._remote_array import _encode_arg

        assert _encode_arg(42) == 42
        assert _encode_arg("hello") == "hello"
        assert _encode_arg(3.14) == 3.14

    def test_make_proxy_encodes_list_args(self):
        """_make_proxy should encode RemoteArrays inside list arguments."""
        from flopscope._remote_array import RemoteArray, _encode_arg

        a = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        b = RemoteArray(handle_id="a1", shape=(3,), dtype="float64")
        # Verify the encoding directly (avoids needing a running server)
        result = _encode_arg([a, b])
        assert result == [{"__handle__": "a0"}, {"__handle__": "a1"}]

    def test_make_proxy_encodes_tuple_args(self):
        """_make_proxy should encode RemoteArrays inside tuple arguments."""
        from flopscope._remote_array import RemoteArray, RemoteScalar, _encode_arg

        a = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        s = RemoteScalar(value=1.0, dtype="float64")
        result = _encode_arg((a, s, 42))
        assert result == [{"__handle__": "a0"}, 1.0, 42]


# =========================================================================
# Fix 2: _encode_index_key handles RemoteArray fancy indexes
# =========================================================================


class TestFix2FancyIndexEncoding:
    """_encode_index_key handles RemoteArray and RemoteScalar."""

    def test_encode_remote_array_key(self):
        from flopscope._remote_array import RemoteArray, _encode_index_key

        idx = RemoteArray(handle_id="a3", shape=(5,), dtype="int64")
        result = _encode_index_key(idx)
        assert result == {"__handle__": "a3"}

    def test_encode_remote_scalar_key(self):
        from flopscope._remote_array import RemoteScalar, _encode_index_key

        idx = RemoteScalar(value=2, dtype="int64")
        result = _encode_index_key(idx)
        assert result == 2

    def test_encode_list_key_with_remote_array(self):
        from flopscope._remote_array import RemoteArray, _encode_index_key

        idx = RemoteArray(handle_id="a3", shape=(5,), dtype="int64")
        # list fancy index containing a RemoteArray
        result = _encode_index_key([idx])
        assert result == [{"__handle__": "a3"}]


# =========================================================================
# Fix 3: Missing methods on RemoteArray
# =========================================================================


class TestFix3RemoteArrayMethods:
    """RemoteArray has reshape, sum, mean, astype, etc."""

    def test_has_reshape(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "reshape")

    def test_has_sum(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "sum")

    def test_has_mean(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "mean")

    def test_has_max(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "max")

    def test_has_min(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "min")

    def test_has_astype(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "astype")

    def test_has_flatten(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "flatten")

    def test_has_ravel(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "ravel")

    def test_has_transpose_method(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "transpose")

    def test_has_dot(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "dot")

    def test_has_copy(self):
        from flopscope._remote_array import RemoteArray

        assert hasattr(RemoteArray, "copy")

    def test_reshape_dispatches(self):
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"id": "a1", "shape": [2, 3], "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(6,), dtype="float64")
            result = arr.reshape(2, 3)
            assert result.shape == (2, 3)

    def test_reshape_with_tuple_arg(self):
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"id": "a1", "shape": [2, 3], "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(6,), dtype="float64")
            result = arr.reshape((2, 3))
            assert result.shape == (2, 3)

    def test_sum_dispatches(self):
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"value": 6.0, "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
            result = arr.sum()
            assert float(result) == 6.0

    def test_sum_with_axis(self):
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"id": "a1", "shape": [3], "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(2, 3), dtype="float64")
            result = arr.sum(axis=0)
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["kwargs"]["axis"] == 0

    def test_dispatch_op_passes_kwargs(self):
        """_dispatch_op forwards kwargs to encode_request."""
        from flopscope._remote_array import RemoteArray

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"id": "a1", "shape": [3], "dtype": "float64"},
            }
        )
        with patch("flopscope._connection.get_connection", return_value=mock_conn):
            arr = RemoteArray(handle_id="a0", shape=(2, 3), dtype="float64")
            arr.mean(axis=1, keepdims=True)
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["kwargs"]["axis"] == 1
            assert decoded["kwargs"]["keepdims"] is True


# =========================================================================
# Fix 4: RemoteScalar arithmetic operators
# =========================================================================


class TestFix4RemoteScalarArithmetic:
    """RemoteScalar supports +, -, *, /, //, %, **, neg, abs."""

    def test_add(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        b = RemoteScalar(2.0, "float64")
        result = a + b
        assert isinstance(result, RemoteScalar)
        assert result._value == 5.0

    def test_add_plain_number(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        result = a + 2.0
        assert result._value == 5.0

    def test_radd(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        result = 10 + a
        assert result._value == 13.0

    def test_sub(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(5.0, "float64")
        b = RemoteScalar(2.0, "float64")
        assert (a - b)._value == 3.0

    def test_rsub(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        assert (10 - a)._value == 7.0

    def test_mul(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        b = RemoteScalar(4.0, "float64")
        assert (a * b)._value == 12.0

    def test_rmul(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        assert (5 * a)._value == 15.0

    def test_truediv(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(10.0, "float64")
        b = RemoteScalar(4.0, "float64")
        assert (a / b)._value == 2.5

    def test_rtruediv(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(4.0, "float64")
        assert (20 / a)._value == 5.0

    def test_floordiv(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(7.0, "float64")
        b = RemoteScalar(3.0, "float64")
        assert (a // b)._value == 2.0

    def test_rfloordiv(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        assert (7 // a)._value == 2.0

    def test_mod(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(7.0, "float64")
        b = RemoteScalar(3.0, "float64")
        assert (a % b)._value == 1.0

    def test_rmod(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        assert (7 % a)._value == 1.0

    def test_pow(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(2.0, "float64")
        b = RemoteScalar(3.0, "float64")
        assert (a**b)._value == 8.0

    def test_rpow(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(3.0, "float64")
        assert (2**a)._value == 8.0

    def test_neg(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(5.0, "float64")
        assert (-a)._value == -5.0

    def test_abs(self):
        from flopscope._remote_array import RemoteScalar

        a = RemoteScalar(-7.0, "float64")
        assert abs(a)._value == 7.0

    def test_scalar_iteration_sum(self):
        """Summing scalars from iteration should work."""
        from flopscope._remote_array import RemoteScalar

        scalars = [
            RemoteScalar(1.0, "float64"),
            RemoteScalar(2.0, "float64"),
            RemoteScalar(3.0, "float64"),
        ]
        total = scalars[0]
        for s in scalars[1:]:
            total = total + s
        assert total._value == 6.0


# =========================================================================
# Fix 5: _infer_dtype returns int64 for int lists
# =========================================================================


class TestFix5InferDtypeInt:
    """_infer_dtype returns 'int64' for pure integer lists."""

    def test_int_list_gives_int64(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([1, 2, 3]) == "int64"

    def test_float_list_gives_float64(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([1.0, 2.0]) == "float64"

    def test_mixed_gives_float64(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([1, 2.0]) == "float64"

    def test_bool_list_gives_bool(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([True, False]) == "bool"

    def test_complex_gives_complex128(self):
        from flopscope import _infer_dtype

        assert _infer_dtype([1 + 2j]) == "complex128"


# =========================================================================
# Fix 6: __setitem__ raises clear error
# =========================================================================


class TestFix6SetitemError:
    """x[0] = 5 raises TypeError with 'immutable' message."""

    def test_setitem_raises_type_error(self):
        from flopscope._remote_array import RemoteArray

        arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        with pytest.raises(TypeError, match="immutable"):
            arr[0] = 5

    def test_setitem_slice_raises_type_error(self):
        from flopscope._remote_array import RemoteArray

        arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        with pytest.raises(TypeError, match="immutable"):
            arr[0:2] = [1, 2]

    def test_setitem_error_is_actionable(self):
        """The immutable error must tell participants HOW to fix it.

        A sandboxed participant only ever sees this as a traceback, so the
        message must carry problem + cause + fix + docs link (the Elm/Rust
        three-tier model) rather than a dead end. Guards against regressing
        back to a bare "cannot assign" message.
        """
        from flopscope._remote_array import RemoteArray

        arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        with pytest.raises(TypeError) as exc:
            arr[0] = 5
        msg = str(exc.value)
        assert "immutable" in msg  # names the cause
        assert "fnp.stack" in msg  # points to the functional alternative
        assert "http" in msg  # links to the docs


# =========================================================================
# Fix 8: ZMQ socket reset after timeout
# =========================================================================


class TestFix8ZmqSocketReset:
    """After a timeout, the socket is reset so the next call can succeed."""

    def test_reset_socket_clears_socket(self):
        from flopscope._connection import Connection

        conn = Connection(url="tcp://127.0.0.1:19999", timeout_ms=100)
        # Manually set a mock socket
        mock_sock = MagicMock()
        conn._socket = mock_sock
        conn._reset_socket()
        assert conn._socket is None
        mock_sock.close.assert_called_once_with(linger=0)

    def test_send_recv_resets_on_timeout(self):
        from flopscope._connection import Connection

        from flopscope.errors import FlopscopeServerError

        conn = Connection(url="tcp://127.0.0.1:19999", timeout_ms=100)
        mock_sock = MagicMock()
        mock_sock.send.side_effect = zmq.Again("timeout")
        conn._socket = mock_sock
        conn._handshake_done = True  # skip handshake; exercise the send/recv reset path

        # send_recv flushes pending frees FIRST; a handle GC-enqueued by an earlier
        # test would make the flush send on this mock (zmq.Again) and reset the
        # socket, surfacing a handshake ConnectionError instead of the expected
        # FlopscopeServerError. Force any deferred finalizer + drain so the queue is
        # empty and the mock's zmq.Again hits the real op, not the free flush.
        import gc

        from flopscope import _handles

        gc.collect()
        _handles.drain_pending()
        with pytest.raises(FlopscopeServerError, match="timeout"):
            conn.send_recv(b"test")
        # Socket should be reset
        assert conn._socket is None

    def test_send_recv_recv_timeout_resets(self):
        from flopscope._connection import Connection

        from flopscope.errors import FlopscopeServerError

        conn = Connection(url="tcp://127.0.0.1:19999", timeout_ms=100)
        mock_sock = MagicMock()
        mock_sock.send.return_value = None
        mock_sock.recv.side_effect = zmq.Again("timeout")
        conn._socket = mock_sock
        conn._handshake_done = True  # skip handshake; exercise the send/recv reset path

        # See test_send_recv_resets_on_timeout: drain any GC-enqueued frees so the
        # send_recv flush is a no-op and the mock's zmq.Again hits the real op.
        import gc

        from flopscope import _handles

        gc.collect()
        _handles.drain_pending()
        with pytest.raises(FlopscopeServerError, match="timeout"):
            conn.send_recv(b"test")
        assert conn._socket is None
