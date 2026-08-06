"""Tests for flopscope_server._protocol module.

TDD approach: these tests were written first, then the module was implemented.
"""

import msgpack
import numpy as np
import pytest
from flopscope_server._protocol import (
    InvalidRequestError,
    decode_request,
    encode_error_response,
    encode_fetch_response,
    encode_response,
    validate_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack(obj) -> bytes:
    """Encode a dict to msgpack bytes (simulating a client message)."""
    return msgpack.packb(obj, use_bin_type=True)


# ---------------------------------------------------------------------------
# decode_request
# ---------------------------------------------------------------------------


class TestDecodeRequest:
    def test_valid_op_request(self):
        raw = _pack({"op": "abs", "args": [[1, -2, 3]]})
        msg = decode_request(raw)
        assert msg["op"] == "abs"
        assert "args" in msg

    def test_valid_create_from_data(self):
        raw = _pack(
            {
                "op": "create_from_data",
                "data": b"\x00\x01\x02",
                "dtype": "float32",
                "shape": [3],
            }
        )
        msg = decode_request(raw)
        assert msg["op"] == "create_from_data"
        assert msg["dtype"] == "float32"
        assert msg["data"] == b"\x00\x01\x02"

    def test_valid_budget_open(self):
        raw = _pack({"op": "budget_open", "budget": 1000})
        msg = decode_request(raw)
        assert msg["op"] == "budget_open"
        assert msg["budget"] == 1000

    def test_invalid_msgpack_raises(self):
        with pytest.raises(InvalidRequestError, match="malformed"):
            decode_request(b"\xff\xfe\xfd invalid bytes")

    def test_missing_op_field_raises(self):
        raw = _pack({"args": [1, 2, 3]})
        with pytest.raises(InvalidRequestError, match="op"):
            decode_request(raw)

    def test_empty_bytes_raises(self):
        with pytest.raises(InvalidRequestError):
            decode_request(b"")

    def test_bytes_keys_normalized_to_strings(self):
        """Ensure all top-level keys are decoded as strings, not bytes."""
        raw = _pack({"op": "fetch", "handle": 42})
        msg = decode_request(raw)
        for key in msg.keys():
            assert isinstance(key, str), f"Key {key!r} should be a string"

    def test_op_value_is_string(self):
        raw = _pack({"op": "free", "handle": 7})
        msg = decode_request(raw)
        assert isinstance(msg["op"], str)

    def test_request_id_decoded_as_string(self):
        raw = _pack({"op": "abs", "request_id": "req-123"})
        msg = decode_request(raw)
        assert msg["request_id"] == "req-123"
        assert isinstance(msg["request_id"], str)

    def test_binary_data_field_stays_bytes(self):
        """Binary 'data' payloads must NOT be decoded to str."""
        payload = b"\x00\x01\x02\x03"
        raw = _pack(
            {
                "op": "create_from_data",
                "data": payload,
                "dtype": "float32",
                "shape": [1],
            }
        )
        msg = decode_request(raw)
        assert isinstance(msg["data"], bytes)
        assert msg["data"] == payload


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------


class TestValidateRequest:
    def test_known_registry_op_passes(self):
        # 'abs' is in REGISTRY
        validate_request({"op": "abs"})

    def test_another_registry_op_passes(self):
        validate_request({"op": "matmul"})

    def test_budget_open_passes(self):
        validate_request({"op": "budget_open"})

    def test_budget_close_passes(self):
        validate_request({"op": "budget_close"})

    def test_budget_status_passes(self):
        validate_request({"op": "budget_status"})

    def test_budget_summary_is_whitelisted(self):
        from flopscope_server._protocol import WHITELIST

        assert "budget_summary" in WHITELIST

    def test_fetch_passes(self):
        validate_request({"op": "fetch"})

    def test_fetch_slice_passes(self):
        validate_request({"op": "fetch_slice"})

    def test_free_passes(self):
        validate_request({"op": "free"})

    def test_create_from_data_passes(self):
        validate_request({"op": "create_from_data"})

    def test_unknown_op_raises(self):
        with pytest.raises(InvalidRequestError, match="unknown op"):
            validate_request({"op": "not_a_real_op_xyz"})

    def test_another_unknown_op_raises(self):
        with pytest.raises(InvalidRequestError):
            validate_request({"op": "execute_shell_command"})


# ---------------------------------------------------------------------------
# encode_response
# ---------------------------------------------------------------------------


class TestEncodeResponse:
    def test_returns_bytes(self):
        result = encode_response(42.0, budget=500, comms_overhead_ns=1000)
        assert isinstance(result, bytes)

    def test_decodable(self):
        result = encode_response(3.14, budget=100, comms_overhead_ns=500)
        msg = msgpack.unpackb(result, raw=False)
        assert isinstance(msg, dict)

    def test_contains_status_ok(self):
        result = encode_response("hello", budget=0, comms_overhead_ns=0)
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("status") == "ok"

    def test_contains_result(self):
        result = encode_response(99, budget=1000, comms_overhead_ns=200)
        msg = msgpack.unpackb(result, raw=False)
        assert "result" in msg
        assert msg["result"] == 99

    def test_contains_budget(self):
        result = encode_response(0, budget=777, comms_overhead_ns=300)
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("budget") == 777

    def test_contains_comms_overhead_ns(self):
        result = encode_response(0, budget=0, comms_overhead_ns=12345)
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("comms_overhead_ns") == 12345


# ---------------------------------------------------------------------------
# encode_error_response
# ---------------------------------------------------------------------------


class TestEncodeErrorResponse:
    def test_returns_bytes(self):
        result = encode_error_response("InvalidRequestError", "bad op")
        assert isinstance(result, bytes)

    def test_decodable(self):
        result = encode_error_response("ValueError", "something went wrong")
        msg = msgpack.unpackb(result, raw=False)
        assert isinstance(msg, dict)

    def test_contains_status_error(self):
        result = encode_error_response("TypeError", "bad type")
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("status") == "error"

    def test_contains_error_type(self):
        result = encode_error_response("BudgetExceededError", "over budget")
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("error_type") == "BudgetExceededError"

    def test_contains_message(self):
        result = encode_error_response("InvalidRequestError", "missing op field")
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("message") == "missing op field"


# ---------------------------------------------------------------------------
# encode_fetch_response
# ---------------------------------------------------------------------------


class TestEncodeFetchResponse:
    def test_returns_bytes(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        assert isinstance(result, bytes)

    def test_decodable(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=False)
        assert isinstance(msg, dict)

    def test_contains_status_ok(self):
        arr = np.array([1.0], dtype=np.float64)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("status") == "ok"

    def test_contains_raw_data_bytes(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        raw_bytes = arr.tobytes()
        result = encode_fetch_response(
            raw_bytes, arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=True)
        # data field should be raw bytes
        assert b"data" in msg
        assert msg[b"data"] == raw_bytes

    def test_contains_shape(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=False)
        assert list(msg["shape"]) == [3, 4]

    def test_contains_dtype(self):
        arr = np.array([1, 2], dtype=np.float64)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("dtype") == "float64"

    def test_contains_comms_overhead_ns(self):
        arr = np.array([0.0], dtype=np.float32)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=9876
        )
        msg = msgpack.unpackb(result, raw=False)
        assert msg.get("comms_overhead_ns") == 9876

    def test_roundtrip_float32(self):
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        result = encode_fetch_response(
            arr.tobytes(), arr.shape, str(arr.dtype), comms_overhead_ns=0
        )
        msg = msgpack.unpackb(result, raw=True)
        recovered = np.frombuffer(msg[b"data"], dtype=np.float32)
        np.testing.assert_array_equal(recovered, arr)
