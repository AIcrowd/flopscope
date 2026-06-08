"""Wire-protocol round-trip tests.

These tests manually construct msgpack bytes and push them through the
server's decode/normalize pipeline (and vice-versa for responses),
verifying that every key and value has the expected Python type at each
stage.  No ZMQ sockets or live server needed.

Because the ``flopscope`` namespace is split across three trees
(``src/``, ``flopscope-client/src/``, ``flopscope-server/src/``), we
use importlib to load specific files without relying on normal
``import`` resolution.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import types

import msgpack
import pytest

# This wire-protocol test loads flopscope_server in-process, which transitively
# imports the numpy-backed core. Skip entirely in the numpy-free client venv.
pytest.importorskip("numpy")

# =====================================================================
# Module loading helpers — avoids namespace collision between
# flopscope (library) and flopscope (client) and flopscope_server.
# =====================================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The server modules loaded below do `from flopscope_server... import ...`; put
# the server's src on sys.path so those intra-package imports resolve.
sys.path.insert(0, os.path.join(_ROOT, "flopscope-server", "src"))


def _load_module(name: str, path: str) -> types.ModuleType:
    """Load a single Python file as a module, by absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # needed so intra-module imports succeed
    spec.loader.exec_module(mod)
    return mod


# ── Client protocol ─────────────────────────────────────────────────
_client_proto = _load_module(
    "flopscope._protocol",
    os.path.join(_ROOT, "flopscope-client", "src", "flopscope", "_protocol.py"),
)
encode_request = _client_proto.encode_request
encode_create_from_data = _client_proto.encode_create_from_data
client_decode_response = _client_proto.decode_response
client_normalize = _client_proto._normalize

# ── Library registry (needed by server protocol) ────────────────────
# Ensure the *library* flopscope._registry is importable before we
# load the server protocol (which does ``from flopscope._registry import REGISTRY``).
_lib_registry = _load_module(
    "flopscope._registry",
    os.path.join(_ROOT, "src", "flopscope", "_registry.py"),
)

# ── Server protocol ─────────────────────────────────────────────────
_server_proto = _load_module(
    "flopscope_server._protocol",
    os.path.join(_ROOT, "flopscope-server", "src", "flopscope_server", "_protocol.py"),
)
server_decode_request = _server_proto.decode_request
server_encode_response = _server_proto.encode_response
encode_fetch_response = _server_proto.encode_fetch_response

# ── Server normalisation helpers ────────────────────────────────────
_server_mod = _load_module(
    "flopscope_server._server",
    os.path.join(_ROOT, "flopscope-server", "src", "flopscope_server", "_server.py"),
)
_normalize_msg = _server_mod._normalize_msg
_normalize_arg = _server_mod._normalize_arg


# =====================================================================
# Helpers
# =====================================================================


def _server_pipeline(raw: bytes) -> dict:
    """Run client bytes through the full server decode + normalize."""
    msg = server_decode_request(raw)
    _normalize_msg(msg)
    return msg


def _assert_all_str_keys(d: dict, path: str = "root") -> None:
    """Recursively verify every dict key in *d* is a str."""
    for k, v in d.items():
        assert isinstance(k, str), f"bytes key {k!r} at {path}"
        if isinstance(v, dict):
            _assert_all_str_keys(v, f"{path}.{k}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _assert_all_str_keys(item, f"{path}.{k}[{i}]")


# =====================================================================
# 1. Simple op round-trip (add)
# =====================================================================


class TestSimpleOpAdd:
    """Encode an 'add' request with two handle-dict args, decode on
    the server side, and verify all keys are str and handle IDs are str.
    """

    def test_keys_and_handles_are_strings(self):
        raw = encode_request(
            "add",
            args=[{"__handle__": "a0"}, {"__handle__": "a1"}],
        )
        msg = _server_pipeline(raw)

        # Top-level keys
        assert isinstance(msg["op"], str)
        assert msg["op"] == "add"

        # Args are a list of dicts with string keys
        args = msg["args"]
        assert isinstance(args, list) and len(args) == 2

        for i, arg in enumerate(args):
            assert isinstance(arg, dict), f"arg[{i}] is not dict"
            _assert_all_str_keys(arg, f"args[{i}]")
            assert "__handle__" in arg, f"arg[{i}] missing __handle__"
            assert isinstance(arg["__handle__"], str), f"arg[{i}] handle not str"

    def test_handle_values(self):
        raw = encode_request(
            "add",
            args=[{"__handle__": "a0"}, {"__handle__": "a1"}],
        )
        msg = _server_pipeline(raw)
        assert msg["args"][0]["__handle__"] == "a0"
        assert msg["args"][1]["__handle__"] == "a1"


# =====================================================================
# 2. create_from_data with small binary payload
# =====================================================================


class TestCreateFromDataBinary:
    """The data field (8 bytes of float64 for 1.0) MUST survive as
    bytes through the full pipeline — no spurious decode to str.
    """

    def test_data_stays_bytes(self):
        payload = struct.pack("<d", 1.0)  # 8 bytes, little-endian float64
        assert len(payload) == 8

        raw = encode_create_from_data(payload, [1], "float64")
        msg = _server_pipeline(raw)

        assert msg["op"] == "create_from_data"
        data_arg = msg["args"][0]
        # Payload contains bytes > 127, so normalize should NOT decode it.
        assert isinstance(data_arg, bytes), (
            f"data was converted to {type(data_arg).__name__}: {data_arg!r}"
        )
        assert data_arg == payload


# =====================================================================
# 3. create_from_data with ASCII-looking payload
# =====================================================================


class TestCreateFromDataAsciiLooking:
    """b"ABCDEFGH" is 8 bytes, all printable ASCII, <=32 bytes long.
    The server's _normalize_arg WILL decode it to str because it passes
    the heuristic.  This test documents the current behavior.
    """

    def test_ascii_payload_is_decoded_to_str(self):
        """Short all-ASCII bytes get decoded — this is the CURRENT behavior.

        For create_from_data the server never uses args[0] as a dict key
        or compares it against anything, so this decode is harmless.  But
        callers that rely on receiving bytes may be surprised.
        """
        payload = b"ABCDEFGH"
        raw = encode_create_from_data(payload, [1], "float64")
        msg = _server_pipeline(raw)
        data_arg = msg["args"][0]

        # Current behavior: short ASCII bytes get decoded to str by _normalize_arg.
        # This IS a potential concern if downstream code does isinstance(data, bytes).
        if isinstance(data_arg, str):
            # Documenting: the heuristic decoded our "binary" data to str.
            assert data_arg == "ABCDEFGH"
        else:
            # If this passes, the heuristic was tightened — even better.
            assert isinstance(data_arg, bytes)

    def test_large_ascii_payload_stays_bytes(self):
        """Payloads > 32 bytes always stay as bytes, even if all ASCII."""
        payload = b"A" * 64
        raw = encode_create_from_data(payload, [8], "float64")
        msg = _server_pipeline(raw)
        data_arg = msg["args"][0]
        assert isinstance(data_arg, bytes), (
            f"large payload was decoded to {type(data_arg).__name__}"
        )


# =====================================================================
# 4. kwargs with integer values
# =====================================================================


class TestKwargsAxisInteger:
    """Encode kwargs={"axis": 0} and verify it arrives as int 0."""

    def test_axis_is_int(self):
        raw = encode_request(
            "sum",
            args=[{"__handle__": "a0"}],
            kwargs={"axis": 0},
        )
        msg = _server_pipeline(raw)

        assert msg["op"] == "sum"
        assert "kwargs" in msg
        kwargs = msg["kwargs"]
        assert isinstance(kwargs, dict)
        _assert_all_str_keys(kwargs, "kwargs")
        assert kwargs["axis"] == 0
        assert isinstance(kwargs["axis"], int)

    def test_kwargs_handle_value(self):
        """A handle dict inside kwargs should normalize properly."""
        raw = encode_request(
            "some_op",
            args=[],
            kwargs={"out": {"__handle__": "a5"}},
        )
        msg = _server_pipeline(raw)
        out_val = msg["kwargs"]["out"]
        assert isinstance(out_val, dict)
        assert "__handle__" in out_val
        assert isinstance(out_val["__handle__"], str)
        assert out_val["__handle__"] == "a5"


# =====================================================================
# 5. Fetch response round-trip (server encode -> client decode)
# =====================================================================


class TestFetchResponseRoundTrip:
    """Server encodes a fetch response with 16 bytes of float64 data.
    Client decodes it.  The data field must remain bytes.
    """

    def test_data_stays_bytes(self):
        payload = struct.pack("<dd", 1.0, 2.0)  # 16 bytes
        raw = encode_fetch_response(
            data=payload,
            shape=[2],
            dtype="float64",
            comms_overhead_ns=0,
        )
        resp = client_decode_response(raw)

        assert resp["status"] == "ok"
        assert isinstance(resp["data"], bytes), (
            f"data decoded to {type(resp['data']).__name__}"
        )
        assert resp["data"] == payload
        assert resp["dtype"] == "float64"
        assert resp["shape"] == [2]

    def test_small_data_stays_bytes(self):
        """Even 8 bytes (single float64) with high bytes must stay bytes."""
        payload = struct.pack("<d", 3.14)
        raw = encode_fetch_response(
            data=payload,
            shape=[1],
            dtype="float64",
            comms_overhead_ns=0,
        )
        resp = client_decode_response(raw)
        assert isinstance(resp["data"], bytes)


# =====================================================================
# 6. __getitem__ with slice
# =====================================================================


class TestGetitemSlice:
    """Encode __getitem__ with a __slice__ dict and verify decoding."""

    def test_slice_dict_keys_are_strings(self):
        raw = encode_request(
            "__getitem__",
            args=[
                {"__handle__": "a0"},
                {"__slice__": [0, 5, None]},
            ],
        )
        msg = _server_pipeline(raw)

        assert msg["op"] == "__getitem__"
        slice_arg = msg["args"][1]
        assert isinstance(slice_arg, dict)
        _assert_all_str_keys(slice_arg, "args[1]")

        # Verify __slice__ survived normalization
        assert "__slice__" in slice_arg
        parts = slice_arg["__slice__"]
        assert parts[0] == 0
        assert parts[1] == 5
        assert parts[2] is None

    def test_slice_with_negative_step(self):
        raw = encode_request(
            "__getitem__",
            args=[
                {"__handle__": "a0"},
                {"__slice__": [None, None, -1]},
            ],
        )
        msg = _server_pipeline(raw)
        parts = msg["args"][1]["__slice__"]
        assert parts[0] is None
        assert parts[1] is None
        assert parts[2] == -1


# =====================================================================
# 7. __getitem__ with fancy index handle
# =====================================================================


class TestGetitemFancyIndex:
    """Both args are handle dicts — verify both normalize correctly."""

    def test_both_handles_normalized(self):
        raw = encode_request(
            "__getitem__",
            args=[
                {"__handle__": "a0"},
                {"__handle__": "a1"},
            ],
        )
        msg = _server_pipeline(raw)

        for i in range(2):
            arg = msg["args"][i]
            assert isinstance(arg, dict)
            assert "__handle__" in arg
            assert isinstance(arg["__handle__"], str)


# =====================================================================
# 8. Handle ID that looks like a number ("a100")
# =====================================================================


class TestHandleIdLooksLikeNumber:
    """Handle 'a100' must remain a string through the full pipeline."""

    def test_a100_stays_str(self):
        raw = encode_request(
            "add",
            args=[{"__handle__": "a100"}, {"__handle__": "a999"}],
        )
        msg = _server_pipeline(raw)

        for i, expected in enumerate(["a100", "a999"]):
            handle = msg["args"][i]["__handle__"]
            assert isinstance(handle, str), (
                f"handle {expected!r} became {type(handle).__name__}"
            )
            assert handle == expected

    def test_numeric_looking_handle(self):
        """What about a handle that is purely numeric, e.g. '123'?"""
        raw = encode_request(
            "add",
            args=[{"__handle__": "123"}],
        )
        msg = _server_pipeline(raw)
        handle = msg["args"][0]["__handle__"]
        # msgpack encodes "123" as a bytes string, normalize_arg should
        # decode it back to str since it's short ASCII.
        assert isinstance(handle, str)
        assert handle == "123"


# =====================================================================
# 9. Empty kwargs
# =====================================================================


class TestEmptyKwargs:
    """Server must tolerate both {} and None for kwargs."""

    def test_empty_dict_kwargs(self):
        raw = encode_request("add", args=[{"__handle__": "a0"}], kwargs={})
        msg = _server_pipeline(raw)
        assert msg["kwargs"] == {}

    def test_none_kwargs(self):
        raw = encode_request("add", args=[{"__handle__": "a0"}], kwargs=None)
        msg = _server_pipeline(raw)
        assert msg["kwargs"] is None

    def test_missing_kwargs(self):
        """If client sends no kwargs key at all, normalize should not crash."""
        # Manually build a msgpack payload without kwargs
        raw = msgpack.packb(
            {"op": "add", "args": [{"__handle__": "a0"}]},
            use_bin_type=True,
        )
        msg = _server_pipeline(raw)
        assert msg["op"] == "add"
        # kwargs should simply be absent
        assert "kwargs" not in msg


# =====================================================================
# 10. Stress: 100 handle-dict args
# =====================================================================


class TestStress100Handles:
    """Encode 100 handle dicts, verify all normalize to str keys/values."""

    def test_all_handles_normalized(self):
        handles = [{"__handle__": f"a{i}"} for i in range(100)]
        raw = encode_request("concatenate", args=[handles])
        msg = _server_pipeline(raw)

        # args is a list with one element (the list of handles)
        inner_list = msg["args"][0]
        assert isinstance(inner_list, list)
        assert len(inner_list) == 100

        for i, item in enumerate(inner_list):
            assert isinstance(item, dict), f"item[{i}] is {type(item).__name__}"
            assert "__handle__" in item, f"item[{i}] missing __handle__"
            handle = item["__handle__"]
            assert isinstance(handle, str), (
                f"item[{i}].__handle__ is {type(handle).__name__}: {handle!r}"
            )
            assert handle == f"a{i}"


# =====================================================================
# 11. Server encode_response -> client decode round-trip
# =====================================================================


class TestServerResponseRoundTrip:
    """Verify a normal 'ok' response round-trips through server encode
    and client decode without type confusion.
    """

    def test_ok_response(self):
        result = {"handle": "a42", "shape": [3, 4], "dtype": "float32"}
        raw = server_encode_response(result, budget=1000, comms_overhead_ns=50)
        resp = client_decode_response(raw)

        assert resp["status"] == "ok"
        assert resp["budget"] == 1000
        res = resp["result"]
        assert isinstance(res, dict)
        _assert_all_str_keys(res, "result")
        assert res["handle"] == "a42"
        assert isinstance(res["handle"], str)


# =====================================================================
# 12. Bytes/string boundary: raw msgpack with raw=True keys
# =====================================================================


class TestRawMsgpackBytesKeys:
    """Directly test that server decode_request converts bytes keys to str
    for all top-level keys, not just _STRING_FIELDS.
    """

    def test_all_top_level_keys_are_str(self):
        raw = msgpack.packb(
            {"op": "add", "args": [], "kwargs": {}, "request_id": "r1"},
            use_bin_type=True,
        )
        msg = server_decode_request(raw)
        for k in msg:
            assert isinstance(k, str), f"top-level key {k!r} is bytes"

    def test_nested_kwargs_keys_are_str_after_normalize(self):
        raw = msgpack.packb(
            {
                "op": "sum",
                "args": [{"__handle__": "a0"}],
                "kwargs": {"axis": 0, "keepdims": True},
            },
            use_bin_type=True,
        )
        msg = server_decode_request(raw)
        _normalize_msg(msg)
        assert all(isinstance(k, str) for k in msg["kwargs"]), (
            f"kwargs keys: {list(msg['kwargs'].keys())}"
        )


# =====================================================================
# 13. Client-side _normalize edge cases
# =====================================================================


class TestClientNormalize:
    """Edge cases for the client-side _normalize function."""

    def test_bytes_with_high_bytes_stay_bytes(self):
        """Bytes containing values > 127 should NOT be decoded."""
        val = bytes([0x80, 0xFF, 0x00, 0x01])
        result = client_normalize(val)
        assert isinstance(result, bytes)

    def test_short_ascii_decoded(self):
        result = client_normalize(b"hello")
        assert isinstance(result, str)
        assert result == "hello"

    def test_long_ascii_stays_bytes(self):
        """Bytes > 32 chars stay as bytes even if all ASCII."""
        val = b"a" * 64
        result = client_normalize(val)
        assert isinstance(result, bytes)

    def test_tuple_preserved(self):
        val = (b"hello", 42)
        result = client_normalize(val)
        assert isinstance(result, tuple)
        assert result[0] == "hello"
        assert result[1] == 42

    def test_nested_dict(self):
        val = {b"key": {b"inner": b"val"}}
        result = client_normalize(val)
        assert "key" in result
        assert "inner" in result["key"]
        assert result["key"]["inner"] == "val"


# =====================================================================
# 14. Server _normalize_arg: the critical bytes->str heuristic
# =====================================================================


class TestServerNormalizeArgHeuristic:
    """Directly exercise _normalize_arg to map out the bytes->str boundary."""

    def test_short_ascii_decoded(self):
        assert _normalize_arg(b"a0") == "a0"
        assert isinstance(_normalize_arg(b"a0"), str)

    def test_long_stays_bytes(self):
        val = b"a" * 33
        assert isinstance(_normalize_arg(val), bytes)

    def test_exactly_32_decoded(self):
        val = b"a" * 32
        assert isinstance(_normalize_arg(val), str)

    def test_33_stays_bytes(self):
        val = b"a" * 33
        assert isinstance(_normalize_arg(val), bytes)

    def test_binary_data_8_bytes(self):
        """8 bytes of float64 for 1.0 contains 0x3F which is '?' — but
        also contains 0xF0 which is > 127.  Should stay bytes."""
        data = struct.pack("<d", 1.0)
        result = _normalize_arg(data)
        assert isinstance(result, bytes)

    def test_null_byte_stays_bytes(self):
        """Bytes containing null (0x00) should stay bytes."""
        data = b"\x00\x01\x02\x03"
        result = _normalize_arg(data)
        assert isinstance(result, bytes)

    def test_dict_keys_decoded(self):
        d = {b"__handle__": b"a0"}
        result = _normalize_arg(d)
        assert isinstance(result, dict)
        # _normalize_arg uses _decode_if_bytes for keys
        assert "__handle__" in result
        assert result["__handle__"] == "a0"
