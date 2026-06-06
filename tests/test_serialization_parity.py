"""Serialization round-trip tests: client encode -> server decode, without a network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent
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

    module_file = Path(_CLIENT_SRC) / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportArgumentType]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
    return module


_load_client_module("flopscope/_constants.py", "flopscope._constants")
_load_client_module("flopscope/_math_compat.py", "flopscope._math_compat")
_load_client_module("flopscope/_perm_group.py", "flopscope._perm_group")
_load_client_module("flopscope/_dispatch.py", "flopscope._dispatch")
_client_remote_array = _load_client_module(
    "flopscope/_remote_array.py", "flopscope._remote_array"
)
_client_perm_group = sys.modules["flopscope._perm_group"]

_encode_arg = _client_remote_array._encode_arg
_result_from_response = _client_remote_array._result_from_response
ClientSymmetryGroup = _client_perm_group.SymmetryGroup

import flopscope as flops


@pytest.fixture()
def handler_session():
    session = Session(flop_budget=10_000_000)
    handler = RequestHandler(session)
    yield session, handler
    if session.is_open:
        session.close()


class TestSymmetryGroupRoundTrip:
    def test_symmetry_group_symmetric(self, handler_session):
        group = ClientSymmetryGroup.symmetric(axes=(0, 1, 2))
        encoded = _encode_arg(group)

        assert encoded == {
            "__symmetry_group__": {
                "axes": [0, 1, 2],
                "generators": [[1, 0, 2], [0, 2, 1]],
            }
        }

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert isinstance(resolved, flops.SymmetryGroup)
        assert resolved.axes == (0, 1, 2)
        assert resolved.order() == 6

    def test_symmetry_group_in_list(self, handler_session):
        group = ClientSymmetryGroup.cyclic(axes=(0, 1, 2))
        encoded = _encode_arg([group, 42])

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert isinstance(resolved, list)
        assert isinstance(resolved[0], flops.SymmetryGroup)
        assert resolved[0].axes == (0, 1, 2)
        assert resolved[0].order() == 3
        assert resolved[1] == 42


class TestSymmetryTransportRoundTrip:
    def _make_symmetric_array(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        data = rng.random((3, 3))
        return (data + data.T) / 2.0

    def test_symmetric_tensor_pack_unpack(self, handler_session):
        data = self._make_symmetric_array()
        sym_tensor = flops.as_symmetric(
            data,
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        _, handler = handler_session
        packed = handler._pack_result(sym_tensor)

        assert packed["status"] == "ok"
        assert "symmetry" in packed["result"]
        assert "symmetry_info" not in packed["result"]

        remote_arr = _result_from_response(packed)
        assert remote_arr.symmetry == ClientSymmetryGroup.from_payload(
            {"axes": [0, 1], "generators": [[1, 0]]}
        )

    def test_plain_array_no_symmetry(self, handler_session):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        _, handler = handler_session
        packed = handler._pack_result(data)

        assert packed["status"] == "ok"
        assert "symmetry" not in packed["result"]

        remote_arr = _result_from_response(packed)
        assert remote_arr.symmetry is None

    def test_symmetric_tensor_shape_preserved(self, handler_session):
        rng = np.random.default_rng(7)
        raw = rng.random((4, 4))
        data = (raw + raw.T) / 2.0
        sym_tensor = flops.as_symmetric(
            data,
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        _, handler = handler_session
        packed = handler._pack_result(sym_tensor)
        remote_arr = _result_from_response(packed)

        assert remote_arr.shape == (4, 4)
        assert remote_arr.symmetry == ClientSymmetryGroup.from_payload(
            {"axes": [0, 1], "generators": [[1, 0]]}
        )

    def test_symmetric_tensor_multi_group(self, handler_session):
        rng = np.random.default_rng(99)
        raw = rng.random((3, 3, 2, 2))
        raw = (raw + raw.transpose(1, 0, 2, 3)) / 2.0
        raw = (raw + raw.transpose(0, 1, 3, 2)) / 2.0
        sym_tensor = flops.as_symmetric(
            raw,
            symmetry=flops.SymmetryGroup.direct_product(
                flops.SymmetryGroup.symmetric(axes=(0, 1)),
                flops.SymmetryGroup.symmetric(axes=(2, 3)),
            ),
        )

        _, handler = handler_session
        packed = handler._pack_result(sym_tensor)
        remote_arr = _result_from_response(packed)

        assert remote_arr.symmetry is not None
        assert remote_arr.symmetry.to_payload() == {
            "axes": [0, 1, 2, 3],
            "generators": [[1, 0, 2, 3], [0, 1, 3, 2]],
        }
