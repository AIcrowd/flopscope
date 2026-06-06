"""Transport tests for exact symmetry metadata across the client/server boundary."""

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
_client_perm_group = _load_client_module(
    "flopscope/_perm_group.py", "flopscope._perm_group"
)
_load_client_module("flopscope/_dispatch.py", "flopscope._dispatch")
_client_remote_array = _load_client_module(
    "flopscope/_remote_array.py", "flopscope._remote_array"
)

RemoteArray = _client_remote_array.RemoteArray
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


def _make_symmetric_tensor():
    data = np.array([[1.0, 2.0], [2.0, 3.0]])
    return flops.as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))


def test_pack_result_emits_symmetry(handler_session):
    _, handler = handler_session

    packed = handler._pack_result(_make_symmetric_tensor())

    assert packed["status"] == "ok"
    assert packed["result"]["symmetry"] == {"axes": [0, 1], "generators": [[1, 0]]}
    assert "symmetry_info" not in packed["result"]


def test_result_from_response_exposes_symmetry():
    payload = {"axes": [0, 1], "generators": [[1, 0]]}

    out = _result_from_response(
        {
            "status": "ok",
            "result": {
                "id": "a5",
                "shape": [2, 2],
                "dtype": "float64",
                "symmetry": payload,
            },
        }
    )

    assert isinstance(out, RemoteArray)
    assert out.symmetry == ClientSymmetryGroup.from_payload(payload)
    assert not hasattr(out, "symmetry_info")


def test_remote_symmetric_result_keeps_symmetry_through_second_operation(
    handler_session,
):
    session, handler = handler_session

    remote = _result_from_response(handler._pack_result(_make_symmetric_tensor()))
    first = _result_from_response(
        handler.handle({"op": "add", "args": [remote.handle_id, 1.0], "kwargs": {}})
    )
    second = _result_from_response(
        handler.handle({"op": "add", "args": [first.handle_id, 1.0], "kwargs": {}})
    )

    payload = {"axes": [0, 1], "generators": [[1, 0]]}
    assert first.symmetry == ClientSymmetryGroup.from_payload(payload)
    assert second.symmetry == ClientSymmetryGroup.from_payload(payload)
    assert session.get_array(second.handle_id).symmetry.to_payload() == payload
