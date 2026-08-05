"""The client's local-callback ops raise RemoteCallbackError (not an opaque
msgpack TypeError) when handed a Python callable on the client/server backend.

These tests need no live server: the error fires at encode time, before any
connection is made.
"""

import time

import pytest
from flopscope._protocol import encode_request
from flopscope._registry_data import LOCAL_CALLBACK_OPS

import flopscope
from flopscope.errors import RemoteCallbackError, RemoteSerializationError


def test_local_callback_ops_set():
    # LOCAL_CALLBACK_OPS is generated from the server registry's
    # `local_callback` flag (scripts/sync_client.py) and content-checked
    # against it independently: CI's client-server-sync job runs
    # `sync_client.py --check`, and the root suite's
    # test_callback_op_registration.py::test_client_registry_data_is_in_sync
    # re-derives the expected set from the registry. A hand-copied literal
    # here would be a third copy of that list to keep in sync by hand --
    # exactly how `mask_indices` went missing before. Assert the structural
    # invariant this module actually relies on instead: every flagged op is a
    # callable reachable on the client's public surface.
    assert isinstance(LOCAL_CALLBACK_OPS, frozenset) and LOCAL_CALLBACK_OPS
    assert all(
        isinstance(op, str) and callable(getattr(flopscope, op, None))
        for op in LOCAL_CALLBACK_OPS
    )


def test_encode_request_rejects_callables():
    # Precondition: msgpack genuinely cannot serialize a Python function.
    with pytest.raises((TypeError, ValueError)):
        encode_request("apply_along_axis", args=[lambda x: x], kwargs={})


class _PropertyDtype:
    def __init__(self):
        self.calls = 0

    @property
    def _flopscope_dtype_name(self):
        self.calls += 1
        time.sleep(0.1)
        raise AssertionError("participant property executed")


class _NameDescriptor:
    def __get__(self, instance, owner):
        instance.calls += 1
        time.sleep(0.1)
        raise AssertionError("participant descriptor executed")


class _DescriptorDtype:
    name = _NameDescriptor()

    def __init__(self):
        self.calls = 0


@pytest.mark.parametrize("spec", [_PropertyDtype(), _DescriptorDtype()])
def test_dtype_descriptor_raises_remote_callback_without_execution(spec, monkeypatch):
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteCallbackError, match="dtype.*descriptor"):
        flopscope.array([1.0], dtype=spec)
    assert spec.calls == 0
    assert network_calls == 0


def test_inert_unsupported_dtype_like_value_remains_serialization_error():
    class InertDtype:
        pass

    with pytest.raises(RemoteSerializationError):
        flopscope.concatenate([InertDtype()])


@pytest.mark.parametrize("op", sorted(LOCAL_CALLBACK_OPS))
def test_callback_op_raises_remote_callback_error(op):
    fn = getattr(flopscope, op)
    with pytest.raises(RemoteCallbackError) as ei:
        fn(lambda *a, **k: 0.0)  # a Python callable can't cross the wire
    assert op in str(ei.value)
