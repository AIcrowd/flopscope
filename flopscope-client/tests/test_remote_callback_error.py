"""The client's local-callback ops raise RemoteCallbackError (not an opaque
msgpack TypeError) when handed a Python callable on the client/server backend.

These tests need no live server: the error fires at encode time, before any
connection is made.
"""

import pytest
from flopscope._protocol import encode_request
from flopscope._registry_data import LOCAL_CALLBACK_OPS

import flopscope
from flopscope.errors import RemoteCallbackError


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


@pytest.mark.parametrize("op", sorted(LOCAL_CALLBACK_OPS))
def test_callback_op_raises_remote_callback_error(op):
    fn = getattr(flopscope, op)
    with pytest.raises(RemoteCallbackError) as ei:
        fn(lambda *a, **k: 0.0)  # a Python callable can't cross the wire
    assert op in str(ei.value)
