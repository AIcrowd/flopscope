"""Regression coverage for the parity harness's server-launch safety.

This bug is invisible on a developer machine: a fresh checkout never has a
leftover server sitting on the harness's port, so nothing here is caught by
"it works locally". The failure only shows up when something else already
holds the port at launch time (a leaked server from a prior suite, or a
socket still lingering in Linux's longer TIME_WAIT window) — which is exactly
what these tests simulate directly, without needing to actually race another
test suite.
"""

from __future__ import annotations

import socket

import pytest

from tests.client_compat._server_fixture import (
    ServerPortInUseError,
    _worker_port,
    start_server,
)
from tests.parity.runner import _PARITY_BASE_PORT


def test_start_server_raises_instead_of_adopting_a_foreign_listener():
    """Occupy the exact port the parity harness would use, then confirm
    ``start_server`` fails loudly rather than handing back a subprocess that
    is not actually the one listening on that port.

    A plain TCP listener is enough to reproduce the failure class: the
    launched server's own pre-bind probe (see ``start_server``) hits the same
    "address already in use" condition regardless of whether the real
    occupant is a raw socket or another flopscope-server with an open
    session.
    """
    port = _worker_port(_PARITY_BASE_PORT)
    foreign = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    foreign.bind(("127.0.0.1", port))
    foreign.listen(1)
    try:
        with pytest.raises(ServerPortInUseError) as exc_info:
            start_server(_PARITY_BASE_PORT)
        message = str(exc_info.value)
        assert str(port) in message
        assert "already holds it" in message
    finally:
        foreign.close()


def test_parity_and_client_compat_default_to_different_ports():
    # The two harnesses import the SAME launch function; this pins the one
    # fact that keeps them from colliding when run back-to-back in the same
    # job — that they are configured with different port bases — so a future
    # edit can't quietly re-merge the ranges.
    from tests.client_compat._server_fixture import _BASE_PORT

    assert _PARITY_BASE_PORT != _BASE_PORT
