"""Native flopscope-server subprocess for the client-parity harness."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

# tests/client_compat/ -> repo root is two levels up.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_ROOT, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_ROOT, "flopscope-server", "src")
_REAL_SRC = os.path.join(_ROOT, "src")
_SERVER_VENV = os.path.join(_ROOT, "flopscope-server", ".venv", "bin", "python")
_ROOT_VENV = os.path.join(_ROOT, ".venv", "bin", "python")
_VENV_PYTHON = _SERVER_VENV if os.path.exists(_SERVER_VENV) else _ROOT_VENV

# Base port for the harness; distinct from test_full_integration (15560).
# When running under pytest-xdist, each worker (gw0, gw1, ...) gets its own
# port offset so multiple workers don't collide on the same TCP address.
_BASE_PORT = 15571

#: First line a launched server prints on stdout when its own pre-bind probe
#: (see ``start_server``) finds the target port already taken. Distinct from
#: ``SERVER_READY`` so ``start_server`` can tell "failed to start" apart from
#: "started but bound to a port someone else already holds".
_BIND_FAILED_MARKER = "BIND_FAILED"


class ServerPortInUseError(RuntimeError):
    """``start_server`` could not claim its target port.

    Raised instead of silently handing back a subprocess handle for a server
    that never actually bound: without this check, a launch racing a
    leftover process on the same port (a leaked server, or a socket still in
    Linux's longer TIME_WAIT window) would return successfully while the
    caller's client ends up talking to whatever else is listening there.
    """


def _worker_port(base_port: int) -> int:
    """Return a per-xdist-worker port (*base_port* + worker index, or base)."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id.startswith("gw"):
        try:
            return base_port + int(worker_id[2:])
        except ValueError:
            pass
    return base_port


def ensure_client_on_path() -> None:
    """Put the CLIENT flopscope first on sys.path (before native src/)."""
    if _CLIENT_SRC not in sys.path:
        sys.path.insert(0, _CLIENT_SRC)


def start_server(base_port: int = _BASE_PORT) -> subprocess.Popen:
    """Launch a flopscope-server subprocess bound to *base_port* (+ xdist offset).

    Raises ``ServerPortInUseError`` if the target port is already held by
    another process. The launched script bind-tests the port itself, on the
    same host/port ``FlopscopeServer.run()`` will bind, and only signals
    ``SERVER_READY`` once that probe succeeds — a bind failure inside
    ``run()`` itself happens deep inside a blocking call this process can't
    observe directly, so the readiness line would otherwise go out
    regardless of whether the real bind ever succeeds.
    """
    port = _worker_port(base_port)
    server_url = f"tcp://127.0.0.1:{port}"
    os.environ["FLOPSCOPE_SERVER_URL"] = server_url

    script = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
import zmq
from flopscope_server._server import FlopscopeServer

# Bind-test the exact address ourselves before signalling readiness: the
# real bind happens inside FlopscopeServer.run(), which blocks in its recv
# loop on success, so there is no other point at which this process could
# observe a bind failure and report it.
_probe_ctx = zmq.Context()
_probe_sock = _probe_ctx.socket(zmq.REP)
try:
    _probe_sock.bind({server_url!r})
except zmq.error.ZMQError as exc:
    print("{_BIND_FAILED_MARKER}:{port}:" + str(exc), flush=True)
    sys.exit(1)
finally:
    _probe_sock.close(linger=0)
    _probe_ctx.term()

print("SERVER_READY", flush=True)
FlopscopeServer(url={server_url!r}).run()
"""
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline() if proc.stdout else ""
    if line.startswith(_BIND_FAILED_MARKER):
        proc.wait(timeout=5)
        err = proc.stderr.read() if proc.stderr else ""
        raise ServerPortInUseError(
            f"cannot start flopscope-server on port {port} ({server_url}): "
            f"another process already holds it ({line.strip()!r}). "
            f"{err[:500]}".strip()
        )
    if "SERVER_READY" not in line:
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"flopscope-server failed to start: {line!r} / {err[:500]}")
    time.sleep(0.3)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
