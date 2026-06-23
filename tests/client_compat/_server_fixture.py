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


def _worker_port() -> int:
    """Return a per-xdist-worker port (base + worker index, or base if not xdist)."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id.startswith("gw"):
        try:
            return _BASE_PORT + int(worker_id[2:])
        except ValueError:
            pass
    return _BASE_PORT


def _make_server_url() -> str:
    return f"tcp://127.0.0.1:{_worker_port()}"


def ensure_client_on_path() -> None:
    """Put the CLIENT flopscope first on sys.path (before native src/)."""
    if _CLIENT_SRC not in sys.path:
        sys.path.insert(0, _CLIENT_SRC)


def start_server() -> subprocess.Popen:
    server_url = _make_server_url()
    os.environ["FLOPSCOPE_SERVER_URL"] = server_url

    script = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
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
