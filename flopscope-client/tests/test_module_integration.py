"""Client Module integration tests — require a live subprocess server."""

import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_REAL_SRC = str(_ROOT / "src")
_SERVER_SRC = str(_ROOT / "flopscope-server" / "src")
_VENV_PYTHON = sys.executable
_SERVER_URL = "tcp://127.0.0.1:15572"

# When running from the client venv (no numpy), supplement PYTHONPATH with the
# root venv's site-packages so the server subprocess can import numpy.
_root_sp = next(
    iter(glob.glob(str(_ROOT / ".venv" / "lib" / "python*" / "site-packages"))),
    "",
)
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        p for p in [_root_sp, os.environ.get("PYTHONPATH", "")] if p
    ),
}

_SERVER_SCRIPT = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="module", autouse=True)
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_SUBPROCESS_ENV,
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"server failed: {line}{proc.stderr.read()}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    from flopscope._budget import _reset_global_default

    reset_connection()
    _reset_global_default()
    yield
    reset_connection()
    _reset_global_default()


def test_mlp_save_from_file_roundtrip(tmp_path):
    import flopscope as we

    class Linear(we.Module):
        def __init__(self, i, o):
            self.W = we.array([[0.0] * i for _ in range(o)])
            self.b = we.array([0.0] * o)

    class MLP(we.Module):
        def __init__(self, sizes):
            self.sizes = list(sizes)
            self.layers = [Linear(a, b) for a, b in zip(sizes, sizes[1:], strict=False)]

        def config(self):
            return {"sizes": self.sizes}

    with we.BudgetContext(flop_budget=10_000_000):
        m = MLP([3, 2])
        m.layers[0].W = we.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m.save(str(tmp_path / "mlp.npz"))
        m2 = MLP.from_file(str(tmp_path / "mlp.npz"))
        assert m2.sizes == [3, 2]
        assert m2.layers[0].W.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_state_dict_keys_and_underscore_excluded():
    import flopscope as we

    class Net(we.Module):
        def __init__(self):
            self.w = we.array([1.0, 2.0])
            self._scratch = we.array([9.0])

    with we.BudgetContext(flop_budget=1_000_000):
        keys = set(Net().state_dict())
        assert keys == {"w"}
