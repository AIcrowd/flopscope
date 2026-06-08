"""Client I/O integration tests — require a live subprocess server."""

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
_SERVER_URL = "tcp://127.0.0.1:15571"

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


def test_savez_load_roundtrip(tmp_path):
    import flopscope as we

    with we.BudgetContext(flop_budget=1_000_000):
        a = we.array([[1.0, 2.0], [3.0, 4.0]])
        we.savez(str(tmp_path / "w.npz"), a=a, __meta__={"k": 1})
        out = we.load(str(tmp_path / "w.npz"))
        assert out["__meta__"] == {"k": 1}
        assert out["a"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_save_load_single_npy(tmp_path):
    import flopscope as we

    with we.BudgetContext(flop_budget=1_000_000):
        a = we.array([1.5, 2.5, 3.5])
        we.save(str(tmp_path / "x.npy"), a)
        assert we.load(str(tmp_path / "x.npy")).tolist() == [1.5, 2.5, 3.5]


def test_load_is_free(tmp_path):
    import flopscope as we

    with we.BudgetContext(flop_budget=1_000_000):
        a = we.array([1.0] * 50)
        we.savez(str(tmp_path / "f.npz"), a=a)
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        we.load(str(tmp_path / "f.npz"))
        # use whichever accessor the client BudgetContext exposes for flops used
        assert _flops_used(budget) == 0


def _flops_used(budget):
    if hasattr(budget, "flops_used"):
        return budget.flops_used
    return budget.budget_status()["flops_used"]


def test_load_numpy_authored_file(tmp_path):
    np = pytest.importorskip("numpy")
    import flopscope as we

    np.savez(str(tmp_path / "authored.npz"), W=np.array([5.0, 6.0, 7.0]))
    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "authored.npz"))
        assert out["W"].tolist() == [5.0, 6.0, 7.0]
