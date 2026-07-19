"""Client I/O integration tests — require a live subprocess server."""

import glob
import json
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


# ---------------------------------------------------------------------------
# save/savez/savez_compressed billing round-trip (budget-bypass fix).
#
# Before the fix, `save`/`savez`/`savez_compressed` were fully local: `_as_triple`
# fetched a RemoteArray's bytes via the existing FREE `_fetch_data` egress and
# wrote the .npy/.npz file with the stdlib codec, never dispatching a request
# the server could bill. A participant could compute an expensive lookup
# table server-side, then `we.save` it to disk for 0 FLOPs. `save`/`savez`/
# `savez_compressed` now round-trip to the server FIRST (handle refs only, no
# array data) so the server -- sole owner of the budget -- deducts the same
# 4*numel egress cost the in-process reference (`flops.save`) always charged.
# ---------------------------------------------------------------------------


def test_save_bills_egress_on_server(tmp_path):
    """Exact repro: saving a 1000-element float32 RemoteArray now bills
    4*1000 = 4000 FLOPs (dtype rate 1.0 keeps the arithmetic exact) -- was 0
    before the fix."""
    import flopscope as we

    values = [float(i) for i in range(1000)]
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(values, dtype="float32")
        we.save(str(tmp_path / "x.npy"), a)
    assert _flops_used(budget) == 4000

    # Round-trip result is still correct: the local file write is unaffected.
    with we.BudgetContext(flop_budget=1_000_000):
        assert we.load(str(tmp_path / "x.npy")).tolist() == values


def test_savez_bills_sum_of_egress_on_server(tmp_path):
    """savez bills 4*(n1+n2+meta_len) -- the sum across every array in the
    call PLUS the __meta__ block's serialized byte length. __meta__ is
    ingested to its own server handle and billed exactly like a named array
    (see _write_npz) -- excluding it was a budget-bypass, see
    test_savez_large_meta_bills_proportionally_on_server below."""
    import flopscope as we

    a_vals = [float(i) for i in range(300)]
    b_vals = [float(i) for i in range(200)]
    meta = {"k": 1}
    meta_len = len(json.dumps(meta).encode("utf-8"))
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        b = we.array(b_vals, dtype="float32")
        we.savez(str(tmp_path / "wz.npz"), a=a, b=b, __meta__=meta)
    assert _flops_used(budget) == 4 * (300 + 200 + meta_len)

    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "wz.npz"))
        assert out["a"].tolist() == a_vals
        assert out["b"].tolist() == b_vals
        assert out["__meta__"] == meta


def test_savez_large_meta_bills_proportionally_on_server(tmp_path):
    """Exploit regression (budget bypass), client+server path: before the
    fix, `_write_npz` never included __meta__ in the handles it billed to
    the server, so `savez(path, __meta__={...huge...})` round-tripped
    arbitrary data through __meta__ for a flat, size-independent cost --
    e.g. a 2,000,000-float payload (~10MB on disk) billed 4 FLOPs. A large
    __meta__ must now bill 4*len(json-encoded-blob), dominating a small
    named array's own cost."""
    import flopscope as we

    payload = {"payload": [0.0] * 100_000}
    meta_len = len(json.dumps(payload).encode("utf-8"))
    a_vals = [float(i) for i in range(10)]
    with we.BudgetContext(flop_budget=10_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        we.savez(str(tmp_path / "exploit.npz"), a=a, __meta__=payload)
    total = _flops_used(budget)
    assert total == 4 * (10 + meta_len)
    array_only_cost = 4 * 10
    assert total > 1000 * array_only_cost  # dominated by meta, not the tiny array
    assert total != 4  # the pre-fix floor-of-1 exploit value

    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "exploit.npz"))
        assert out["a"].tolist() == a_vals
        assert out["__meta__"] == payload


def test_savez_compressed_bills_sum_of_egress_on_server(tmp_path):
    """savez_compressed shares savez's exact billing formula."""
    import flopscope as we

    a_vals = [float(i) for i in range(300)]
    b_vals = [float(i) for i in range(200)]
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        b = we.array(b_vals, dtype="float32")
        we.savez_compressed(str(tmp_path / "wzc.npz"), a=a, b=b)
    assert _flops_used(budget) == 4 * (300 + 200)

    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "wzc.npz"))
        assert out["a"].tolist() == a_vals
        assert out["b"].tolist() == b_vals


def test_load_still_free_after_save_billing_fix(tmp_path):
    """load stays 0 FLOPs -- the fix only touches save/savez/savez_compressed."""
    import flopscope as we

    with we.BudgetContext(flop_budget=1_000_000):
        a = we.array([1.0] * 50, dtype="float32")
        we.save(str(tmp_path / "f.npy"), a)
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        we.load(str(tmp_path / "f.npy"))
    assert _flops_used(budget) == 0


def test_save_plain_value_bills_via_free_ingest_then_charges(tmp_path):
    """A plain Python list passed straight to save() (not wrapped in
    we.array() first) has no server handle yet. `_bill_save_on_server`'s
    non-RemoteArray branch ingests it via the existing free `create_from_data`
    path to get one, then bills exactly like every other save shape. Plain
    values always ingest as float64 (see `_as_triple`), so the bill is
    4*numel*dtype_rate(float64) = 4*numel*2, not 4*numel."""
    import flopscope as we

    values = [float(i) for i in range(100)]
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        we.save(str(tmp_path / "plain.npy"), values)
    assert _flops_used(budget) == 4 * 100 * 2

    with we.BudgetContext(flop_budget=1_000_000):
        assert we.load(str(tmp_path / "plain.npy")).tolist() == values


def test_save_insufficient_budget_raises_and_writes_nothing(tmp_path):
    """Server-owned counting closes the bypass end-to-end: a budget too small
    to afford the egress raises BEFORE any local file is written, instead of
    silently succeeding for free."""
    import flopscope as we

    target = tmp_path / "denied.npy"
    with we.BudgetContext(flop_budget=100):
        a = we.array([float(i) for i in range(1000)], dtype="float32")
        with pytest.raises(we.BudgetExhaustedError):
            we.save(str(target), a)
    assert not target.exists()
