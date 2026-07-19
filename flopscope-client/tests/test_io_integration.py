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
    4*(1000 + 8) = 4032 FLOPs (numel plus the array's 1-D shape-header
    channel, ndim*8 = 8 bytes; dtype rate 1.0 keeps the arithmetic exact) --
    was 0 before the save-billing fix."""
    import flopscope as we

    values = [float(i) for i in range(1000)]
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(values, dtype="float32")
        we.save(str(tmp_path / "x.npy"), a)
    assert _flops_used(budget) == 4 * (1000 + 8)

    # Round-trip result is still correct: the local file write is unaffected.
    with we.BudgetContext(flop_budget=1_000_000):
        assert we.load(str(tmp_path / "x.npy")).tolist() == values


def test_savez_bills_sum_of_egress_on_server(tmp_path):
    """savez bills 4*(n1+n2+meta_len+name_bytes+shape_header_bytes+
    names_shape_header) -- the sum across every array in the call PLUS the
    __meta__ block's serialized byte length PLUS the archive member names'
    own byte length ("a", "b", "__meta__"). __meta__ and the names blob are
    each ingested to their own server handle and billed exactly like a named
    array (see _write_npz) -- excluding them was a budget-bypass, see
    test_savez_large_meta_bills_proportionally_on_server and
    test_savez_name_channel_bills_proportionally_on_server below. Every
    billed array (a, b, the __meta__ blob) also bills an 8-byte-per-dimension
    shape-header channel, and the names blob -- ingested server-side as one
    synthetic 1-D uint8 array -- bills one more 8-byte header of its own."""
    import flopscope as we

    a_vals = [float(i) for i in range(300)]
    b_vals = [float(i) for i in range(200)]
    meta = {"k": 1}
    meta_len = len(json.dumps(meta).encode("utf-8"))
    name_bytes = len("a") + len("b") + len("__meta__")
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        b = we.array(b_vals, dtype="float32")
        we.savez(str(tmp_path / "wz.npz"), a=a, b=b, __meta__=meta)
    shape_header_bytes = 3 * 8  # a, b, __meta__ blob: all 1-D
    names_shape_header = 8  # non-empty name blob
    assert _flops_used(budget) == 4 * (
        300 + 200 + meta_len + name_bytes + shape_header_bytes + names_shape_header
    )

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
    name_bytes = len("a") + len("__meta__")
    a_vals = [float(i) for i in range(10)]
    with we.BudgetContext(flop_budget=10_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        we.savez(str(tmp_path / "exploit.npz"), a=a, __meta__=payload)
    total = _flops_used(budget)
    shape_header_bytes = 2 * 8  # a + __meta__ blob: both 1-D
    names_shape_header = 8  # non-empty name blob
    assert total == 4 * (
        10 + meta_len + name_bytes + shape_header_bytes + names_shape_header
    )
    array_only_cost = 4 * 10
    assert total > 1000 * array_only_cost  # dominated by meta, not the tiny array
    assert total != 4  # the pre-fix floor-of-1 exploit value

    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "exploit.npz"))
        assert out["a"].tolist() == a_vals
        assert out["__meta__"] == payload


def test_savez_name_channel_bills_proportionally_on_server(tmp_path):
    """Exploit regression (budget bypass), NAME channel, client+server path:
    before this fix, `_write_npz` never billed the archive's MEMBER NAMES
    (savez's kwargs keys) -- only array VALUES were billed (4*numel each) --
    so a participant could smuggle data through many tiny 1-element arrays
    given huge names for near-zero cost (the root-package pin test exercises
    the full confirmed exploit shape: 200 one-element arrays with
    60,000-char names smuggling ~12MB for a pre-fix bill of 4*200=800 FLOPs;
    this test uses a smaller N to keep the live socket round-trips fast
    while still proving the same proportional-billing property). Member
    names must now bill 4*sum(len(name bytes)) via their own server handle
    (see _write_npz), dominating the tiny arrays' own cost."""
    import flopscope as we

    n_arrays = 20
    name_len = 5_000
    names = [f"{i:04d}" + "x" * (name_len - 4) for i in range(n_arrays)]
    assert all(len(n) == name_len for n in names)
    name_bytes = sum(len(n.encode("utf-8")) for n in names)
    with we.BudgetContext(flop_budget=10_000_000) as budget:
        kwargs = {name: we.array([1.0], dtype="float32") for name in names}
        we.savez(str(tmp_path / "smuggle.npz"), **kwargs)
    total = _flops_used(budget)
    array_only_cost = 4 * n_arrays  # the pre-fix exploit value (80 FLOPs)
    shape_header_bytes = n_arrays * 8  # 20 arrays, each 1-D
    names_shape_header = 8  # non-empty name blob
    assert total == 4 * (
        n_arrays + name_bytes + shape_header_bytes + names_shape_header
    )
    assert total > 1000 * array_only_cost  # dominated by names, not array values
    assert total != array_only_cost  # the pre-fix name-channel exploit value

    with we.BudgetContext(flop_budget=1_000_000):
        out = we.load(str(tmp_path / "smuggle.npz"))
        assert sorted(out.keys()) == sorted(names)


def test_savez_compressed_bills_sum_of_egress_on_server(tmp_path):
    """savez_compressed shares savez's exact billing formula, including the
    per-array shape-header channel and the names blob's own header."""
    import flopscope as we

    a_vals = [float(i) for i in range(300)]
    b_vals = [float(i) for i in range(200)]
    name_bytes = len("a") + len("b")
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        a = we.array(a_vals, dtype="float32")
        b = we.array(b_vals, dtype="float32")
        we.savez_compressed(str(tmp_path / "wzc.npz"), a=a, b=b)
    shape_header_bytes = 2 * 8  # a, b: both 1-D
    names_shape_header = 8  # non-empty name blob
    assert _flops_used(budget) == 4 * (
        300 + 200 + name_bytes + shape_header_bytes + names_shape_header
    )

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
    4*(numel+shape_header_bytes)*dtype_rate(float64) = 4*(100+8)*2 = 864,
    not 4*numel*2 -- the array is 1-D, so shape_header_bytes = 1*8 = 8."""
    import flopscope as we

    values = [float(i) for i in range(100)]
    with we.BudgetContext(flop_budget=1_000_000) as budget:
        we.save(str(tmp_path / "plain.npy"), values)
    assert _flops_used(budget) == 4 * (100 + 8) * 2

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
