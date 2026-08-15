"""n-D buffer input to ``array()`` and buffer input to ``asarray()`` (#194).

Both documented conversions used to fail: ``asarray`` was an auto-generated
proxy with no encoding for a buffer-backed object, and ``array``'s
buffer-protocol path was gated on ``mv.ndim <= 1`` even though the wire call
``create_from_data(data, shape, dtype)`` already carries an arbitrary shape.

The client is numpy-free, so these build n-D buffers with ``array.array`` +
``memoryview.cast`` — the same ``memoryview`` path an ``numpy.ndarray`` takes
through ``array()``. The ndarray-specific cases (Fortran order, non-contiguous
strides, string dtypes) live in ``tests/client_compat/test_array_buffer_input.py``
in the root repo, where numpy is available.
"""

from __future__ import annotations

import array as _array
import os
import signal
import subprocess
import sys
import time

import pytest

import flopscope as fnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
_SERVER_VENV_PYTHON = os.path.join(
    _WORKTREE, "flopscope-server", ".venv", "bin", "python"
)
_ROOT_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")
_VENV_PYTHON = (
    _SERVER_VENV_PYTHON if os.path.exists(_SERVER_VENV_PYTHON) else _ROOT_VENV_PYTHON
)

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SERVER_URL = "tcp://127.0.0.1:15573"

_SERVER_SCRIPT = f"""
import sys, os
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
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture()
def budget():
    with fnp.BudgetContext(flop_budget=10**12, quiet=True) as ctx:
        yield ctx


def _reshape(buf, fmt, shape):
    """Reshape a 1-D buffer without numpy (memoryview casts only via bytes)."""
    return memoryview(buf).cast("B").cast(fmt, shape)


def _buf_2d():
    """A 2x3 float64 buffer, built without numpy."""
    return _reshape(_array.array("d", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), "d", (2, 3))


def _buf_3d():
    return _reshape(_array.array("i", list(range(24))), "i", (2, 3, 4))


# ---------------------------------------------------------------------------
# array(): the rank gate
# ---------------------------------------------------------------------------


def test_array_accepts_2d_buffer(budget):
    out = fnp.array(_buf_2d())
    assert out.shape == (2, 3)
    assert out.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_array_accepts_3d_buffer(budget):
    out = fnp.array(_buf_3d())
    assert out.shape == (2, 3, 4)
    assert out.tolist() == [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ]


def test_array_accepts_2d_buffer_with_dtype_cast(budget):
    out = fnp.array(_buf_2d(), dtype="float32")
    assert out.shape == (2, 3)
    assert out.dtype == "float32"


def test_array_1d_buffer_shape_is_unchanged(budget):
    out = fnp.array(_array.array("d", [1.0, 2.0, 3.0]))
    assert out.shape == (3,)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_array_of_zero_d_buffer_has_rank_zero(budget):
    """A 0-d buffer is rank 0, not rank 1.

    The old path derived the length from ``nbytes // itemsize`` and always sent
    ``[n]``, so a 0-d buffer came back with shape ``(1,)``. ``list(mv.shape)``
    sends ``[]`` instead — which the wire call has always accepted, since the
    scalar branch of ``array()`` sends exactly that.
    """
    zero_d = _reshape(_array.array("d", [3.0]), "d", ())
    assert zero_d.ndim == 0
    out = fnp.array(zero_d)
    assert out.shape == ()


def test_array_still_rejects_bytes(budget):
    for bad in (b"abc", bytearray(b"abc")):
        with pytest.raises(TypeError, match="Cannot create array from"):
            fnp.array(bad)


# Non-native buffer FORMATS (big-endian, structured, unicode) need numpy to
# construct, so those rejections are pinned in
# tests/client_compat/test_array_buffer_input.py.


# ---------------------------------------------------------------------------
# asarray(): buffer input
# ---------------------------------------------------------------------------


def test_asarray_accepts_1d_buffer(budget):
    out = fnp.asarray(_array.array("d", [1.0, 2.0, 3.0]))
    assert out.shape == (3,)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_asarray_accepts_2d_buffer(budget):
    out = fnp.asarray(_buf_2d())
    assert out.shape == (2, 3)
    assert out.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_asarray_accepts_buffer_with_dtype(budget):
    out = fnp.asarray(_buf_2d(), dtype="float32")
    assert out.shape == (2, 3)
    assert out.dtype == "float32"


# ---------------------------------------------------------------------------
# asarray(): inputs the wire already carries must keep their exact dispatch
# ---------------------------------------------------------------------------
#
# These pin the amounts billed TODAY, before the special case existed. #194 is
# scoped to un-refusing calls that error; nothing that already works may move.


def test_asarray_billing_is_unchanged_for_wire_carried_inputs(budget):
    a = fnp.ones((4, 4))

    before = budget.flops_used
    out = fnp.asarray(a)
    assert out.shape == (4, 4)
    assert budget.flops_used - before == 0, "same-dtype asarray is a no-op copy"

    before = budget.flops_used
    out = fnp.asarray(a, dtype="float32")
    assert out.dtype == "float32"
    assert budget.flops_used - before == 32, "the dtype cast must still be billed"

    before = budget.flops_used
    out = fnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert budget.flops_used - before == 8, "asarray on a list must still be billed"

    before = budget.flops_used
    out = fnp.asarray(3.0)
    assert out.shape == ()
    assert budget.flops_used - before == 2, "asarray on a scalar must still be billed"


def test_asarray_still_reaches_the_server(budget):
    """The local materialization must not short-circuit the dispatch.

    A RemoteArray is handed to the server as a handle exactly as before, so
    server-side validation still runs. ``"data type ... not understood"`` is
    numpy's own wording, and this package is numpy-free, so that message can
    only have come back from the server — which is what makes this a pin on
    the round trip and not merely on "something raised".
    """
    a = fnp.ones((2, 2))
    with pytest.raises(TypeError, match="data type 'not_a_dtype' not understood"):
        fnp.asarray(a, dtype="not_a_dtype")


def test_buffer_dtype_cast_billing_is_pinned(budget):
    """Pin what the newly-reachable n-D buffer + dtype route costs.

    A buffer carrying a dtype is ingested free and then cast server-side, while
    a nested list is packed directly in the target dtype and bills nothing. The
    asymmetry is pre-existing — the 1-D figure below is unchanged from before
    #194 — and #194 only makes it reachable above rank 1. Pinned because the
    buffer route is now a route participants can take: note it is the MORE
    expensive of the two, so it is not a cheaper path to the same array.
    """
    before = budget.flops_used
    fnp.array(_array.array("d", [1.0, 2.0, 3.0]), dtype="float32")
    assert budget.flops_used - before == 6, "1-D buffer cast: unchanged by #194"

    before = budget.flops_used
    fnp.array(_buf_2d(), dtype="float32")
    assert budget.flops_used - before == 12, "2-D buffer cast: same per-element rate"

    before = budget.flops_used
    fnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float32")
    assert budget.flops_used - before == 0, "a nested list packs in the target dtype"

    before = budget.flops_used
    fnp.array(_buf_2d())
    assert budget.flops_used - before == 0, "ingest with no cast stays free"
