"""Tests for flopscope.numpy file I/O (load/save/savez/savez_compressed)."""

import struct

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def test_savez_load_roundtrip(tmp_path):
    p = tmp_path / "w.npz"
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.array([1, 2, 3], dtype=np.int32)
    fnp.savez(str(p), a=a, b=b)
    out = fnp.load(str(p))
    np.testing.assert_array_equal(np.asarray(out["a"]), a)
    np.testing.assert_array_equal(np.asarray(out["b"]), b)


def test_save_load_single_npy(tmp_path):
    p = tmp_path / "x.npy"
    a = np.array([[1.5, 2.5]], dtype=np.float32)
    fnp.save(str(p), a)
    np.testing.assert_array_equal(np.asarray(fnp.load(str(p))), a)


def test_meta_roundtrip(tmp_path):
    p = tmp_path / "m.npz"
    fnp.savez(str(p), W=np.zeros((2, 2)), __meta__={"sizes": [2, 2], "name": "x"})
    out = fnp.load(str(p))
    assert out["__meta__"] == {"sizes": [2, 2], "name": "x"}


def test_numpy_can_read_our_npz(tmp_path):
    p = tmp_path / "interop.npz"
    fnp.savez(str(p), W=np.ones((3,), dtype=np.float64))
    with np.load(str(p)) as z:
        np.testing.assert_array_equal(z["W"], np.ones((3,)))


def test_we_can_read_numpy_npz(tmp_path):
    p = tmp_path / "fromnumpy.npz"
    np.savez(str(p), W=np.full((2,), 7.0))
    out = fnp.load(str(p))
    np.testing.assert_array_equal(np.asarray(out["W"]), np.full((2,), 7.0))


def test_load_is_free(tmp_path):
    p = tmp_path / "free.npz"
    fnp.savez(str(p), W=np.zeros((100, 100)))
    with flops.BudgetContext(flop_budget=1_000_000) as budget:
        fnp.load(str(p))
        assert budget.flops_used == 0


def test_load_rejects_object_array_pickle(tmp_path):
    # Build a malicious .npy with an object dtype header + a pickle-ish payload.
    # Assert load REFUSES (object dtype) and never executes/loads it.
    sentinel = tmp_path / "PWNED"
    payload = b"\x80\x04\x95\x2a\x00\x00\x00\x00\x00\x00\x00\x8c\x02os"
    descr = "|O"
    header = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': (1,), }}"
    header_b = header.encode("latin1")
    total = 10 + len(header_b) + 1
    pad = (64 - total % 64) % 64
    header_b = header_b + b" " * pad + b"\n"
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_b)) + header_b + payload
    p = tmp_path / "evil.npy"
    p.write_bytes(npy)
    with pytest.raises((ValueError, OSError)):
        fnp.load(str(p))
    assert not sentinel.exists()


def test_savez_rejects_reserved_meta_array(tmp_path):
    p = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="reserved"):
        fnp.savez(str(p), __meta__=np.zeros((2,)))


def test_save_bills_shape_header_on_zero_element_array(tmp_path):
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    # numel = 0, ndim = 2 -> shape channel = 2*8 = 16 bytes -> 4 * 16 = 64
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc:
        fnp.save(
            str(tmp_path / "z.npy"), fnp.asarray(np.zeros((0, 123), dtype=np.int8))
        )
    assert bc.flops_used == 64


def test_save_bills_numel_plus_shape_header(tmp_path):
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    # numel = 12, ndim = 2 -> 4 * (12 + 16) = 112
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc:
        fnp.save(
            str(tmp_path / "a.npy"), fnp.asarray(np.ones((3, 4), dtype=np.float64))
        )
    assert bc.flops_used == 4 * (12 + 2 * 8)


# ---------------------------------------------------------------------------
# savez/savez_compressed positional arrays (numpy's *args -> arr_0, arr_1, ...)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "savez_fn", [fnp.savez, fnp.savez_compressed], ids=["savez", "savez_compressed"]
)
def test_savez_positional_roundtrip(tmp_path, savez_fn):
    """Positional arrays are auto-named arr_0, arr_1, ... (matching
    numpy.savez), merged with any keyword arrays -- both plain numpy and our
    own loader must read all of them back correctly."""
    p = tmp_path / "positional.npz"
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.array([1, 2, 3], dtype=np.int32)
    c = np.array([9.5, 8.5])
    savez_fn(str(p), a, b, x=c)

    with np.load(str(p)) as z:
        assert sorted(z.files) == ["arr_0", "arr_1", "x"]
        np.testing.assert_array_equal(z["arr_0"], a)
        np.testing.assert_array_equal(z["arr_1"], b)
        np.testing.assert_array_equal(z["x"], c)

    out = fnp.load(str(p))
    np.testing.assert_array_equal(np.asarray(out["arr_0"]), a)
    np.testing.assert_array_equal(np.asarray(out["arr_1"]), b)
    np.testing.assert_array_equal(np.asarray(out["x"]), c)


@pytest.mark.parametrize(
    "savez_fn", [fnp.savez, fnp.savez_compressed], ids=["savez", "savez_compressed"]
)
def test_savez_positional_bills_same_as_keyword(tmp_path, savez_fn):
    """A positional array must bill exactly like the same array passed under
    its auto-generated arr_N keyword name: the merge into a single
    name->value mapping happens before billing, so both call shapes hit the
    identical _bill_save_egress formula (data + shape header + name bytes)."""
    a = np.arange(500, dtype=np.float64).reshape(20, 25)

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc_positional:
        savez_fn(str(tmp_path / "positional.npz"), a)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc_keyword:
        savez_fn(str(tmp_path / "keyword.npz"), arr_0=a)

    assert bc_positional.flops_used == bc_keyword.flops_used
    # numel=500, shape header ndim(2)*8=16, name "arr_0" (5 bytes) + 8-byte
    # header for the names blob itself.
    assert bc_positional.flops_used == 4 * (500 + 2 * 8 + len("arr_0") + 8)


@pytest.mark.parametrize(
    "savez_fn", [fnp.savez, fnp.savez_compressed], ids=["savez", "savez_compressed"]
)
def test_savez_positional_keyword_collision_raises_and_bills_nothing(
    tmp_path, savez_fn
):
    """A positional array together with a colliding arr_N keyword must raise
    the exact same ValueError numpy raises (numpy.lib._npyio_impl._savez),
    and must bill nothing -- the collision is detected before any dtype
    conversion or billing happens, so a call numpy would reject is never
    billed."""
    p = tmp_path / "collide.npz"
    a = np.arange(4, dtype=np.float64)
    b = np.arange(4, dtype=np.float64)

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc:
        with pytest.raises(
            ValueError, match=r"^Cannot use un-named variables and keyword arr_0$"
        ):
            savez_fn(str(p), a, arr_0=b)

    assert bc.flops_used == 0
    assert not p.exists()
