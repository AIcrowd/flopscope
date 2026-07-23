"""Behavioral coverage: Generator-proxy stream parity and symmetry propagation.

* ``fnp.random.default_rng(seed)`` methods must produce the exact same streams
  as ``np.random.default_rng(seed)`` — the proxies bill FLOPs but may not
  perturb values, dtypes, or state (this drives the pool-form/axis branches of
  the RNG cost formulas).
* ``SymmetricTensor`` results of slicing/reduction/binary ops carry the
  documented symmetry propagation: surviving axis groups stay attached, full
  loss downgrades to a plain counted array with a ``SymmetryLossWarning``.
* einsum plumbing branches: identical-operand dedup with explicit list paths,
  ``optimize=True``, and the k>=8 'auto'->greedy switch all reproduce numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import SymmetryLossWarning


@pytest.fixture
def budget():
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        yield b


# ── Generator proxy: exact stream parity with numpy ──


def test_generator_permutation_and_integers_parity(budget):
    ours, theirs = fnp.random.default_rng(42), np.random.default_rng(42)
    np.testing.assert_array_equal(
        np.asarray(ours.permutation(8)), theirs.permutation(8)
    )
    ours2, theirs2 = fnp.random.default_rng(7), np.random.default_rng(7)
    np.testing.assert_array_equal(
        np.asarray(ours2.integers(0, 100, size=5)), theirs2.integers(0, 100, size=5)
    )


def test_generator_choice_with_axis_parity(budget):
    pool = np.arange(12).reshape(3, 4)
    ours, theirs = fnp.random.default_rng(1), np.random.default_rng(1)
    np.testing.assert_array_equal(
        np.asarray(ours.choice(pool, size=2, axis=1)),
        theirs.choice(pool, size=2, axis=1),
    )
    # List pools sample from the same values.
    ours3 = fnp.random.default_rng(3)
    picked = np.asarray(ours3.choice([5, 6, 7], size=2))
    assert set(picked.tolist()) <= {5, 6, 7}


def test_generator_shuffle_and_permuted_parity(budget):
    a_ours, a_theirs = np.arange(10.0), np.arange(10.0)
    ours, theirs = fnp.random.default_rng(2), np.random.default_rng(2)
    ours.shuffle(a_ours)
    theirs.shuffle(a_theirs)
    np.testing.assert_array_equal(a_ours, a_theirs)

    grid = np.arange(12).reshape(3, 4)
    ours2, theirs2 = fnp.random.default_rng(5), np.random.default_rng(5)
    np.testing.assert_array_equal(
        np.asarray(ours2.permuted(grid, axis=1)), theirs2.permuted(grid, axis=1)
    )


def test_generator_bytes_parity_and_billing(budget):
    before = budget.flops_used
    ours, theirs = fnp.random.default_rng(3), np.random.default_rng(3)
    assert ours.bytes(16) == theirs.bytes(16)
    assert budget.flops_used > before  # RNG output is billed


# ── SymmetricTensor propagation through slicing / reduction / binary ops ──


def _s2_tensor():
    base = np.arange(16.0).reshape(4, 4)
    symd = (base + base.T) / 2
    return symd, flops.as_symmetric(
        symd, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
    )


def _s3_tensor():
    # Runs inside the test's active BudgetContext (symmetrize is counted).
    g3 = flops.SymmetryGroup.symmetric(axes=(0, 1, 2))
    data = np.random.default_rng(0).random((3, 3, 3))
    return flops.symmetrize(data, symmetry=g3)


def test_slicing_2d_symmetric_drops_group_with_warning(budget):
    symd, T = _s2_tensor()
    with pytest.warns(SymmetryLossWarning, match="slicing removed all symmetric"):
        row = T[0]
    from flopscope._symmetric import SymmetricTensor

    assert not isinstance(row, SymmetricTensor)
    np.testing.assert_array_equal(np.asarray(row), symd[0])


def test_slicing_3d_symmetric_keeps_residual_group(budget):
    T3 = _s3_tensor()
    with pytest.warns(SymmetryLossWarning, match="slicing reduced symmetric group"):
        sl = T3[1]
    from flopscope._symmetric import SymmetricTensor

    # Slicing one of three exchangeable axes leaves an S2 on the survivors.
    assert isinstance(sl, SymmetricTensor)
    assert sl.symmetry is not None and sl.symmetry.axes == (0, 1)
    np.testing.assert_allclose(np.asarray(sl), np.asarray(sl).T)


def test_reduction_propagates_residual_group(budget):
    T3 = _s3_tensor()
    with pytest.warns(SymmetryLossWarning, match="sum reduced dims"):
        red = fnp.sum(T3, axis=0)
    from flopscope._symmetric import SymmetricTensor

    assert isinstance(red, SymmetricTensor)
    assert red.symmetry is not None and red.symmetry.axes == (0, 1)
    np.testing.assert_allclose(np.asarray(red), np.asarray(T3).sum(axis=0))

    symd, T2 = _s2_tensor()
    with pytest.warns(SymmetryLossWarning, match="sum removed all symmetric"):
        flat = fnp.sum(T2, axis=0)
    assert not isinstance(flat, SymmetricTensor)
    np.testing.assert_allclose(np.asarray(flat), symd.sum(axis=0))


def test_same_group_binary_op_keeps_symmetry(budget):
    import warnings

    symd, T = _s2_tensor()
    from flopscope._symmetric import SymmetricTensor

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no SymmetryLossWarning may fire
        added = fnp.add(T, T)
    assert isinstance(added, SymmetricTensor)
    assert added.symmetry == T.symmetry
    np.testing.assert_allclose(np.asarray(added), symd * 2)


def test_mixed_group_binary_op_downgrades_with_warning(budget):
    symd, T = _s2_tensor()
    plain = np.arange(16.0).reshape(4, 4)
    with pytest.warns(SymmetryLossWarning, match="no symmetry groups shared"):
        mixed = fnp.add(T, plain)
    from flopscope._symmetric import SymmetricTensor

    assert not isinstance(mixed, SymmetricTensor)
    np.testing.assert_allclose(np.asarray(mixed), symd + plain)


# ── einsum plumbing branches ──


def test_einsum_identity_dedup_with_list_path_and_true(budget):
    same = np.random.default_rng(0).random((4, 4))
    expected = same @ same @ same
    got_list = fnp.einsum("ij,jk,kl->il", same, same, same, optimize=[(0, 1), (0, 1)])
    np.testing.assert_allclose(np.asarray(got_list), expected)
    got_true = fnp.einsum("ij,jk,kl->il", same, same, same, optimize=True)
    np.testing.assert_allclose(np.asarray(got_true), expected)


def test_einsum_large_k_auto_switches_to_greedy(budget):
    # k >= 8 operands: 'auto' hands off to greedy; values must stay exact.
    ops = [np.full((2, 2), 1.0) for _ in range(8)]
    subs = "ab,bc,cd,de,ef,fg,gh,hi->ai"
    got = fnp.einsum(subs, *ops, optimize="auto")
    np.testing.assert_allclose(np.asarray(got), np.full((2, 2), 2.0**7))
    _, info = fnp.einsum_path(subs, *ops, optimize="auto")
    assert info.optimizer_used == "greedy"


def test_symmetric_einsum_path_renders_inner_savings(budget):
    import io
    import re

    from rich.console import Console

    _, T = _s2_tensor()
    _, info = fnp.einsum_path("ij,ij->", T, T)
    buf = io.StringIO()
    Console(
        file=buf,
        force_terminal=True,
        no_color=True,
        _environ={"COLUMNS": "200", "LINES": "60"},
    ).print(info)
    out = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
    collapsed = re.sub(r"[\s│┃╭╮╰╯─━┏┓┗┛┣┫┳┻╋]+", "", out)
    # The savings column reports the contracted-side (W) unique/total ratio
    # and the symmetry column names the S2 group.
    assert "W:" in collapsed
    assert "S2" in collapsed
