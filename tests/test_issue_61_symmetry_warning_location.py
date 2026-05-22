"""Regression tests for issue #61: SymmetryLossWarning must point to the
user's call site, not internal flopscope code.

See https://github.com/AIcrowd/flopscope/issues/61.
"""

from __future__ import annotations

import os
import warnings

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope.errors import SymmetryLossWarning

_FLOPSCOPE_PKG_DIR = os.path.dirname(os.path.abspath(flops.__file__))


def _record_symmetry_loss(callable_):
    """Run *callable_* and return the recorded ``SymmetryLossWarning`` entries."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        callable_()
    return [w for w in caught if issubclass(w.category, SymmetryLossWarning)]


def test_binary_add_warning_points_to_caller():
    """``A + B`` with mismatched symmetries must blame the user's `+` line."""
    n = 4
    with flops.BudgetContext(flop_budget=10**9):
        A = flops.symmetrize(
            fnp.random.randn(n, n, n), symmetry=SymmetryGroup.symmetric(axes=(0, 1))
        )
        B = flops.symmetrize(
            fnp.random.randn(n, n, n), symmetry=SymmetryGroup.cyclic(axes=(0, 1, 2))
        )

        def _do_add():
            _ = A + B  # noqa: F841

        warnings_caught = _record_symmetry_loss(_do_add)

    assert len(warnings_caught) == 1
    w = warnings_caught[0]
    assert w.filename == __file__, (
        f"warning should point at this test file, got {w.filename!r}"
    )
    assert not w.filename.startswith(_FLOPSCOPE_PKG_DIR), (
        f"warning should NOT point inside flopscope/, got {w.filename!r}"
    )


def test_reduction_warning_points_to_caller():
    """``A.sum(axis=...)`` over a symmetric axis must blame the user's reduction."""
    n = 4
    with flops.BudgetContext(flop_budget=10**9):
        A = flops.symmetrize(
            fnp.random.randn(n, n), symmetry=SymmetryGroup.symmetric(axes=(0, 1))
        )

        def _do_reduce():
            _ = A.sum(axis=0)  # noqa: F841

        warnings_caught = _record_symmetry_loss(_do_reduce)

    assert len(warnings_caught) >= 1
    w = warnings_caught[0]
    assert w.filename == __file__, (
        f"warning should point at this test file, got {w.filename!r}"
    )
    assert not w.filename.startswith(_FLOPSCOPE_PKG_DIR), (
        f"warning should NOT point inside flopscope/, got {w.filename!r}"
    )


def test_slicing_warning_points_to_caller():
    """Slicing a SymmetricTensor that breaks the symmetric group must blame the
    user's `[...]` line."""
    n = 4
    with flops.BudgetContext(flop_budget=10**9):
        A = flops.symmetrize(
            fnp.random.randn(n, n, n),
            symmetry=SymmetryGroup.symmetric(axes=(0, 1, 2)),
        )

        def _do_slice():
            _ = A[0, :, :]  # noqa: F841

        warnings_caught = _record_symmetry_loss(_do_slice)

    assert len(warnings_caught) >= 1
    w = warnings_caught[0]
    assert w.filename == __file__, (
        f"warning should point at this test file, got {w.filename!r}"
    )
    assert not w.filename.startswith(_FLOPSCOPE_PKG_DIR), (
        f"warning should NOT point inside flopscope/, got {w.filename!r}"
    )
