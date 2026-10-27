"""A ``where=`` mask decides whether an elementwise result keeps its symmetry (#285).

numpy writes only the positions the mask selects. The others keep the previous
contents of ``out=``, or are left uninitialised without it, so a symmetric
operand yields a symmetric result only when the mask is invariant under the
group and the destination is a tagged buffer whose contents are checked
against it. Every other masked call must compute exactly what numpy computes
and is priced and tagged like any other operation that breaks the symmetry.
A mask that selects every position changes nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import SymmetryLossWarning

N = 64
_ASYM = np.triu(np.ones((N, N), dtype=bool))  # the issue's mask
_SYM = np.add.outer(np.arange(N), np.arange(N)) % 3 == 0


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def _operands():
    """Symmetric values, their tagged form, and an untagged twin."""
    a = np.random.default_rng(0).standard_normal((N, N))
    values = a + a.T
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(values), symmetry=(0, 1))
    return values, tagged, fnp.asarray(values)


def _tagged_zeros():
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        return flops.as_symmetric(fnp.zeros((N, N)), symmetry=(0, 1))


def _inferred_zeros():
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        return fnp.zeros((N, N))


def _tag(arr):
    return getattr(arr, "symmetry", None)


_ISSUE_CALLS = {
    "add(S, S)": (
        lambda X, **kw: fnp.add(X, X, **kw),
        lambda v, **kw: np.add(v, v, **kw),
    ),
    "exp(S)": (lambda X, **kw: fnp.exp(X, **kw), lambda v, **kw: np.exp(v, **kw)),
    "add(S, 1.0)": (
        lambda X, **kw: fnp.add(X, 1.0, **kw),
        lambda v, **kw: np.add(v, 1.0, **kw),
    ),
}


@pytest.mark.parametrize("call", list(_ISSUE_CALLS))
@pytest.mark.parametrize("dest", ["inferred", "plain"])
def test_issue_out_forms_compute_what_numpy_computes(call, dest):
    # These raised SymmetryError with the issue's fnp.zeros destination.
    flop_call, np_call = _ISSUE_CALLS[call]
    values, S, twin = _operands()

    def make_out():
        return _inferred_zeros() if dest == "inferred" else np.zeros((N, N))

    want = np_call(values, where=_ASYM, out=np.zeros((N, N)))
    dest, twin_dest = make_out(), make_out()
    billed, got = _bill(lambda: flop_call(S, where=_ASYM, out=dest))
    np.testing.assert_array_equal(np.asarray(got), want)
    assert _tag(got) is None
    assert billed == _bill(lambda: flop_call(twin, where=_ASYM, out=twin_dest))[0]


def test_issue_bare_form_is_untagged_and_billed_like_the_untagged_twin():
    values, S, twin = _operands()
    with warnings.catch_warnings():
        # The mask, not the operands, broke the symmetry: no warning that
        # blames the operands.
        warnings.simplefilter("error", SymmetryLossWarning)
        billed, R = _bill(lambda: fnp.add(S, S, where=_ASYM))
    assert _tag(R) is None
    np.testing.assert_array_equal(np.asarray(R)[_ASYM], (values + values)[_ASYM])
    assert billed == _bill(lambda: fnp.add(twin, twin, where=_ASYM))[0]
    # Later ops on R bill like an untagged (64, 64) array: numel(R) for exp
    # under the test weights (the issue's 131,072 at production weights).
    assert _bill(lambda: fnp.exp(R))[0] == _bill(lambda: fnp.exp(twin))[0] == N * N


def test_symmetric_mask_over_unvouched_positions_drops_the_tag():
    # The mask is symmetric, but the positions it skips keep a plain
    # destination's values (asymmetric here) or uninitialised memory.
    values, S, twin = _operands()
    old = np.random.default_rng(1).standard_normal((N, N))
    want = np.exp(values, where=_SYM, out=old.copy())
    dest, twin_dest = fnp.asarray(old.copy()), fnp.asarray(old.copy())
    billed, got = _bill(lambda: fnp.exp(S, where=_SYM, out=dest))
    np.testing.assert_array_equal(np.asarray(got), want)
    assert billed == _bill(lambda: fnp.exp(twin, where=_SYM, out=twin_dest))[0]
    _, bare = _bill(lambda: fnp.exp(S, where=_SYM))
    assert _tag(bare) is None


@pytest.mark.parametrize("make_out", [_inferred_zeros, _tagged_zeros])
def test_symmetric_mask_into_tagged_destination_keeps_tag_and_price(make_out):
    values, S, _ = _operands()
    want = np.exp(values, where=_SYM, out=np.zeros((N, N)))
    dest = make_out()
    billed, got = _bill(lambda: fnp.exp(S, where=_SYM, out=dest))
    np.testing.assert_array_equal(np.asarray(got), want)
    assert _tag(got) == S.symmetry
    assert billed == _bill(lambda: fnp.exp(S))[0]


@pytest.mark.parametrize(
    "where",
    [True, np.True_, np.array(True), np.ones((N, N), dtype=bool)],
    ids=["True", "np.True_", "0-d", "all-true"],
)
def test_masks_selecting_every_position_are_unchanged(where):
    values, S, _ = _operands()
    plain_bill, plain = _bill(lambda: fnp.exp(S))
    billed, got = _bill(lambda: fnp.exp(S, where=where))
    assert billed == plain_bill
    assert _tag(got) == S.symmetry
    np.testing.assert_array_equal(np.asarray(got), np.asarray(plain))


def test_multi_output_and_clip_drop_the_tag():
    values, S, twin = _operands()
    for name, op in (
        ("modf", lambda X: fnp.modf(X, where=_ASYM)),
        ("divmod", lambda X: fnp.divmod(X, 2.0, where=_ASYM)),
        ("clip", lambda X: fnp.clip(X, -1.0, 1.0, where=_ASYM)),
    ):
        billed, got = _bill(lambda op=op: op(S))
        for part in got if isinstance(got, tuple) else (got,):
            assert _tag(part) is None, name
        assert billed == _bill(lambda op=op: op(twin))[0], name


def test_clip_keeps_a_tagged_destination_where_the_mask_skips():
    # clip handed numpy a fresh buffer instead of the tagged destination, so
    # the positions the mask skips came back uninitialised.
    values, S, _ = _operands()
    want = np.clip(values, -1.0, 1.0, where=_SYM, out=np.full((N, N), 7.0))
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        dest = flops.as_symmetric(fnp.asarray(np.full((N, N), 7.0)), symmetry=(0, 1))
    _, got = _bill(lambda: fnp.clip(S, -1.0, 1.0, where=_SYM, out=dest))
    np.testing.assert_array_equal(np.asarray(got), want)
    assert _tag(got) == S.symmetry


def test_tagged_destination_is_refused_before_billing():
    # As for any other operation whose result cannot keep the destination's
    # group: refused in flopscope's own words, before anything is charged.
    _, S, _ = _operands()
    dest = _tagged_zeros()
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        with pytest.raises(ValueError, match="out symmetry does not match"):
            fnp.exp(S, where=_ASYM, out=dest)
        assert int(bc.flops_used) == 0
    np.testing.assert_array_equal(np.asarray(dest), 0.0)
