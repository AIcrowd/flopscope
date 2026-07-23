"""Behavioral coverage: SymmetryGroup group-theory API and small op tails.

* ``flops.SymmetryGroup`` — orders and structure of the named families
  (S_n / C_n / D_n), orbits, stabilizers, abelian/transitive predicates,
  payload round trip, and the young-diagram disjointness validation.
* tails — ``fnp.average(returned=True)`` (the reduction+extra-output cost
  branch), ``fnp.fromregex`` (regex text ingestion), and SymmetricTensor's
  method surface under counted ufuncs.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

G = flops.SymmetryGroup


@pytest.fixture
def budget():
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        yield b


# ── SymmetryGroup structure ──


def test_named_family_orders():
    assert G.symmetric(axes=(0, 1, 2)).order() == 6  # S3
    assert G.cyclic(axes=(0, 1, 2)).order() == 3  # C3
    assert G.dihedral(axes=(0, 1, 2, 3)).order() == 8  # D4
    assert G.symmetric(axes=(0, 1)).order() == 2  # S2
    # |elements| agrees with order().
    assert len(list(G.dihedral(axes=(0, 1, 2, 3)).elements())) == 8


def test_predicates_orbits_and_identity():
    s3 = G.symmetric(axes=(0, 1, 2))
    c3 = G.cyclic(axes=(0, 1, 2))
    assert not s3.is_abelian
    assert c3.is_abelian
    assert s3.is_transitive
    assert sorted(s3.orbit(0)) == [0, 1, 2]
    assert [sorted(o) for o in s3.orbits()] == [[0, 1, 2]]
    identity = s3.identity
    assert s3.contains(identity)
    assert identity.is_identity


def test_stabilizers():
    s3 = G.symmetric(axes=(0, 1, 2))
    # Fixing point 0 leaves the S2 swapping {1,2}.
    point_stab = s3.pointwise_stabilizer({0})
    assert point_stab.order() == 2
    assert all(g(0) == 0 for g in point_stab.elements())
    # Preserving {0,1} setwise: identity and the (0 1) swap.
    set_stab = s3.setwise_stabilizer({0, 1})
    assert set_stab.order() == 2
    assert all({g(0), g(1)} == {0, 1} for g in set_stab.elements())
    # Fixing nothing returns the whole group.
    assert s3.pointwise_stabilizer(set()).order() == 6


def test_equality_and_payload_roundtrip():
    s3 = G.symmetric(axes=(0, 1, 2))
    assert s3.equals(G.symmetric(axes=(0, 1, 2)))
    assert not s3.equals(G.cyclic(axes=(0, 1, 2)))
    revived = G.from_payload(s3.to_payload())
    assert revived == s3
    assert revived.order() == 6


def test_from_generators_and_young_validation():
    swap01 = G.from_generators([[1, 0, 2]], axes=(0, 1, 2))
    assert swap01.order() == 2
    assert sorted(swap01.orbit(0)) == [0, 1]
    # Young blocks must have disjoint supports.
    with pytest.raises(ValueError, match="disjoint supports"):
        G.young([(0, 1), (1, 2)])


# ── op tails ──


def test_average_returned_matches_numpy(budget):
    a = np.arange(12.0).reshape(3, 4)
    w = np.array([1.0, 2.0, 3.0])
    avg, cnt = fnp.average(a, axis=0, weights=w, returned=True)
    ref_avg, ref_cnt = np.average(a, axis=0, weights=w, returned=True)
    np.testing.assert_allclose(np.asarray(avg), ref_avg)
    np.testing.assert_allclose(np.asarray(cnt), ref_cnt)
    # keepdims variant reshapes both outputs.
    avg_k, cnt_k = fnp.average(a, axis=1, returned=True, keepdims=True)
    ref_avg_k, ref_cnt_k = np.average(a, axis=1, returned=True, keepdims=True)
    assert np.asarray(avg_k).shape == ref_avg_k.shape == (3, 1)
    np.testing.assert_allclose(np.asarray(avg_k), ref_avg_k)
    np.testing.assert_allclose(np.asarray(cnt_k), ref_cnt_k)
    # Full reduction (axis=None) with returned=True.
    avg_n, cnt_n = fnp.average(a, returned=True)
    ref_avg_n, ref_cnt_n = np.average(a, returned=True)
    assert float(avg_n) == pytest.approx(float(ref_avg_n))
    assert float(cnt_n) == pytest.approx(float(ref_cnt_n))


def test_fromregex_matches_numpy_and_bills(budget):
    pattern = r"(\w) (\d+)"
    dtype = [("key", "U1"), ("value", np.int64)]
    before = budget.flops_used
    got = fnp.fromregex(io.StringIO("a 1\nb 22\nc 333\n"), pattern, dtype)
    assert budget.flops_used > before
    expected = np.fromregex(io.StringIO("a 1\nb 22\nc 333\n"), pattern, dtype)
    np.testing.assert_array_equal(np.asarray(got), expected)
    assert got["value"].tolist() == [1, 22, 333]


# Plain-list inputs across the bilinear/quantile families: every op must
# coerce exactly like numpy and return numpy's values (these are the
# asarray/dims branches of each custom op).
_NAN = float("nan")
_LIST_CASES = [
    ("matmul", ([1.0, 2.0], [3.0, 4.0]), {}),
    ("matmul", ([1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]]), {}),
    ("matmul", ([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]), {}),
    ("inner", ([1.0, 2.0], [3.0, 4.0]), {}),
    ("inner", ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]), {}),
    ("outer", ([1.0, 2.0], [3.0, 4.0]), {}),
    ("tensordot", ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]), {}),
    ("vdot", ([1.0, 2.0], [3.0, 4.0]), {}),
    ("kron", ([1.0, 2.0], [3.0, 4.0]), {}),
    ("cross", ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), {}),
    ("median", ([3.0, 1.0, 2.0],), {}),
    ("median", ([[3.0, 1.0], [2.0, 4.0]],), {"axis": 1, "keepdims": True}),
    ("nanmedian", ([3.0, _NAN, 2.0],), {}),
    ("percentile", ([3.0, 1.0, 2.0], 50), {}),
    ("percentile", ([3.0, 1.0, 2.0], [25, 75]), {"keepdims": True}),
    ("quantile", ([3.0, 1.0, 2.0], 0.5), {}),
    ("quantile", ([3.0, 1.0, 2.0], [0.25, 0.75]), {}),
    ("nanpercentile", ([3.0, _NAN, 2.0], 50), {}),
    ("nanquantile", ([3.0, _NAN, 2.0], 0.5), {}),
    ("ptp", ([3.0, 1.0, 2.0],), {}),
    ("count_nonzero", ([[0, 1], [2, 0]],), {"axis": 1}),
    ("ediff1d", ([1.0, 4.0, 9.0],), {"to_begin": [0.0]}),
]


@pytest.mark.parametrize(
    "name,args,kwargs",
    _LIST_CASES,
    ids=[f"{c[0]}-{i}" for i, c in enumerate(_LIST_CASES)],
)
def test_list_inputs_match_numpy(budget, name, args, kwargs):
    got = getattr(fnp, name)(*args, **kwargs)
    expected = getattr(np, name)(*args, **kwargs)
    np.testing.assert_allclose(np.asarray(got), expected, equal_nan=True)


def test_gradient_list_input_and_spacing(budget):
    y = [1.0, 4.0, 9.0, 16.0]
    np.testing.assert_allclose(np.asarray(fnp.gradient(y)), np.gradient(y))
    # Scalar spacing surcharge branch.
    np.testing.assert_allclose(np.asarray(fnp.gradient(y, 0.5)), np.gradient(y, 0.5))
    # 2-D input returns one gradient per axis.
    z = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    got = fnp.gradient(z)
    expected = np.gradient(z)
    assert len(got) == len(expected) == 2
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_allclose(np.asarray(g), e)


def test_symmetric_tensor_method_surface(budget):
    s2 = G.symmetric(axes=(0, 1))
    base = np.arange(16.0).reshape(4, 4)
    T = flops.as_symmetric((base + base.T) / 2, symmetry=s2)
    # The instance method checks the tensor against its own group.
    assert T.is_symmetric()
    # Negation preserves the symmetry (counted ufunc route through the
    # SymmetricTensor wrap machinery).
    neg = fnp.negative(T)
    from flopscope._symmetric import SymmetricTensor

    assert isinstance(neg, SymmetricTensor)
    np.testing.assert_allclose(np.asarray(neg), -np.asarray(T))
    assert neg.is_symmetric()


def test_randint_scalar_and_generator_list_shuffle(budget):
    # size=None randint returns a bare python int within the half-open range.
    r = fnp.random.randint(10)
    assert isinstance(r, int)
    assert 0 <= r < 10
    # Generator.shuffle accepts mutable sequences and matches numpy's stream.
    ours, theirs = fnp.random.default_rng(11), np.random.default_rng(11)
    l_ours, l_theirs = [1, 2, 3, 4], [1, 2, 3, 4]
    ours.shuffle(l_ours)
    theirs.shuffle(l_theirs)
    assert l_ours == l_theirs


def test_fft_empty_input_with_explicit_n(budget):
    # Zero-length input with explicit n zero-pads exactly like numpy.
    got = np.asarray(fnp.fft.fft(np.zeros(0), n=4))
    np.testing.assert_allclose(got, np.fft.fft(np.zeros(0), n=4))


def test_pathinfo_render_names_cyclic_and_dihedral_groups(budget):
    import re as _re

    from rich.console import Console

    def collapsed_render(info):
        buf = io.StringIO()
        Console(
            file=buf,
            force_terminal=True,
            no_color=True,
            _environ={"COLUMNS": "200", "LINES": "60"},
        ).print(info)
        out = _re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        return _re.sub(r"[\s│┃╭╮╰╯─━┏┓┗┛┣┫┳┻╋]+", "", out)

    c3 = flops.symmetrize(
        np.random.default_rng(0).random((3, 3, 3)),
        symmetry=G.cyclic(axes=(0, 1, 2)),
    )
    _, info_c3 = fnp.einsum_path("ijk,ijk->", c3, c3)
    assert "C3" in collapsed_render(info_c3)

    d4 = flops.symmetrize(
        np.random.default_rng(1).random((2, 2, 2, 2)),
        symmetry=G.dihedral(axes=(0, 1, 2, 3)),
    )
    _, info_d4 = fnp.einsum_path("ijkl,ijkl->", d4, d4)
    assert "D4" in collapsed_render(info_d4)
