"""Behavioral coverage for input-coercion and branch paths across public APIs.

Three families:

* ``fnp.random`` — scalar draws, int/list/array pool equivalence for the
  cost formulas, global-state round trip, the object-dtype-ban refusal on
  a ``choice`` pool, and the ``random.symmetric`` shape/distribution parser
  (values stay honest: results are checked for symmetry / reproducibility,
  errors for type).
* ``fnp.linalg`` / ``fnp.fft`` — plain-list inputs must produce numpy's exact
  values (the ``asarray`` coercion branch of each op), plus norm's invalid-ord
  error passthrough, cond's NaN handling, and matrix_rank's 1-D/tol branches.
* symmetric constructors — ``flops.symmetrize`` equals the manual group
  average for cyclic/dihedral/young groups, ``is_symmetric`` distinguishes
  symmetric from asymmetric data, ``as_symmetric`` validates.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import UnsupportedDtypeError


@pytest.fixture
def budget():
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        yield b


def _billed(b, fn):
    start = b.flops_used
    result = fn()
    return result, b.flops_used - start


# ── random module branches ──


def test_scalar_draws_bill_like_single_element(budget):
    x, cost_scalar = _billed(budget, lambda: fnp.random.exponential())
    assert isinstance(x, float)
    y, cost_one = _billed(budget, lambda: fnp.random.exponential(size=1))
    assert np.asarray(y).shape == (1,)
    # size=None returns a bare scalar but is billed as one element.
    assert cost_scalar == cost_one > 0
    u, cost_u = _billed(budget, lambda: fnp.random.uniform())
    assert isinstance(u, float) and 0.0 <= u < 1.0
    assert cost_u > 0


def test_permutation_pool_forms_bill_identically(budget):
    # NOTE: unlike numpy, fnp.random.permutation(list) currently crashes with
    # AttributeError ('list' has no .shape) -- a numpy-parity gap deliberately
    # not asserted here; int and ndarray pools must agree.
    _, cost_int = _billed(budget, lambda: fnp.random.permutation(7))
    _, cost_arr = _billed(budget, lambda: fnp.random.permutation(np.arange(7)))
    assert cost_int == cost_arr > 0
    # Cost follows shape[0] (the permuted axis), not total element count.
    _, cost_2d = _billed(budget, lambda: fnp.random.permutation(np.zeros((3, 50))))
    _, cost_3 = _billed(budget, lambda: fnp.random.permutation(3))
    assert cost_2d == cost_3
    # Values: a permutation of range(7).
    perm = np.asarray(fnp.random.permutation(7))
    assert sorted(perm.tolist()) == list(range(7))


def test_choice_pool_forms_bill_identically(budget):
    _, cost_list = _billed(budget, lambda: fnp.random.choice([10, 20, 30], size=2))
    _, cost_int = _billed(budget, lambda: fnp.random.choice(3, size=2))
    _, cost_arr = _billed(
        budget, lambda: fnp.random.choice(np.array([10, 20, 30]), size=2)
    )
    assert cost_list == cost_int == cost_arr > 0
    picked = fnp.random.choice([10, 20, 30], size=2)
    assert set(np.asarray(picked).tolist()) <= {10, 20, 30}


def test_choice_object_dtype_pool_is_refused(budget):
    """Superseded by the object-dtype ban (see test_object_dtype_ban.py):
    ``choice``/``permutation``/``shuffle`` relocate values without touching
    them, so an object-dtype pool used to be let through with identity
    preserved on a ``size=None`` pick (the prior behavior this test used to
    pin). The ban is unconditional -- object dtype is refused everywhere,
    including pure data-movement ops -- so an object-dtype pool now raises
    before any pick happens, rather than returning one of its elements."""
    first, second = {"a": 1}, {"b": 2}
    pool = np.empty(2, dtype=object)
    pool[0], pool[1] = first, second
    with pytest.raises(UnsupportedDtypeError):
        fnp.random.choice(pool)


def test_global_state_roundtrip_reproduces_stream(budget):
    state = fnp.random.get_state()
    first = np.asarray(fnp.random.random(5))
    fnp.random.set_state(state)
    second = np.asarray(fnp.random.random(5))
    np.testing.assert_array_equal(first, second)


def test_random_symmetric_shape_and_distribution_forms(budget):
    s2 = flops.SymmetryGroup.symmetric(axes=(0, 1))

    def keyword_only_sampler(*, size):
        # Rejects positional shape args, forcing the parser's size= fallback.
        return np.random.default_rng(0).standard_normal(size)

    # list shape, string distributions incl. the rand/randn *args-form, and a
    # custom callable (keyword-only -> exercises the TypeError fallback).
    for shape, dist, kwargs in [
        ([4, 4], "rand", {}),
        ((4, 4), "randn", {}),
        ((4, 4), "standard_normal", {}),
        ((4, 4), "uniform", {"low": -1.0, "high": 1.0}),
        ((4, 4), keyword_only_sampler, {}),
    ]:
        got = fnp.random.symmetric(shape, s2, distribution=dist, **kwargs)
        arr = np.asarray(got)
        assert arr.shape == (4, 4)
        np.testing.assert_allclose(arr, arr.T, atol=1e-12)
    # int shape (with a matching 1-axis group).
    s1 = flops.SymmetryGroup.symmetric(axes=(0,))
    one_d = fnp.random.symmetric(4, s1, distribution="randn")
    assert np.asarray(one_d).shape == (4,)


def test_random_symmetric_rejects_bad_arguments(budget):
    s2 = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with pytest.raises(AttributeError, match="does not provide distribution"):
        fnp.random.symmetric((4, 4), s2, distribution="not_a_distribution")
    with pytest.raises(TypeError, match="distribution must be"):
        fnp.random.symmetric((4, 4), s2, distribution=12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape must be"):
        fnp.random.symmetric(object(), s2)  # type: ignore[arg-type]


# ── linalg branch paths ──


def test_linalg_ops_accept_plain_lists(budget):
    m = [[3.0, 1.0], [1.0, 2.0]]
    np.testing.assert_allclose(float(fnp.linalg.trace(m)), np.trace(np.asarray(m)))
    np.testing.assert_allclose(float(fnp.linalg.det(m)), np.linalg.det(m))
    sign, logdet = fnp.linalg.slogdet(m)
    ref_sign, ref_logdet = np.linalg.slogdet(m)
    assert float(sign) == float(ref_sign)
    np.testing.assert_allclose(float(logdet), float(ref_logdet))
    np.testing.assert_allclose(float(fnp.linalg.norm(m)), np.linalg.norm(m))
    np.testing.assert_allclose(float(fnp.linalg.vector_norm([3.0, 4.0])), 5.0)
    np.testing.assert_allclose(
        float(fnp.linalg.matrix_norm(m)), np.linalg.norm(m, ord="fro")
    )
    np.testing.assert_allclose(float(fnp.linalg.cond(m)), np.linalg.cond(m))
    assert int(fnp.linalg.matrix_rank(m)) == 2


def test_norm_error_passthrough_matches_numpy(budget):
    # Invalid ord and out-of-bounds axis are delegated to numpy so the error
    # type/message match plain numpy exactly.
    v = np.arange(4.0)
    with pytest.raises(ValueError):
        fnp.linalg.norm(v, ord="not-an-ord")  # type: ignore[arg-type]
    try:
        np.linalg.norm(v, axis=5)
        raise AssertionError("numpy accepted axis=5")
    except Exception as np_exc:  # noqa: BLE001 - mirror whatever numpy raises
        with pytest.raises(type(np_exc)):
            fnp.linalg.norm(v, axis=5)


def test_cond_nan_matrix_returns_nan(budget):
    bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert np.isnan(float(fnp.linalg.cond(bad)))
    # Batch: only the NaN matrix degrades; the clean one keeps its value.
    batch = np.stack([bad, np.eye(2)])
    got = np.asarray(fnp.linalg.cond(batch))
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1], np.linalg.cond(np.eye(2)))


def test_matrix_rank_low_dim_and_tolerances(budget):
    assert int(fnp.linalg.matrix_rank([1.0, 0.0, 2.0])) == 1
    assert int(fnp.linalg.matrix_rank(np.zeros(3))) == 0
    m = np.diag([1.0, 1e-12])
    assert int(fnp.linalg.matrix_rank(m, tol=1e-6)) == 1
    assert int(fnp.linalg.matrix_rank(m, rtol=1e-15)) == 2


# ── fft list-coercion across the transform family ──


@pytest.mark.parametrize(
    "name",
    ["fft", "ifft", "rfft", "ihfft"],
)
def test_fft_1d_accepts_lists(budget, name):
    data = [0.0, 1.0, 2.0, 3.0]
    got = np.asarray(getattr(fnp.fft, name)(data))
    expected = getattr(np.fft, name)(data)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("name", ["irfft", "hfft"])
def test_fft_halfspectrum_accepts_lists(budget, name):
    data = [1.0 + 0j, 2.0 + 1j, 0.5 + 0j]
    got = np.asarray(getattr(fnp.fft, name)(data))
    expected = getattr(np.fft, name)(data)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize(
    "name",
    ["fft2", "ifft2", "rfft2", "fftn", "ifftn", "rfftn"],
)
def test_fft_2d_accepts_lists(budget, name):
    data = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    got = np.asarray(getattr(fnp.fft, name)(data))
    expected = getattr(np.fft, name)(data)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("name", ["irfft2", "irfftn"])
def test_fft_inverse_real_2d_accepts_lists(budget, name):
    data = [[1.0 + 0j, 2.0 + 1j], [0.5 + 0j, 0.25 + 0j]]
    got = np.asarray(getattr(fnp.fft, name)(data))
    expected = getattr(np.fft, name)(data)
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ── symmetric constructors: numerics vs manual group averages ──


def test_symmetrize_cyclic_matches_manual_average(budget):
    g = flops.SymmetryGroup.cyclic(axes=(0, 1, 2))
    a = np.arange(27.0).reshape(3, 3, 3)
    got = np.asarray(flops.symmetrize(a, symmetry=g))
    manual = (a + np.transpose(a, (1, 2, 0)) + np.transpose(a, (2, 0, 1))) / 3.0
    np.testing.assert_allclose(got, manual)
    assert flops.is_symmetric(got, symmetry=g)
    assert not flops.is_symmetric(a, symmetry=g)


def test_symmetrize_dihedral_matches_manual_average(budget):
    g = flops.SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
    b = np.random.default_rng(0).random((2, 2, 2, 2))
    got = np.asarray(flops.symmetrize(b, symmetry=g))
    rotations = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    reflections = [(3, 2, 1, 0), (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3)]
    manual = sum(np.transpose(b, p) for p in rotations + reflections) / 8.0
    np.testing.assert_allclose(got, manual)
    assert flops.is_symmetric(got, symmetry=g)


def test_symmetrize_young_blocks_match_manual_average(budget):
    g = flops.SymmetryGroup.young([(0, 1), (2, 3)])
    b = np.random.default_rng(1).random((2, 2, 2, 2))
    got = np.asarray(flops.symmetrize(b, symmetry=g))
    perms = [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
    ]
    manual = sum(np.transpose(b, p) for p in perms) / 4.0
    np.testing.assert_allclose(got, manual)
    assert flops.is_symmetric(got, symmetry=g)


def test_as_symmetric_validates_data(budget):
    s2 = flops.SymmetryGroup.symmetric(axes=(0, 1))
    sym = np.array([[1.0, 2.0], [2.0, 5.0]])
    wrapped = flops.as_symmetric(sym, symmetry=s2)
    assert wrapped.symmetry == s2
    asym = np.array([[1.0, 2.0], [3.0, 5.0]])
    with pytest.raises(flops.errors.SymmetryError):
        flops.as_symmetric(asym, symmetry=s2)
