"""Einsum pairwise execution must go through numpy's optimized dispatch.

``_execute_pairwise`` receives steps whose path was chosen (and billed)
before execution, so optimized dispatch may only change *how* a step
executes, never what is contracted or what the caller observes beyond
float summation order. These tests pin the properties of that
arrangement:

1. every two-operand numeric step passes ``optimize=True`` to
   ``numpy.einsum``, and runs at BLAS speed rather than at the speed of
   numpy's non-dispatching sum-of-products loop;
2. a multi-operand contraction still executes the billed path step by
   step, never as one fused numpy call over all operands — including a
   user-pinned n-ary step, which numpy must not re-plan;
3. object-dtype operands never reach a dispatch decision at all -- the
   object-dtype ban (0.10.1) refuses them before ``numpy.einsum`` is ever
   called, on either path;
4. billed FLOPs and result values are unaffected by dispatch.
"""

import time

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor
from flopscope.errors import UnsupportedDtypeError


def test_pairwise_steps_call_numpy_einsum_with_optimize(monkeypatch):
    rng = np.random.default_rng(0)
    ops = [
        rng.standard_normal(shape) for shape in [(24, 28), (28, 20), (20, 36), (36, 16)]
    ]
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        _, info = fnp.einsum_path("ab,bc,cd,de->ae", *ops)
    expected_steps = [step.subscript for step in info.steps]
    assert len(expected_steps) == 3

    real_einsum = np.einsum
    calls = []

    def spy(*args, **kwargs):
        calls.append(
            (
                args[0],
                len(args) - 1,
                kwargs.get("optimize", "MISSING"),
            )
        )
        return real_einsum(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", spy)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        result = fnp.einsum("ab,bc,cd,de->ae", *ops)

    # The billed path is executed step by step, two operands at a time.
    assert [call[0] for call in calls] == expected_steps
    assert all(call[1] == 2 for call in calls)
    # Each step goes through numpy's optimized (BLAS-capable) dispatch.
    assert all(call[2] is True for call in calls)

    expected = np.einsum("ab,bc,cd,de->ae", *ops, optimize=True)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12)


def test_explicit_nary_path_step_executes_as_billed(monkeypatch):
    # A caller may pin an n-ary step (all three operands contracted at
    # once). numpy's optimizer would silently decompose it into its own
    # pairwise path; execution must instead honor the step as billed.
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 7))
    b = rng.standard_normal((7, 8))
    c = rng.standard_normal((8, 5))

    real_einsum = np.einsum
    calls = []

    def spy(*args, **kwargs):
        calls.append((len(args) - 1, kwargs.get("optimize", False)))
        return real_einsum(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", spy)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        result = fnp.einsum("ij,jk,kl->il", a, b, c, optimize=[(0, 1, 2)])

    assert calls == [(3, False)]
    np.testing.assert_allclose(
        np.asarray(result),
        real_einsum("ij,jk,kl->il", a, b, c),
        rtol=1e-12,
        atol=1e-10,
    )


class _NC:
    """Object payload that records evaluation order (non-commutative)."""

    def __init__(self, tag):
        self.tag = tag

    def __mul__(self, other):
        return _NC(f"({self.tag}*{other.tag})")

    def __add__(self, other):
        return _NC(f"({self.tag}+{other.tag})")

    def __radd__(self, other):
        return _NC(f"({other}+{self.tag})")


def test_object_dtype_operands_are_refused_before_any_dispatch(monkeypatch):
    """Through 0.10.0 the optimized route needed to stay off object-dtype
    steps: tensordot flips per-term multiplication order and drops
    c_einsum's `0 + first_term` accumulator seeding relative to the plain
    loop. As of the 0.10.1 object-dtype ban neither path runs at all -- an
    object operand is refused before dispatch is even chosen, which this
    pins by asserting numpy.einsum (spied below) is never called."""
    A = np.empty((2, 3), dtype=object)
    B = np.empty((3, 2), dtype=object)
    for i in range(2):
        for j in range(3):
            A[i, j] = _NC(f"a{i}{j}")
    for i in range(3):
        for j in range(2):
            B[i, j] = _NC(f"b{i}{j}")

    real_einsum = np.einsum
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("optimize", False))
        return real_einsum(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", spy)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("ij,jk->ik", A, B)

    assert calls == []


def test_object_dtype_scalar_form_is_refused_before_any_dispatch():
    """Same refusal for the 'i,i->' scalar-output form, which used to need
    its own pin -- plain np.einsum returns a raw Python scalar for object
    input (asarray coerces it to int64) while the optimized route would
    have returned a 0-d object array instead. Moot now: object never
    reaches either path."""
    obj = np.array([2, 3, 4], dtype=object)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("i,i->", obj, obj)


def test_nonfinite_result_with_declared_symmetry_is_never_stamped():
    # Symmetry validation skips non-finite results and must then refuse
    # to stamp: an unverifiable claim never becomes a SymmetricTensor
    # (which would grant downstream discounts). Summation order may
    # decide *whether* an overflow-boundary result is finite, so this
    # invariant is what keeps that dispatch-dependent — but always
    # safe — rather than exploitable.
    big = np.full((2, 8), 9.5e153)
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        result = fnp.einsum("ij,kj->ik", big, big, symmetry=(0, 1))
    assert not np.all(np.isfinite(np.asarray(result)))
    assert not isinstance(result, SymmetricTensor)


def test_two_operand_einsum_reaches_blas_speed():
    N = 896
    rng = np.random.default_rng(1)
    A = rng.random((N, N), dtype=np.float32)
    B = rng.random((N, N), dtype=np.float32)

    # Reference: numpy's non-dispatching loop on this machine, so the
    # comparison is relative and survives slow or busy CI hosts.
    plain = min(_timed(lambda: np.einsum("ij,jk->ik", A, B)) for _ in range(3))

    # Warm the path cache so the timed runs measure execution only.
    with flops.BudgetContext(flop_budget=10**18, quiet=True):
        fnp.einsum("ij,jk->ik", A, B)

    dispatched = []
    for _ in range(3):
        with flops.BudgetContext(flop_budget=10**18, quiet=True) as bc:
            fnp.einsum("ij,jk->ik", A, B)
        dispatched.append(bc.flopscope_backend_time_s)

    # A non-dispatching fallback runs the same non-BLAS loop as ``plain``,
    # so it lands at ~1x; genuine BLAS dispatch on this workload measures
    # ~2.8-3.0x faster. The 2x bar sits comfortably between the two, so it
    # still fails loudly if dispatch regresses while tolerating the run-to-run
    # timing noise that made a 3x bar flaky on busy CI hosts.
    assert min(dispatched) < plain / 2, (
        f"fnp.einsum backend time {min(dispatched) * 1e3:.2f}ms is not "
        f"meaningfully faster than numpy's non-BLAS loop "
        f"({plain * 1e3:.2f}ms); the pairwise step is not dispatching "
        f"through BLAS"
    )


def _timed(thunk):
    t0 = time.perf_counter()
    thunk()
    return time.perf_counter() - t0


def _bill(thunk):
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        thunk()
    return bc.flops_used


def test_billed_flops_unaffected_by_execution_dispatch():
    # Regression pins: how a step executes (BLAS vs numpy's loop) must never
    # move the analytical bill. All values predate the optimize=True dispatch.
    rng = np.random.default_rng(42)
    a = rng.standard_normal((64, 64))
    b = rng.standard_normal((64, 64))
    c = rng.standard_normal((48, 40))
    d = rng.standard_normal((40, 56))
    e = rng.standard_normal((56, 32))
    ab = rng.standard_normal((4, 32, 32))
    bb = rng.standard_normal((4, 32, 32))
    s = rng.standard_normal((40, 40))
    s = s + s.T
    v = rng.standard_normal((40, 24))
    p1 = rng.standard_normal((24, 28))
    p2 = rng.standard_normal((28, 20))
    p3 = rng.standard_normal((20, 36))
    p4 = rng.standard_normal((36, 16))

    assert _bill(lambda: fnp.einsum("ij,jk->ik", a, b)) == 520_192
    assert _bill(lambda: fnp.einsum("ij,jk,kl->il", c, d, e)) == 263_424
    assert (
        _bill(lambda: fnp.einsum("ij,jk,kl->il", c, d, e, optimize=[(0, 1), (0, 1)]))
        == 263_424
    )
    assert _bill(lambda: fnp.einsum("bij,bjk->bik", ab, bb)) == 258_048
    assert (
        _bill(
            lambda: fnp.einsum("ij,jk->ik", flops.as_symmetric(s, symmetry=(0, 1)), v)
        )
        == 87_039
    )
    assert _bill(lambda: fnp.einsum("ab,bc,cd,de->ae", p1, p2, p3, p4)) == 61_312


def test_results_match_numpy_reference_within_tolerance():
    rng = np.random.default_rng(5)
    f32 = [
        rng.standard_normal((30, 40, 20)).astype(np.float32),
        rng.standard_normal((40, 20, 24)).astype(np.float32),
    ]
    c128 = [
        rng.standard_normal((48, 56)) + 1j * rng.standard_normal((48, 56)),
        rng.standard_normal((56, 32)) + 1j * rng.standard_normal((56, 32)),
    ]
    cases = [
        (
            "ij,jk->ik",
            [rng.standard_normal((80, 96)), rng.standard_normal((96, 64))],
            1e-12,
            1e-10,
        ),
        (
            "bij,bjk->bik",
            [rng.standard_normal((4, 48, 56)), rng.standard_normal((4, 56, 40))],
            1e-12,
            1e-10,
        ),
        (
            "ij,jk,kl->il",
            [
                rng.standard_normal((32, 48)),
                rng.standard_normal((48, 40)),
                rng.standard_normal((40, 24)),
            ],
            1e-12,
            1e-10,
        ),
        ("ijk,jkl->il", f32, 1e-5, 1e-4),
        ("ij,jk->ik", c128, 1e-12, 1e-10),
    ]
    for subscripts, operands, rtol, atol in cases:
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fnp.einsum(subscripts, *operands)
        np.testing.assert_allclose(
            np.asarray(got),
            np.einsum(subscripts, *operands),
            rtol=rtol,
            atol=atol,
        )
