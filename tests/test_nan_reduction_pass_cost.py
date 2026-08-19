"""``nan*`` reductions run an extra isnan pass and must be charged for it.

Each nan-prefixed reduction tests every element for NaN before reducing -- work
its plain sibling does not do. The cost model charges every other value test
(count_nonzero, 1-arg where, isclose), so these must be charged too.

Two limits on that surcharge are pinned here as well, because charging it
where numpy does not run the pass is an over-bill just as real as the
under-bill above:

* **dtype.** The eleven factory-built ops route through numpy's
  ``_replace_nan``, which returns ``mask = None`` for any NON-INEXACT dtype --
  an integer or bool input runs no isnan pass at all, so it must bill exactly
  like its plain sibling. The three hand-written ops (``nanmedian``,
  ``nanpercentile``, ``nanquantile``) instead route through ``_remove_nan_1d``,
  which calls ``np.isnan`` unconditionally, so they are charged for every
  dtype.
* **symmetry.** The pass runs over the STORED orbits of a symmetric operand,
  not over its dense ``numel``, exactly like every other pass in these ops.
* **op.** ``nanmax``/``nanmin`` carry NO surcharge at any dtype (Ruling R13).
  For a plain non-object ndarray -- every operand flopscope hands them --
  NumPy takes a fast path that reduces with ``fmax``/``fmin`` and then tests
  the REDUCED OUTPUT with ``np.isnan(res).any()``; it never reaches
  ``_replace_nan``, so there is no input-sized pass to charge.

Beyond the eleven factory-built ops (`_counted_reduction`, `_counted_mean`,
`_counted_variance`), `nanmedian`, `nanpercentile`, and `nanquantile` are
hand-written functions in `_pointwise.py` with the identical defect -- they
are not reached by a factory-level `op_name.startswith("nan")` rule, so they
are covered here too.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

# (nan op, plain sibling) -- every nan* reduction on the counted surface that
# shares its plain sibling's single-array call signature AND carries the
# surcharge. ``nanmax``/``nanmin`` are deliberately absent: see
# ``_NAN_PASS_EXEMPT`` and its tests at the bottom of this module.
_PAIRS = [
    ("nansum", "sum"),
    ("nanprod", "prod"),
    ("nanmean", "mean"),
    ("nanvar", "var"),
    ("nanstd", "std"),
    ("nanargmax", "argmax"),
    ("nanargmin", "argmin"),
    ("nancumsum", "cumsum"),
    ("nancumprod", "cumprod"),
    ("nanmedian", "median"),
]

# (nan op, plain sibling, q) -- the quantile-family ops need a second
# positional argument, so they cannot share _PAIRS's single-arg call shape.
_Q_PAIRS = [
    ("nanpercentile", "percentile", 50),
    ("nanquantile", "quantile", 0.5),
]


def _billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.mark.parametrize("nan_name, plain_name", _PAIRS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128])
def test_nan_variant_costs_more_than_its_plain_sibling(nan_name, plain_name, dtype):
    """The isnan pass is real work; the nan* form must never bill the same."""
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x)) > _billed(lambda: plain_op(x))


@pytest.mark.parametrize("nan_name, plain_name", _PAIRS)
def test_nan_pass_surcharge_is_one_per_element(nan_name, plain_name):
    """The surcharge is exactly one pass over the input: numel(input)."""
    n = 10_000
    x = fnp.array(np.ones(n, dtype=np.float64))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    delta = _billed(lambda: nan_op(x)) - _billed(lambda: plain_op(x))
    # Weight-independent: the surcharge scales linearly with n at a fixed rate,
    # so doubling the input doubles the surcharge.
    x2 = fnp.array(np.ones(2 * n, dtype=np.float64))
    delta2 = _billed(lambda: nan_op(x2)) - _billed(lambda: plain_op(x2))
    assert delta > 0
    assert delta2 == 2 * delta


@pytest.mark.parametrize("nan_name, plain_name, q", _Q_PAIRS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nan_quantile_variant_costs_more_than_its_plain_sibling(
    nan_name, plain_name, q, dtype
):
    """Same invariant as the single-array family, for the quantile ops.

    No complex128 case here: numpy itself rejects complex input for both
    percentile and quantile ("a must be an array of real numbers"), so the
    registry marks both families complex_factor="illegal" -- that is a
    pre-existing, unrelated restriction, not part of this defect.
    """
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x, q)) > _billed(lambda: plain_op(x, q))


@pytest.mark.parametrize("nan_name, plain_name, q", _Q_PAIRS)
def test_nan_quantile_pass_surcharge_is_one_per_element(nan_name, plain_name, q):
    """Same one-pass-per-element invariant, for the quantile ops."""
    n = 10_000
    x = fnp.array(np.ones(n, dtype=np.float64))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    delta = _billed(lambda: nan_op(x, q)) - _billed(lambda: plain_op(x, q))
    x2 = fnp.array(np.ones(2 * n, dtype=np.float64))
    delta2 = _billed(lambda: nan_op(x2, q)) - _billed(lambda: plain_op(x2, q))
    assert delta > 0
    assert delta2 == 2 * delta


def test_plain_reductions_are_unchanged():
    """Only the nan* family moves; plain reductions keep their price."""
    x = fnp.array(np.ones(1000, dtype=np.float64))
    y = fnp.array(np.ones(2000, dtype=np.float64))
    # A plain reduction's cost still scales with its own element count only.
    assert _billed(lambda: fnp.sum(y)) > _billed(lambda: fnp.sum(x))


# ---------------------------------------------------------------------------
# The surcharge's dtype limit
# ---------------------------------------------------------------------------

#: The eleven ops built by the three reduction FACTORIES. numpy's
#: ``_replace_nan`` skips the isnan pass entirely for these when the input is
#: not inexact, so the surcharge must not apply to an integer or bool operand.
_FACTORY_PAIRS = [
    ("nansum", "sum"),
    ("nanprod", "prod"),
    ("nanmean", "mean"),
    ("nanvar", "var"),
    ("nanstd", "std"),
    ("nanargmax", "argmax"),
    ("nanargmin", "argmin"),
    ("nancumsum", "cumsum"),
    ("nancumprod", "cumprod"),
]

#: The three HAND-WRITTEN ops. ``_remove_nan_1d`` calls ``np.isnan``
#: unconditionally, so their surcharge is dtype-independent.
_HANDWRITTEN = [("nanmedian", "median", ()), ("nanpercentile", "percentile", (50,))]


@pytest.mark.parametrize("nan_name, plain_name", _FACTORY_PAIRS)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.bool_])
def test_non_inexact_input_bills_exactly_like_the_plain_sibling(
    nan_name, plain_name, dtype
):
    """No isnan pass runs for an integer/bool operand, so none may be charged.

    Measured through the real client before this gate: every one of these
    billed exactly 2.00x its plain sibling (1.25x for nanstd/nanvar) -- an
    over-bill the batch introduced on inputs that had been priced correctly.
    """
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x)) == _billed(lambda: plain_op(x))


@pytest.mark.parametrize("nan_name, plain_name, extra", _HANDWRITTEN)
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_handwritten_nan_ops_charge_the_pass_for_every_dtype(
    nan_name, plain_name, extra, dtype
):
    """``_remove_nan_1d`` runs ``np.isnan`` whatever the dtype, so the
    surcharge here is unconditional -- the factory gate must not reach it."""
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x, *extra)) > _billed(lambda: plain_op(x, *extra))


# ---------------------------------------------------------------------------
# The surcharge's symmetry limit
# ---------------------------------------------------------------------------


def _pointwise_reference(operand) -> int:
    """What one orbit-mapped pointwise pass over *operand* costs.

    ``abs`` is a plain one-per-element op that already honours symmetry, so
    its bill is the surcharge's correct value by construction -- no formula is
    restated here.
    """
    return _billed(lambda: fnp.abs(operand))


@pytest.mark.parametrize(
    "nan_name, plain_name, extra",
    [
        ("nansum", "sum", ()),
        ("nanmean", "mean", ()),
        ("nanvar", "var", ()),
        ("nanmedian", "median", ()),
        ("nanpercentile", "percentile", (50,)),
        ("nanquantile", "quantile", (0.5,)),
    ],
)
def test_surcharge_is_orbit_mapped_on_a_symmetric_operand(nan_name, plain_name, extra):
    """The isnan pass reads the stored orbits, not the dense numel.

    Every other pass in these ops is orbit-mapped (the sibling weighted-average
    multiply pass passes ``symmetry`` for exactly this reason), so charging
    this one at full ``numel`` silently erased the symmetry discount --
    measured at 320000 instead of 17710 on a rank-4 symmetric (20,)*4 operand,
    and 7200 instead of 3660 for ``nanmedian`` on a symmetric 60x60.
    """
    group = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.budget(10**15, quiet=True):
        operand = fnp.random.symmetric((60, 60), group)
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    surcharge = _billed(lambda: nan_op(operand, *extra)) - _billed(
        lambda: plain_op(operand, *extra)
    )
    assert surcharge == _pointwise_reference(operand)


def test_symmetric_surcharge_is_strictly_below_the_dense_one():
    """A symmetric operand must not pay a dense operand's isnan pass."""
    group = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.budget(10**15, quiet=True):
        symmetric = fnp.random.symmetric((60, 60), group)
    dense = fnp.array(np.arange(3600, dtype=np.float64).reshape(60, 60))
    sym_surcharge = _billed(lambda: fnp.nansum(symmetric)) - _billed(
        lambda: fnp.sum(symmetric)
    )
    dense_surcharge = _billed(lambda: fnp.nansum(dense)) - _billed(
        lambda: fnp.sum(dense)
    )
    assert 0 < sym_surcharge < dense_surcharge


# ---------------------------------------------------------------------------
# The surcharge's op limit: nanmax / nanmin (Ruling R13)
# ---------------------------------------------------------------------------

#: The two factory-built nan* reductions that never reach numpy's
#: ``_replace_nan`` and therefore carry no input-sized isnan pass at all.
_NAN_PASS_EXEMPT = [("nanmax", "max"), ("nanmin", "min")]


@pytest.mark.parametrize("nan_name, plain_name", _NAN_PASS_EXEMPT)
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.complex128, np.int32, np.bool_]
)
def test_nanmax_nanmin_bill_exactly_like_their_plain_sibling(
    nan_name, plain_name, dtype
):
    """No surcharge on ``nanmax``/``nanmin``, at float OR at integer/bool.

    numpy's fast path (`type(a) is np.ndarray and a.dtype != np.object_`,
    which is every operand flopscope hands it) reduces with
    ``np.fmax.reduce``/``np.fmin.reduce`` and then tests the REDUCED OUTPUT
    with ``np.isnan(res).any()``. The ``_replace_nan`` branch the surcharge
    models is the SLOW path, for ndarray subclasses and object arrays only.

    Charging the input-sized pass here was wrong in both directions: a 2.00x
    over-bill on float (``nanmax(float64[10000])`` billed 39998 against an
    honest ~20002) and, under the inexact-dtype gate, nothing at all for the
    output-sized pass numpy really does run on integer input. Both ops return
    to their v0.11.0 price. This test is the guard against the surcharge
    creeping back onto them.
    """
    x = fnp.array(np.ones(10_000, dtype=dtype))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x)) == _billed(lambda: plain_op(x))


@pytest.mark.parametrize("nan_name, plain_name", _NAN_PASS_EXEMPT)
def test_nanmax_nanmin_exemption_holds_on_an_axis_reduction(nan_name, plain_name):
    """The exemption is not an artefact of the full-reduction shape."""
    x = fnp.array(np.arange(5000, dtype=np.float64).reshape(1, 5000))
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(x, axis=0)) == _billed(lambda: plain_op(x, axis=0))


@pytest.mark.parametrize("nan_name, plain_name", _NAN_PASS_EXEMPT)
def test_nanmax_nanmin_exemption_holds_on_a_symmetric_operand(nan_name, plain_name):
    """Symmetric operands too -- no surcharge means none to orbit-map."""
    group = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.budget(10**15, quiet=True):
        operand = fnp.random.symmetric((60, 60), group)
    nan_op = getattr(fnp, nan_name)
    plain_op = getattr(fnp, plain_name)
    assert _billed(lambda: nan_op(operand)) == _billed(lambda: plain_op(operand))
