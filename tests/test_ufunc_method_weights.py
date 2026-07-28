"""Synthesized ``<ufunc>.<method>`` op names must inherit the base ufunc's weight.

Billing composes a per-element cost with a per-op weight. Op names for ufunc
methods are built at billing time as f"{ufunc.__name__}.{method}", so they are
never registry rows and a missing entry silently resolves to the neutral 1.0 --
making a heavier ufunc cost the same as ``add``.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp
from flopscope._weights import _UFUNC_METHOD_SUFFIXES, get_weight


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        before = ctx.flops_used
        fn()
        return ctx.flops_used - before


@pytest.mark.parametrize("base", ["exp", "log", "sin", "power", "hypot"])
@pytest.mark.parametrize("suffix", _UFUNC_METHOD_SUFFIXES)
def test_method_name_inherits_base_weight(base, suffix):
    assert get_weight(base + suffix) == get_weight(base)


def test_alias_then_method_resolves():
    assert get_weight("pow.at") == get_weight("power")


def test_unknown_dotted_name_still_defaults():
    assert get_weight("totally.bogus") == 1.0


@pytest.mark.parametrize("name", ["exp", "log", "sin"])
def test_at_costs_the_same_as_the_elementwise_equivalent(name):
    """np.<f>.at over N cells must cost what fnp.<f> costs over N elements."""
    n = 4096
    ufunc = getattr(np, name)
    counted = getattr(fnp, name)
    target = fnp.asarray(np.random.rand(n) + 1.0)
    at_cost = billed(lambda: ufunc.at(target, np.arange(n)))
    honest = billed(lambda: counted(fnp.asarray(np.random.rand(n) + 1.0)))
    assert at_cost == honest


def test_outer_costs_the_same_as_the_elementwise_equivalent():
    n = 128
    v = fnp.asarray(np.random.rand(n) + 1.0)
    outer_cost = billed(lambda: np.power.outer(v, v))
    honest = billed(lambda: fnp.power(fnp.asarray(np.random.rand(n, n) + 1.0), 2.0))
    assert outer_cost == honest


def test_unit_weight_ufuncs_are_unaffected():
    assert get_weight("add.at") == get_weight("add") == 1.0
