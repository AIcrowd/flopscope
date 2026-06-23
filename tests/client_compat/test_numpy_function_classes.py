"""Wrap numpy's FUNCTION-surface test classes so they run inside this conftest's
scope (budget context + numpy->client function patch active), exercising the
CLIENT. The previous --pyargs approach was a no-op: --pyargs tests live in
site-packages, outside this conftest's dir scope, so the autouse budget fixture
never fired and ops fell through to native numpy.

Import numpy's test modules at module level (before pytest_configure fires the
patch). For each module we create thin subclasses of every Test* class; pytest
collects them as local items here, so the _server + _fresh_connection_and_budget
autouse fixtures apply and np.<fn> routes to the client during the test body.

NOTE (coverage caveat): this captures numpy's CLASS-based tests only (the large
majority — umath 53, numeric 35, linalg 27, etc. class-level).
A small number of module-level `def test_*` functions are not wrapped.
"""

from __future__ import annotations

import inspect

from numpy._core.tests import test_numeric as _numeric
from numpy._core.tests import test_umath as _umath
from numpy.fft.tests import test_helper as _fft_helper
from numpy.fft.tests import test_pocketfft as _pocketfft
from numpy.linalg.tests import test_linalg as _linalg
from numpy.polynomial.tests import test_polynomial as _poly
from numpy.random.tests import test_random as _random

_SOURCE_MODULES = (
    _umath,
    _numeric,
    _linalg,
    _pocketfft,
    _fft_helper,
    _poly,
    _random,
)


def _make_subclasses() -> dict:
    made: dict = {}
    for mod in _SOURCE_MODULES:
        prefix = mod.__name__.rsplit(".", 1)[-1]  # e.g. test_umath
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if not name.startswith("Test"):
                continue
            if cls.__module__ != mod.__name__:
                continue  # skip imported-not-defined classes
            new_name = f"{name}__{prefix}__Client"
            made[new_name] = type(new_name, (cls,), {})
    return made


globals().update(_make_subclasses())
