"""Conftest that monkeypatches numpy with flopscope for compatibility testing.

This lets us run NumPy's own test suite against flopscope to verify
that our interface matches NumPy's. Tests that fail due to known
divergences are listed in xfails.py.

Key trick: before patching numpy, fnp freeze a copy of the original
numpy module and rebind flopscope's internal `_np` references to it.
This breaks the infinite recursion that would otherwise occur when
flopscope functions call _np.func() → numpy.func() → fnp.func() → ...
"""

import fnmatch
import importlib
import sys
import types

import numpy as np
import pytest

from flopscope._budget import _reset_global_default, budget_reset
from flopscope._registry import REGISTRY

from .xfails import XFAIL_PATTERNS

# Ensure direct imports of `tests.numpy_compat.conftest` reuse the same module
# instance pytest loaded as a conftest plugin, rather than executing this file
# a second time after NumPy has already been patched.
sys.modules.setdefault("tests.numpy_compat.conftest", sys.modules[__name__])


class _NonDescriptor:
    """Callable wrapper that prevents Python descriptor auto-binding.

    Python functions implement ``__get__`` so when stored as a class
    attribute and accessed via ``self.func()``, Python auto-binds
    ``self`` as the first positional argument. C built-in functions
    don't do this.  Wrapping in ``_NonDescriptor`` makes our Python
    replacements behave like C built-ins for attribute access.
    """

    def __init__(self, fn):
        self._fn = fn
        # Copy key attributes so introspection (numpy.ma docstring
        # parsing, inspect.signature, etc.) sees the original metadata.
        self.__name__ = getattr(fn, "__name__", "")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", None)  # pyright: ignore[reportAttributeAccessIssue]
        self.__signature__ = getattr(fn, "__signature__", None)
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._fn, name)


# Functions fnp monkeypatch onto numpy
_PATCHED: dict[str, object] = {}

# flopscope modules whose _np fnp rebind, with their originals
_REBOUND: dict[str, object] = {}

# All flopscope submodules that import numpy as _np
_FLOPSCOPE_MODULES_WITH_NP = [
    "flopscope._ndarray",
    "flopscope._pointwise",
    "flopscope._array_ops",
    "flopscope._sorting_ops",
    "flopscope._counting_ops",
    "flopscope._einsum",
    "flopscope._polynomial",
    "flopscope._unwrap",
    "flopscope._window",
    "flopscope._version_check",
    "flopscope.__init__",
    "flopscope.numpy.fft._transforms",
    "flopscope.numpy.fft._free",
    "flopscope.numpy.linalg._decompositions",
    "flopscope.numpy.linalg._properties",
    "flopscope.numpy.linalg._solvers",
    "flopscope.numpy.linalg._compound",
    "flopscope.numpy.linalg._svd",
    "flopscope.numpy.linalg._aliases",
    "flopscope.numpy.random",
]

# Modules that also import numpy.random as _npr
_FLOPSCOPE_MODULES_WITH_NPR = [
    "flopscope.numpy.random",
]

# Internal modules that import numpy as `np` instead of `_np`. These sit on
# hot paths used by constructor-side symmetry tagging and SymmetricTensor
# creation, so they also need the frozen numpy copy during compat patching.
_FLOPSCOPE_MODULES_WITH_PLAIN_NP = [
    "flopscope._symmetry_utils",
    "flopscope._symmetric",
]


def _snapshot_numpy_module(source_np):
    """Create an unfrozen snapshot copy of numpy and key submodules."""
    frozen = types.ModuleType("_frozen_numpy")
    frozen.__dict__.update(source_np.__dict__)

    for submod_name in ("linalg", "fft", "random"):
        original_submod = getattr(source_np, submod_name)
        frozen_submod = types.ModuleType(f"_frozen_numpy.{submod_name}")
        frozen_submod.__dict__.update(original_submod.__dict__)
        setattr(frozen, submod_name, frozen_submod)

    return frozen


_ORIGINAL_NUMPY = _snapshot_numpy_module(np)


def _current_flopscope_numpy():
    """Return the `flopscope.numpy` module, reimporting if needed.

    The patch loop resolves REGISTRY names via getattr() on the returned
    module. After the JAX-style rebrand, registry names like
    ``zeros_like``, ``einsum``, ``linalg.outer`` live under
    ``flopscope.numpy.*``, not on the top-level ``flopscope`` module.
    Pointing at the wrong module makes every lookup silently
    AttributeError — the entire harness becomes a no-op.
    """
    mod = sys.modules.get("flopscope.numpy")
    if mod is None:
        mod = importlib.import_module("flopscope.numpy")
    return mod


def _freeze_numpy():
    """Create a frozen copy of numpy that won't be affected by patching.

    Returns a module whose attributes are snapshots of numpy's current
    functions. Submodules (linalg, fft, random) are also frozen.
    """
    return _snapshot_numpy_module(_ORIGINAL_NUMPY)


def _rebind_flopscope_np(frozen_np):
    """Replace _np in all flopscope modules with the frozen copy."""
    for mod_name in _FLOPSCOPE_MODULES_WITH_NP:
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        if mod is not None and hasattr(mod, "_np"):
            _REBOUND[mod_name] = mod._np  # pyright: ignore[reportAttributeAccessIssue]
            mod._np = frozen_np  # pyright: ignore[reportAttributeAccessIssue]
    # Also rebind _npr (numpy.random) in modules that use it
    for mod_name in _FLOPSCOPE_MODULES_WITH_NPR:
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        if mod is not None and hasattr(mod, "_npr"):
            _REBOUND[mod_name + "._npr"] = mod._npr  # pyright: ignore[reportAttributeAccessIssue]
            mod._npr = frozen_np.random  # pyright: ignore[reportAttributeAccessIssue]
    for mod_name in _FLOPSCOPE_MODULES_WITH_PLAIN_NP:
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        if mod is not None and hasattr(mod, "np"):
            _REBOUND[mod_name + ".np"] = mod.np  # pyright: ignore[reportAttributeAccessIssue]
            mod.np = frozen_np  # pyright: ignore[reportAttributeAccessIssue]


def _restore_flopscope_np():
    """Restore original _np references in flopscope modules."""
    for key, original in _REBOUND.items():
        if key.endswith("._npr"):
            mod_name = key[: -len("._npr")]
            mod = sys.modules.get(mod_name)
            if mod is not None:
                mod._npr = original  # pyright: ignore[reportAttributeAccessIssue]
        elif key.endswith(".np"):
            mod_name = key[: -len(".np")]
            mod = sys.modules.get(mod_name)
            if mod is not None:
                mod.np = original  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mod = sys.modules.get(key)
            if mod is not None:
                mod._np = original  # pyright: ignore[reportAttributeAccessIssue]
    _REBOUND.clear()


def _patch_numpy():
    """Replace numpy functions with flopscope equivalents.

    Patches all non-blacklisted functions from the registry, including
    ufuncs, custom ops, submodule functions, and free ops. The frozen
    numpy copy prevents infinite recursion.
    """
    fnp = _current_flopscope_numpy()

    for name, meta in REGISTRY.items():
        cat = meta["category"]
        if cat == "blacklisted":
            continue

        # Resolve the flopscope function
        parts = name.split(".")
        try:
            if len(parts) == 1:
                we_fn = getattr(fnp, name)
            elif len(parts) == 2:
                submod = getattr(fnp, parts[0])
                we_fn = getattr(submod, parts[1])
            else:
                continue
        except AttributeError:
            continue

        # Skip ufuncs — our replacements are plain functions and lack
        # .reduce/.accumulate/.outer/.nargs/etc. which tests check at
        # collection time.
        try:
            if len(parts) == 1:
                np_obj = getattr(np, name, None)
            elif len(parts) == 2:
                np_obj = getattr(getattr(np, parts[0], None), parts[1], None)
            else:
                np_obj = None
            if isinstance(np_obj, np.ufunc):
                continue
        except (AttributeError, TypeError):
            pass

        # Skip functions where flopscope delegates to a different numpy function
        # than the one being patched (e.g., fnp.linalg.outer → np.outer, not
        # np.linalg.outer). Patching causes collection-time errors in tests that
        # check the real np.linalg.outer's behaviour at class-definition time.
        _SKIP_PATCH = {
            "linalg.outer",
            # Python functions auto-bind self via descriptor protocol when
            # used as class attributes; C built-in functions and bound
            # methods don't. Skip patching these to avoid "multiple values
            # for keyword argument" errors in tests.
            "array",
            "arange",
            "random.randint",
            "random.shuffle",
        }
        if name in _SKIP_PATCH:
            continue

        # Patch numpy
        try:
            if len(parts) == 1:
                if hasattr(np, name):
                    _PATCHED[name] = getattr(np, name)
                    setattr(np, name, we_fn)
            elif len(parts) == 2:
                np_submod = getattr(np, parts[0])
                if hasattr(np_submod, parts[1]):
                    _PATCHED[name] = getattr(np_submod, parts[1])
                    setattr(np_submod, parts[1], we_fn)
        except (AttributeError, TypeError):
            continue


def _unpatch_numpy():
    """Restore original numpy functions."""
    for name, original in _PATCHED.items():
        parts = name.split(".")
        if len(parts) == 1:
            setattr(np, name, original)
        elif len(parts) == 2:
            setattr(getattr(np, parts[0]), parts[1], original)
    _PATCHED.clear()


@pytest.fixture(autouse=True)
def reset_budget():
    """Reset global budget between tests to avoid cross-test leakage."""
    _reset_global_default()
    budget_reset()
    yield
    _reset_global_default()
    budget_reset()


def pytest_configure(config):
    """Freeze numpy, rebind flopscope internals, patch, register global plugins."""
    frozen = _freeze_numpy()
    _rebind_flopscope_np(frozen)
    _patch_numpy()
    if not config.pluginmanager.has_plugin("flopscope-immutability-xfail"):
        config.pluginmanager.register(
            _ImmutabilityXfailPlugin(), "flopscope-immutability-xfail"
        )


def pytest_unconfigure(config):
    """Restore everything."""
    _unpatch_numpy()
    _restore_flopscope_np()


def pytest_collection_modifyitems(config, items):
    """Mark known-divergent tests as xfail."""
    for item in items:
        node_id = item.nodeid
        for pattern, reason in XFAIL_PATTERNS.items():
            if fnmatch.fnmatch(node_id, pattern) or pattern in node_id:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break


# ---------------------------------------------------------------------------
# Immutability divergence -> xfail (runtime, exception-matched)
# ---------------------------------------------------------------------------
# flopscope arrays are immutable by design (competition rule #immutable-arrays):
# __setitem__, the in-place operators, and in-place sort/partition raise. Many
# of NumPy's own tests mutate arrays in place — often via an ``out=`` scratch
# buffer or ``arr[...] = `` setup — so when run against flopscope they fail with
# our immutability TypeError/ValueError. That is a deliberate, documented
# divergence, not a parity bug. Convert exactly those failures to xfail at
# report time. Matching the raised exception (rather than enumerating test ids in
# .xfails) keeps this robust across NumPy versions, which reshuffle which tests
# happen to mutate. The dedicated surface-parity test pins the immutable method
# *set*; this only suppresses NumPy tests that exercise that documented behaviour.

# Contiguous phrase present in every immutability guard message
# (src/flopscope/_ndarray.py). Specific enough not to match NumPy's own
# read-only errors (e.g. "assignment destination is read-only").
_IMMUTABLE_SENTINEL = "flopscope arrays are immutable"


def _hit_immutability_guard(excinfo) -> bool:
    """True if the raised exception chain is flopscope's immutability guard."""
    exc = getattr(excinfo, "value", None)
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (TypeError, ValueError)) and _IMMUTABLE_SENTINEL in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class _ImmutabilityXfailPlugin:
    """Reclassify in-place-mutation failures as xfail (immutability is by design).

    Registered as a global plugin in :func:`pytest_configure` rather than left as
    a bare conftest hook so it also fires for the ``--pyargs`` NumPy tests: a
    per-item runtest hook defined directly in a conftest only runs for items
    inside that conftest's directory, and the borrowed NumPy tests live in
    site-packages (the same ``--pyargs`` scoping that makes session hooks like
    ``pytest_configure`` work but per-item ones silently no-op).
    """

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        # ``call`` covers in-test mutation; ``setup`` covers fixtures that mutate
        # while preparing operands (reported as an ERROR, not a FAILURE).
        if (
            report.when in ("setup", "call")
            and report.failed
            and _hit_immutability_guard(call.excinfo)
        ):
            report.outcome = "skipped"
            report.wasxfail = (
                "flopscope arrays are immutable by design (#immutable-arrays); "
                "NumPy's test mutates in place"
            )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector):
        # Some NumPy test modules mutate a (patched) flopscope array at *import*
        # time — e.g. test_linalg builds strided test cases with ``xi[...] = x``.
        # Under immutability that raises during collection, so the whole module
        # is un-collectable and individual tests can never be reached to xfail.
        # Convert that specific collection failure into a module-level skip
        # (same by-design rationale as the runtime hook). The skip is loud in
        # ``-rs`` output, so the coverage trade-off stays visible.
        outcome = yield
        report = outcome.get_result()
        if report.outcome == "failed" and _IMMUTABLE_SENTINEL in str(report.longrepr):
            report.outcome = "skipped"
            report.longrepr = (
                str(getattr(collector, "path", collector.nodeid)),
                None,
                "Skipped: flopscope arrays are immutable (#immutable-arrays); "
                "this NumPy test module mutates an array at import time",
            )
