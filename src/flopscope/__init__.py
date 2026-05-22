"""Flopscope: FLOP-counting numpy primitives for the Mechanistic Estimation Challenge.

The public API is structured JAX-style:

- ``flopscope`` (this module) exports flopscope-specific primitives: budget
  contexts, configuration, symmetric tensors, the unified
  :class:`SymmetryGroup`, errors, and the :class:`FlopscopeArray` type.
- ``flopscope.numpy`` exports the full counted numpy-shaped surface
  (``einsum``, ``array``, ``linspace``, ``linalg``, ``fft``, ``random``, ...).
  Attributes not explicitly implemented there fall back to raw ``numpy``.
- ``flopscope.accounting`` exposes analytical cost helpers
  (``einsum_cost``, ``pointwise_cost``, ``reduction_cost``, ...).
- ``flopscope.stats`` hosts statistical-distribution primitives (not a
  numpy submodule; closer in spirit to scipy.stats).

Usage::

    import flopscope as flops
    import flopscope.numpy as fnp

    flops.configure(symmetry_warnings=False)
    with flops.BudgetContext(flop_budget=1_000_000) as budget:
        W = fnp.array(weight_matrix)
        h = fnp.einsum('ij,j->i', W, x)
        h = fnp.maximum(h, 0)
        print(budget.summary())
"""

import importlib as _importlib

import numpy as _np

from flopscope._registry import REGISTRY_META as _REGISTRY_META

__version__ = f"0.2.0+np{_np.__version__}"
__numpy_version__ = _np.__version__
__numpy_pinned__ = _REGISTRY_META["numpy_version"]
__numpy_supported__ = _REGISTRY_META.get("numpy_supported", ">=2.0.0,<2.5.0")

from flopscope._version_check import check_numpy_version as _check_numpy_version

_check_numpy_version(__numpy_supported__)

# --- Symmetry-aware accumulation cost (einsum + reduction) ---
from flopscope._accumulation import (  # noqa: F401,E402
    AccumulationCost,
    ComponentCost,
    RegimeStep,
    einsum_accumulation_cost,
    reduction_accumulation_cost,
)

# --- Budget and diagnostics ---
from flopscope._budget import (  # noqa: F401,E402
    BudgetContext,
    OpRecord,
    budget,
    budget_reset,
    budget_summary_dict,
    namespace,
)
from flopscope._config import configure  # noqa: F401,E402
from flopscope._display import budget_live, budget_summary  # noqa: F401,E402

# --- Array type (flopscope-specific) ---
from flopscope._ndarray import FlopscopeArray  # noqa: F401,E402

# --- Path optimization types ---
from flopscope._opt_einsum import PathInfo, StepInfo  # noqa: F401,E402

# --- Symmetry (post-#51 unified surface) ---
from flopscope._perm_group import SymmetryGroup  # noqa: F401,E402

# --- Symmetric tensor ---
from flopscope._symmetric import (  # noqa: F401,E402
    SymmetricTensor,
    as_symmetric,
    is_symmetric,
    symmetrize,
)

# --- Errors ---
from flopscope.errors import (  # noqa: F401,E402
    BudgetExhaustedError,
    FlopscopeError,
    FlopscopeWarning,
    NoBudgetContextError,
    SymmetryError,
    SymmetryLossWarning,
    TimeExhaustedError,
    UnsupportedFunctionError,
)


def einsum_clear_caches() -> None:
    """Clear all flopscope einsum-related LRU caches.

    Clears both the einsum path cache (consulted by ``fnp.einsum`` and
    ``fnp.einsum_path``) and the einsum accumulation-cost cache (consulted
    by ``fnp.einsum`` and ``flopscope.einsum_accumulation_cost``).

    Useful when benchmarking cold-call latency. ``fnp.clear_einsum_cache``
    still exists and clears only the path cache.
    """
    from flopscope._accumulation._cache import _accumulation_cache
    from flopscope._einsum import _path_cache

    _path_cache.cache_clear()
    _accumulation_cache.cache_clear()


def einsum_cache_info() -> dict:
    """Return cache statistics for the einsum path + accumulation caches.

    Returns
    -------
    dict
        ``{"path": CacheInfo, "accumulation": CacheInfo}`` where each value
        is a standard ``functools.lru_cache`` info tuple with ``hits``,
        ``misses``, ``maxsize``, and ``currsize``.
    """
    from flopscope._accumulation._cache import _accumulation_cache
    from flopscope._einsum import _path_cache

    return {
        "path": _path_cache.cache_info(),
        "accumulation": _accumulation_cache.cache_info(),
    }


def reduction_clear_cache() -> None:
    """Clear the reduction accumulation-cost cache.

    Consulted by ``flopscope.reduction_accumulation_cost`` and the
    ``np.ufunc.reduce`` family wrappers (``fnp.sum``, ``fnp.mean``,
    ``fnp.median``, etc.).
    """
    from flopscope._accumulation._cache import _reduction_cache

    _reduction_cache.cache_clear()


def reduction_cache_info():
    """Return cache statistics for the reduction accumulation-cost cache.

    Returns a standard ``functools.lru_cache`` info tuple with ``hits``,
    ``misses``, ``maxsize``, and ``currsize``.
    """
    from flopscope._accumulation._cache import _reduction_cache

    return _reduction_cache.cache_info()


def clear_cache() -> None:
    """Clear all flopscope LRU caches (einsum + reduction).

    Convenience aggregate over :func:`einsum_clear_caches` and
    :func:`reduction_clear_cache`. Use the per-domain variants if you
    only need to invalidate one cache.
    """
    einsum_clear_caches()
    reduction_clear_cache()


def tier2_reduction_cost(a, axis=None, *, dense_per_output_cost=None):
    """Cost for selection-style reductions (median, percentile, quantile).

    Returns ``num_output_orbits × dense_per_output_cost`` — one partition
    pass per unique output cell. Under symmetry, orbit-shared output
    cells share the partition pass; ``num_output_orbits`` is computed
    from the input's declared symmetry.

    Parameters
    ----------
    a : numpy.ndarray or SymmetricTensor
        Input array. ``SymmetricTensor`` inputs contribute their declared
        symmetry to the orbit count.
    axis : int, tuple of int, or None
        Axes being reduced. ``None`` (default) reduces all axes.
    dense_per_output_cost : int, optional
        The dense per-output cost. If ``None`` (the default), it is set
        to the product of the reduced axes' lengths — i.e., one
        partition pass per output cell, which is how ``fnp.median``,
        ``fnp.percentile``, and ``fnp.quantile`` are charged. Pass an
        explicit value to model a custom Tier-2 selection cost.

    Returns
    -------
    int
        Total flops charged.
    """
    import math as _math

    import numpy as _np

    from flopscope._pointwise import _tier2_reduction_cost

    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)

    if dense_per_output_cost is None:
        if axis is None:
            dense_per_output_cost = _math.prod(a.shape) if a.shape else 1
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            dense_per_output_cost = _math.prod(a.shape[i] for i in axes)

    return _tier2_reduction_cost(a, axis, dense_per_output_cost)


_LAZY_SUBMODULES = frozenset({"numpy", "accounting", "stats"})

__all__ = [
    "AccumulationCost",
    "BudgetContext",
    "BudgetExhaustedError",
    "ComponentCost",
    "FlopscopeArray",
    "FlopscopeError",
    "FlopscopeWarning",
    "NoBudgetContextError",
    "OpRecord",
    "PathInfo",
    "RegimeStep",
    "StepInfo",
    "SymmetricTensor",
    "SymmetryError",
    "SymmetryGroup",
    "SymmetryLossWarning",
    "TimeExhaustedError",
    "UnsupportedFunctionError",
    "__numpy_pinned__",
    "__numpy_supported__",
    "__numpy_version__",
    "__version__",
    "accounting",
    "as_symmetric",
    "budget",
    "budget_live",
    "budget_reset",
    "budget_summary",
    "budget_summary_dict",
    "clear_cache",
    "configure",
    "einsum_accumulation_cost",
    "einsum_cache_info",
    "einsum_clear_caches",
    "is_symmetric",
    "namespace",
    "numpy",
    "reduction_accumulation_cost",
    "reduction_cache_info",
    "reduction_clear_cache",
    "stats",
    "symmetrize",
    "tier2_reduction_cost",
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"flopscope.{name}")
        globals()[name] = module
        return module
    raise AttributeError(
        f"flopscope does not provide {name!r}. "
        f"Numpy-shaped operations live under 'flopscope.numpy' "
        f"(try `import flopscope.numpy as fnp; fnp.{name}`)."
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
