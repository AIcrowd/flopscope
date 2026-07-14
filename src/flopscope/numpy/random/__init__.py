"""Counted wrappers for ``numpy.random``.

Policy (issue #18):

* **Module-level samplers** (``randn``, ``normal``, ``uniform``,
  ``choice``, ``shuffle``, ``bytes``, ...): same semantics as numpy plus
  FLOP accounting. Most deduct ``numel(output)`` FLOPs;
  ``permutation``/``shuffle``/``choice(replace=False)`` deduct
  ``n * ceil(log2(n))``; ``bytes(n)`` deducts ``n``. No deprecation —
  flopscope mirrors numpy's runtime behavior.

* **``default_rng(seed)``** returns a counted ``Generator`` subclass
  (``_CountedGenerator``) whose sampler methods deduct FLOPs and return
  ``FlopscopeArray``. The constructor itself costs 0 FLOPs.

* **``RandomState(seed)``** is a counted subclass of
  ``numpy.random.RandomState`` (``_CountedRandomState``) — same shape as
  the modern Generator path, legacy method names. Constructor is 0 FLOPs.

* **Configuration / state methods** (``seed``, ``get_state``,
  ``set_state``, ``Generator.spawn``, ``Generator.bit_generator``)
  are free.

* **``__getattr__`` fallback**: bit-generator classes
  (``BitGenerator``, ``MT19937``, ``PCG64``, ``PCG64DXSM``, ``Philox``,
  ``SFC64``) pass through unchanged; everything else raises
  ``AttributeError``. New numpy methods are invisible to user code until
  they are explicitly added to the registry — ``scripts/numpy_audit.py
  --ci`` flags this on every numpy version bump.

* **``SeedSequence``** passes through unchanged (pure utility, no math).
"""

from __future__ import annotations

import builtins as _builtins
import inspect as _inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as _np
import numpy.random as _npr
from numpy.random import SeedSequence

# Public exports below; concrete counted classes pulled in lazily to avoid
# circular import with _counted_classes.py.
from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._flops import _ceil_log2, sort_cost  # noqa: F401
from flopscope._ndarray import FlopscopeArray
from flopscope._perm_group import SymmetryGroup
from flopscope._validation import require_budget

if TYPE_CHECKING:
    from flopscope.numpy.random._counted_classes import _CountedGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_size(*dims, size=None):
    """Compute the total number of output elements."""
    if size is not None:
        if isinstance(size, (int, _np.integer)):
            return int(size)
        return int(_np.prod(size))
    if dims:
        return int(_np.prod(dims))
    return 1


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


@_counted_wrapper
def _counted_sampler(
    np_func: Callable[..., Any],
    op_name: str,
) -> Callable[..., Any]:
    """Factory for simple samplers: cost = numel(output).

    Passes all args/kwargs through transparently to numpy. Derives the
    output size from the actual result to correctly handle ``size`` passed
    as either a positional or keyword argument.
    """

    @_counted_wrapper
    def wrapper(*args, **kwargs):
        budget = require_budget()
        result = np_func(*args, **kwargs)
        # This factory wraps many distributions with heterogeneous output
        # dtypes (float64 for normal/exponential/..., int64 for
        # poisson/binomial/randint/...); read the already-computed result's
        # actual dtype rather than guessing a per-distribution default.
        if isinstance(result, _np.ndarray):
            n = _builtins.max(result.size, 1)
            dtypes: tuple = (result.dtype,)
        elif isinstance(result, (int, float, _np.integer, _np.floating)):
            n = 1
            dtypes = (_np.dtype(type(result)),)
        else:
            n = 1
            dtypes = ()
        with budget.deduct(
            op_name, flop_cost=n, subscripts=None, shapes=((n,),), dtypes=dtypes
        ):
            pass  # numpy already executed
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


@_counted_wrapper
def _counted_dims_sampler(
    np_func: Callable[..., Any],
    op_name: str,
) -> Callable[..., Any]:
    """Factory for rand/randn that take *dims instead of size=."""

    @_counted_wrapper
    def wrapper(*dims):
        budget = require_budget()
        n = int(_np.prod(dims)) if dims else 1
        cost = _builtins.max(n, 1)
        # rand/randn have no dtype= param and always return float64.
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=((n,),),
            dtypes=(_np.dtype(_np.float64),),
        ):
            if dims:
                result = np_func(*dims)
            else:
                result = np_func()
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Free (configuration) functions
# ---------------------------------------------------------------------------


def default_rng(seed: Any = None) -> _CountedGenerator:
    """Construct a flopscope-counted ``Generator`` (canonical entry point).

    The returned object is a subclass of ``numpy.random.Generator``. Its
    sampler methods (``standard_normal``, ``normal``, ``uniform``,
    ``integers``, ``choice``, ``shuffle``, ...) deduct FLOPs from the active
    ``BudgetContext`` and return ``FlopscopeArray``. The constructor itself
    costs 0 FLOPs.

    Parameters
    ----------
    seed : int, sequence of int, ``SeedSequence``, ``BitGenerator``, or None
        Same semantics as ``numpy.random.default_rng``.

    Returns
    -------
    _CountedGenerator
        A counted ``Generator`` subclass. ``isinstance(rng, np.random.Generator)``
        holds, so it works anywhere a numpy ``Generator`` is expected.

    Examples
    --------
    Basic usage::

        import flopscope.numpy as fnp
        from flopscope import BudgetContext

        with BudgetContext(flop_budget=10_000) as b:
            rng = fnp.random.default_rng(42)
            x = rng.standard_normal((100,))
        # b.flops_used == 1600 under empirical weights (100 × weight=16)
        # type(x).__name__ == 'FlopscopeArray'

    Pickle / copy round-trips preserve counting::

        import pickle
        rng = fnp.random.default_rng(42)
        revived = pickle.loads(pickle.dumps(rng))
        with BudgetContext(flop_budget=10_000):
            revived.standard_normal(10)   # still counted

    Spawn returns counted children for parallel work::

        rng = fnp.random.default_rng(42)
        for child in rng.spawn(4):
            with BudgetContext(flop_budget=10_000):
                child.standard_normal(100)  # each child is independently counted

    Cross-API parity: same physical op, same FLOPs::

        # Under empirical weights, all three charge 1600 FLOPs:
        with BudgetContext() as b: fnp.random.randn(100)
        with BudgetContext() as b: fnp.random.default_rng(0).standard_normal(100)
        with BudgetContext() as b: fnp.random.RandomState(0).randn(100)

    See Also
    --------
    fnp.random.RandomState : Counted legacy ``RandomState`` subclass.
    fnp.random.randn : Module-level sampler (uses numpy's global state).

    Notes
    -----
    Closes flopscope#18. The ``_CountedGenerator`` subclass uses an
    ``__getattribute__`` gate that consults a registry-derived allowlist;
    new numpy methods that haven't been added to the registry raise
    ``UnsupportedFunctionError`` rather than silently bypassing FLOP
    accounting.
    """
    from flopscope.numpy.random._counted_classes import _CountedGenerator

    raw = _npr.default_rng(seed)
    return _CountedGenerator(raw.bit_generator)


def seed(seed: int | None = None) -> None:
    """Seed numpy's legacy global RNG. Cost: 0 FLOPs."""
    _npr.seed(seed)


def get_state(legacy: bool = True) -> dict[str, Any] | tuple[Any, ...]:
    """Return numpy's global RNG state. Cost: 0 FLOPs."""
    return _npr.get_state(legacy=legacy)


def set_state(state: dict[str, Any] | tuple[Any, ...]) -> None:
    """Set numpy's global RNG state. Cost: 0 FLOPs."""
    _npr.set_state(state)


__all__ = [
    "seed",
    "get_state",
    "set_state",
    "default_rng",
    "Generator",
    "RandomState",
    "SeedSequence",
    "rand",
    "randn",
    "normal",
    "uniform",
    "standard_normal",
    "standard_exponential",
    "exponential",
    "poisson",
    "binomial",
    "geometric",
    "hypergeometric",
    "negative_binomial",
    "logseries",
    "power",
    "pareto",
    "rayleigh",
    "standard_cauchy",
    "standard_t",
    "standard_gamma",
    "weibull",
    "zipf",
    "gumbel",
    "laplace",
    "logistic",
    "lognormal",
    "vonmises",
    "wald",
    "triangular",
    "chisquare",
    "noncentral_chisquare",
    "noncentral_f",
    "f",
    "beta",
    "gamma",
    "multinomial",
    "multivariate_normal",
    "dirichlet",
    "randint",
    "random",
    "random_sample",
    "ranf",
    "sample",
    "permutation",
    "shuffle",
    "choice",
    "symmetric",
    "bytes",
]


# ---------------------------------------------------------------------------
# Counted class re-exports (issue #18)
# ---------------------------------------------------------------------------
from flopscope.numpy.random._counted_classes import (  # noqa: E402
    _CountedGenerator as Generator,
)
from flopscope.numpy.random._counted_classes import (  # noqa: E402
    _CountedRandomState as RandomState,
)

# ---------------------------------------------------------------------------
# Dims-based samplers (rand, randn)
# ---------------------------------------------------------------------------

rand = _counted_dims_sampler(_npr.rand, "random.rand")
randn = _counted_dims_sampler(_npr.randn, "random.randn")


# ---------------------------------------------------------------------------
# Simple samplers (cost = numel(output))
# ---------------------------------------------------------------------------

normal = _counted_sampler(_npr.normal, "random.normal")
standard_normal = _counted_sampler(_npr.standard_normal, "random.standard_normal")


@_counted_wrapper
def uniform(low=0.0, high=1.0, size=None):
    """Counted version of ``numpy.random.uniform``. Cost: 3*numel(output).

    Beyond the per-element draw (numel), uniform applies the affine map
    ``low + (high - low) * U`` per element — one multiply + one add (FMA=2) —
    so the honest cost is ``3 * numel(output)``.
    """
    budget = require_budget()
    result = _npr.uniform(low, high, size)
    n = _builtins.max(result.size, 1) if isinstance(result, _np.ndarray) else 1
    # np.random.uniform has no dtype= param and always returns float64.
    with budget.deduct(
        "random.uniform",
        flop_cost=3 * n,
        subscripts=None,
        shapes=((n,),),
        dtypes=(_np.dtype(_np.float64),),
    ):
        pass  # numpy already executed
    return result


try:
    uniform.__signature__ = _inspect.signature(_npr.uniform)  # pyright: ignore[reportFunctionMemberAccess]
except (ValueError, TypeError):
    pass
standard_exponential = _counted_sampler(
    _npr.standard_exponential, "random.standard_exponential"
)
exponential = _counted_sampler(_npr.exponential, "random.exponential")
poisson = _counted_sampler(_npr.poisson, "random.poisson")
binomial = _counted_sampler(_npr.binomial, "random.binomial")
geometric = _counted_sampler(_npr.geometric, "random.geometric")
hypergeometric = _counted_sampler(_npr.hypergeometric, "random.hypergeometric")
negative_binomial = _counted_sampler(_npr.negative_binomial, "random.negative_binomial")
logseries = _counted_sampler(_npr.logseries, "random.logseries")
power = _counted_sampler(_npr.power, "random.power")
pareto = _counted_sampler(_npr.pareto, "random.pareto")
rayleigh = _counted_sampler(_npr.rayleigh, "random.rayleigh")
standard_cauchy = _counted_sampler(_npr.standard_cauchy, "random.standard_cauchy")
standard_t = _counted_sampler(_npr.standard_t, "random.standard_t")
standard_gamma = _counted_sampler(_npr.standard_gamma, "random.standard_gamma")
weibull = _counted_sampler(_npr.weibull, "random.weibull")
zipf = _counted_sampler(_npr.zipf, "random.zipf")
gumbel = _counted_sampler(_npr.gumbel, "random.gumbel")
laplace = _counted_sampler(_npr.laplace, "random.laplace")
logistic = _counted_sampler(_npr.logistic, "random.logistic")
lognormal = _counted_sampler(_npr.lognormal, "random.lognormal")
vonmises = _counted_sampler(_npr.vonmises, "random.vonmises")
wald = _counted_sampler(_npr.wald, "random.wald")
triangular = _counted_sampler(_npr.triangular, "random.triangular")
chisquare = _counted_sampler(_npr.chisquare, "random.chisquare")
noncentral_chisquare = _counted_sampler(
    _npr.noncentral_chisquare, "random.noncentral_chisquare"
)
noncentral_f = _counted_sampler(_npr.noncentral_f, "random.noncentral_f")
f = _counted_sampler(_npr.f, "random.f")
beta = _counted_sampler(_npr.beta, "random.beta")
gamma = _counted_sampler(_npr.gamma, "random.gamma")
multinomial = _counted_sampler(_npr.multinomial, "random.multinomial")


def multivariate_normal_cost(N: int, d: int) -> int:
    """FLOP cost of N draws from a d-dim Gaussian: covariance factorization
    (svd_cost(d,d,with_vectors=True)=26*d^3, numpy default method='svd') +
    affine transform (2*N*d^2) + N*d standard-normal draws at the
    transcendental rate (16/draw, matching random.normal's tier).
    Composite op => tier factors folded into flop_cost; weight stays 1.0.
    """
    from flopscope.numpy.random._cost_formulas import multivariate_normal_flops

    return multivariate_normal_flops(N, d)


@_counted_wrapper
def multivariate_normal(mean, cov, size=None, check_valid="warn", tol=1e-8):
    """Counted version of ``numpy.random.multivariate_normal``.

    Cost: svd_cost(d,d,with_vectors=True) + 2*N*d^2 + 16*N*d at weight 1.0,
    where d = len(mean) and N = size (or 1 when size is None).
    The SVD factorization (26*d^3) models numpy's default method='svd' which
    calls np.linalg.svd(cov) on the d×d covariance matrix. Composite op: the
    transcendental-tier factor for the N*d standard-normal draws is folded
    into flop_cost.
    """
    budget = require_budget()
    mean_arr = _np.asarray(mean)
    d = int(mean_arr.shape[-1]) if mean_arr.ndim else 1
    N = _output_size(size=size) if size is not None else 1
    flop_cost = multivariate_normal_cost(N, d)
    result = _npr.multivariate_normal(
        mean,
        cov,
        size=size,
        check_valid=check_valid,  # pyright: ignore[reportArgumentType]
        tol=tol,
    )
    with budget.deduct(
        "random.multivariate_normal",
        flop_cost=flop_cost,
        subscripts=None,
        shapes=((N, d),),
        dtypes=(result.dtype,),
    ):
        pass  # numpy already executed (module convention for samplers)
    return result


try:
    multivariate_normal.__signature__ = _inspect.signature(  # pyright: ignore[reportFunctionMemberAccess]
        _npr.multivariate_normal
    )
except (ValueError, TypeError):
    pass


dirichlet = _counted_sampler(_npr.dirichlet, "random.dirichlet")
randint = _counted_sampler(_npr.randint, "random.randint")


@_counted_wrapper
def _counted_size_only_sampler(
    np_func: Callable[..., Any],
    op_name: str,
) -> Callable[..., Any]:
    """Factory for samplers where the only arg is ``size`` (positional or kw)."""

    @_counted_wrapper
    def wrapper(size=None):
        budget = require_budget()
        n = _output_size(size=size)
        cost = _builtins.max(n, 1)
        # random/random_sample/ranf/sample have no dtype= param and always
        # return float64.
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=((n,),),
            dtypes=(_np.dtype(_np.float64),),
        ):
            result = _call_numpy(np_func, size=size)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


random = _counted_size_only_sampler(_npr.random, "random.random")
random_sample = _counted_size_only_sampler(_npr.random_sample, "random.random_sample")
ranf = _counted_size_only_sampler(_npr.ranf, "random.ranf")
sample = _counted_size_only_sampler(_npr.sample, "random.sample")


# ---------------------------------------------------------------------------
# Hand-coded ops with special cost formulas
# ---------------------------------------------------------------------------


@_counted_wrapper
def permutation(x):
    """Counted version of ``numpy.random.permutation``.

    Cost: numel(output) FLOPs.
    """
    budget = require_budget()
    n = int(x) if isinstance(x, (int, _np.integer)) else x.shape[0]
    cost = _builtins.max(n, 1)
    # permutation draws from arbitrary (possibly complex) caller-supplied
    # data rather than sampling a new value from a distribution; the
    # registry's complex_factor="illegal" for this op is only valid for the
    # genuine real-only distribution samplers, so dtype is intentionally
    # left undeclared here (numpy permutes complex arrays fine).
    with budget.deduct(
        "random.permutation", flop_cost=cost, subscripts=None, shapes=((n,),), dtypes=()
    ):
        result = _call_numpy(_npr.permutation, x)
    return result


@_counted_wrapper
def shuffle(x):
    """Counted version of ``numpy.random.shuffle``.

    Modifies ``x`` in-place. Cost: numel(output) FLOPs.
    """
    budget = require_budget()
    if hasattr(x, "shape"):
        n = x.shape[0]
    else:
        n = len(x)
    cost = _builtins.max(n, 1)
    # shuffle reorders arbitrary (possibly complex) caller-supplied data
    # in place; see the dtype-declaration note on permutation above.
    with budget.deduct(
        "random.shuffle", flop_cost=cost, subscripts=None, shapes=((n,),), dtypes=()
    ):
        _call_numpy(_npr.shuffle, x)


@_counted_wrapper
def choice(a, size=None, replace=True, p=None):
    """Counted version of ``numpy.random.choice``.

    Cost: numel(output) FLOPs if ``replace=True`` (+ CDF/binary-search term
    if ``p`` given); ``n`` FLOPs (Fisher-Yates, matches ``random.permutation``)
    if ``replace=False`` and ``p is None``; ``sort_cost(n)`` conservative floor
    for the data-dependent rejection loop if ``replace=False`` with ``p``.
    """
    budget = require_budget()
    if isinstance(a, (int, _np.integer)):
        n = int(a)
    else:
        a_arr = _np.asarray(a)
        n = a_arr.shape[0] if a_arr.ndim > 0 else 1
    if replace:
        out_size = _output_size(size=size)
        cost = _builtins.max(out_size, 1)
        if p is not None:
            # CDF build: cumsum + normalise + final pass (3*n), then
            # binary-search per draw (out_size * ceil(log2(n))).
            cost += 3 * n + _builtins.max(out_size, 1) * _ceil_log2(n)
        # choice selects from arbitrary (possibly complex) caller-supplied
        # data; see the dtype-declaration note on permutation above.
        with budget.deduct(
            "random.choice",
            flop_cost=cost,
            subscripts=None,
            shapes=((out_size,),),
            dtypes=(),
        ):
            result = _call_numpy(_npr.choice, a, size=size, replace=replace, p=p)
    else:
        if p is None:
            # Legacy RandomState.choice is permutation(pop_size)[:size] —
            # Fisher-Yates O(n), bit-exact with random.permutation.
            # Generator.choice uses Floyd's/tail-shuffle (<= O(n)); charging n
            # is a conservative ceiling.
            cost = _builtins.max(n, 1)
        else:
            # Data-dependent rejection loop: per-iteration cumsum + normalize
            # + searchsorted; sort_cost(n) is the documented conservative floor.
            cost = sort_cost(n)
        with budget.deduct(
            "random.choice", flop_cost=cost, subscripts=None, shapes=((n,),), dtypes=()
        ):
            result = _call_numpy(_npr.choice, a, size=size, replace=replace, p=p)
    # Preserve identity when picking a scalar from an object-dtype array:
    # numpy returns the exact object stored in the input, and user code
    # (e.g. `choice(object_array) is original_object`) relies on that.
    # Wrapping in FlopscopeArray would break the identity. `wrap_module_returns`
    # skips this function (see skip_names below); we do explicit wrapping
    # only for the common numeric case.
    if size is None and isinstance(a, _np.ndarray) and a.dtype.kind == "O":
        return result
    if isinstance(result, _np.ndarray):
        from flopscope._ndarray import _asflopscope

        return _asflopscope(result)
    return result


@_counted_wrapper
def symmetric(
    shape: int | Sequence[int],
    symmetry: SymmetryGroup,
    distribution: str | Callable[..., Any] = "randn",
    **distribution_kwargs: Any,
) -> FlopscopeArray:
    """Sample random data and project it to a symmetry group.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the sampled array.
    symmetry : SymmetryGroup
        Symmetry group used for Reynolds averaging.
    distribution : str or callable, default ``\"randn\"``
        Name of a ``numpy.random`` distribution function (for example
        ``\"randn\"`` or ``\"normal\"``), or a callable that accepts either:

        - ``(*shape, **kwargs)``
        - ``size=shape``

        and returns an array.
    **distribution_kwargs
        Extra keyword arguments forwarded to the distribution function.

    Returns
    -------
    SymmetricTensor
        The symmetrized sample wrapped with :func:`flops.as_symmetric`.

    Raises
    ------
    ValueError
        If ``shape`` is not an integer or a tuple/list of integers.
    TypeError
        If ``distribution`` is neither a NumPy random distribution name nor a
        callable.
    AttributeError
        If ``distribution`` is a string that is not present in NumPy random.
    SymmetryError
        If projected output does not satisfy the requested symmetry constraints.

    Notes
    -----
    This is equivalent to ``flops.symmetrize(sampled_data, symmetry=symmetry)`` where
    ``sampled_data`` is drawn from ``distribution``.

    The implementation currently:

    1. Samples raw values from the selected distribution.
    2. Applies :func:`flops.symmetrize` to project into the symmetry-invariant
       subspace.

    Estimated FLOP cost is approximately:

    ``C_dist(n_elem) + |G| * n_elem + n_elem`` (+ validation),

    where ``n_elem`` is the sampled array size, ``|G|`` is the group order, and
    ``C_dist(n_elem)`` is the cost of the chosen sampling distribution.
    The default ``distribution='randn'`` corresponds to ``C_dist(n_elem)≈n_elem``.

    For existing data, use :func:`flops.symmetrize` directly.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> S = fnp.random.symmetric((4, 4), flops.SymmetryGroup.symmetric(axes=(0, 1)))
    >>> S.is_symmetric((0, 1))
    True

    >>> S = fnp.random.symmetric(
    ...     (3, 3, 3),
    ...     flops.SymmetryGroup.cyclic(axes=(0, 1, 2)),
    ...     distribution="normal",
    ...     loc=0.0,
    ...     scale=1.0,
    ... )
    >>> S.is_symmetric((0, 1, 2))
    True

    >>> import numpy as np
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> def shifted_uniform(shape, **kwargs):
    ...     return np.random.uniform(*shape, **kwargs)
    >>> S = fnp.random.symmetric(
    ...     (2, 2),
    ...     flops.SymmetryGroup.symmetric(axes=(0, 1)),
    ...     distribution=shifted_uniform,
    ... )
    >>> S.is_symmetric((0, 1))
    True
    """
    if isinstance(shape, int):
        shape_tuple = (shape,)
    elif isinstance(shape, tuple):
        shape_tuple = shape
    elif isinstance(shape, list):
        shape_tuple = tuple(shape)
    else:
        try:
            shape_tuple = tuple(shape)
        except TypeError as exc:
            raise ValueError("shape must be an int or a tuple-like of ints") from exc

    shape_tuple = tuple(int(s) for s in shape_tuple)
    sample_size = _builtins.max(int(_np.prod(shape_tuple)), 1)

    if isinstance(distribution, str):
        if not hasattr(_npr, distribution):
            raise AttributeError(
                f"flopscope.numpy.random does not provide distribution '{distribution}'"
            )
        sampler = getattr(_npr, distribution)
        if distribution in {"rand", "randn"}:
            sample = sampler(*shape_tuple, **distribution_kwargs)
        else:
            sample = sampler(size=shape_tuple, **distribution_kwargs)
    elif callable(distribution):
        try:
            sample = distribution(*shape_tuple, **distribution_kwargs)
        except TypeError:
            sample = distribution(size=shape_tuple, **distribution_kwargs)
    else:
        raise TypeError(
            "distribution must be a numpy random function name or a callable"
        )

    budget = require_budget()
    G = _builtins.max(symmetry.order(), 1)
    # sample numel + projection core ((|G|+1)*numel) == sample + symmetrize
    cost = _builtins.max(sample_size + (G + 1) * sample_size, 1)
    # `sample` is usually drawn from a named real-only numpy distribution,
    # but `distribution` may be an arbitrary caller callable that returns
    # complex data; the registry's complex_factor="illegal" here is only
    # valid for the genuine real-only samplers, so dtype is intentionally
    # left undeclared (see the dtype-declaration note on permutation above).
    with budget.deduct(
        "random.symmetric",
        flop_cost=cost,
        subscripts=None,
        shapes=(shape_tuple,),
        dtypes=(),
    ):
        from flopscope._symmetric import (
            SymmetricTensor,
            _project_core,
            validate_symmetry_groups,
        )

        projected = _project_core(sample, symmetry)  # _project_core does np.asarray
        validate_symmetry_groups(projected, [symmetry])  # uncounted safety check
        return SymmetricTensor(projected, symmetry=symmetry)


@_counted_wrapper
def bytes(length):
    """Counted version of ``numpy.random.bytes``.

    Cost: ``length`` FLOPs.
    """
    budget = require_budget()
    cost = _builtins.max(int(length), 1)
    # Output is a Python `bytes` object, not a numpy array -- no billing dtype.
    with budget.deduct(
        "random.bytes", flop_cost=cost, subscripts=None, shapes=((length,),), dtypes=()
    ):
        result = _call_numpy(_npr.bytes, length)
    return result


# ---------------------------------------------------------------------------
# Fallback __getattr__ for anything not explicitly listed
# ---------------------------------------------------------------------------

# Explicit allowlist of numpy.random types that pass through without counting:
# these are bit-generator classes (no math; FLOP counting happens at the
# sampler-method level on the resulting Generator) and the SeedSequence utility.
_PASSTHROUGH_TYPES: frozenset[str] = frozenset(
    {
        "BitGenerator",
        "MT19937",
        "PCG64",
        "PCG64DXSM",
        "Philox",
        "SFC64",
    }
)


def __getattr__(name):
    if name in _PASSTHROUGH_TYPES:
        return getattr(_npr, name)
    raise AttributeError(
        f"flopscope.numpy.random has no attribute '{name}'.\n"
        f"For new code: rng = fnp.random.default_rng(seed); rng.<sampler>(...).\n"
        f"If '{name}' should be supported, please file an issue at "
        f"https://github.com/AIcrowd/flopscope/issues."
    )


import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(
    _sys.modules[__name__],
    skip_names={
        "default_rng",
        "RandomState",
        "SeedSequence",
        "seed",
        "get_state",
        "set_state",
        # choice does its own wrapping because it needs to preserve the
        # identity of picked objects from object-dtype arrays.
        "choice",
    },
)
