"""Counted subclasses of numpy.random.Generator and numpy.random.RandomState.

The classes use ``__getattribute__`` as a gate that consults registry-derived
``_COUNTED`` and ``_FREE`` sets. Sampler methods are added at module init by
``_build_counted_class``, which iterates the registry and emits wrappers via
``_make_counted_method``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as _np

from flopscope._budget import _counted_wrapper
from flopscope._ndarray import _asflopscope
from flopscope._registry import REGISTRY
from flopscope._validation import require_budget
from flopscope.errors import UnsupportedFunctionError
from flopscope.numpy.random._cost_formulas import COST_FORMULAS


@_counted_wrapper
def _make_counted_method(
    op_name: str, formula_name: str, base_cls: type, plain_factory: Callable[..., Any]
) -> Callable[..., Any]:
    """Build a counted-method wrapper for the parent class's named method.

    The wrapper calls the parent method on a *plain* base-class instance that
    shares the same RNG state (so internal sibling-method calls do not go
    through the counted gate and double-count).  After numpy's C path returns,
    cost is computed via the named formula, deducted from the active
    BudgetContext, and the result is wrapped as FlopscopeArray if it is an
    ndarray.
    """
    method_name = op_name.split(".")[-1]
    base_method = getattr(base_cls, method_name)
    formula = COST_FORMULAS[formula_name]

    # choice/permutation/permuted/shuffle draw from arbitrary (possibly
    # complex) caller-supplied data rather than sampling a new value from a
    # distribution; numpy permutes/shuffles/selects complex arrays fine, but
    # the registry's complex_factor="illegal" for these methods is only
    # valid for the genuine real-only distribution samplers this dispatcher
    # also builds. Leave dtype undeclared for the pass-through methods to
    # avoid a false-positive UnsupportedDtypeError regression; every other
    # method synthesizes new real-only values, so its actual output dtype
    # (read off the already-computed result) is safe to declare.
    _dtype_neutral_methods = frozenset({"choice", "permutation", "permuted", "shuffle"})

    @functools.wraps(base_method)
    @_counted_wrapper
    def wrapped(self, *args: Any, **kwargs: Any) -> Any:
        budget = require_budget()
        # Dispatch through a plain base-class instance to avoid the counted
        # __getattribute__ gate intercepting any internal sibling calls
        # (e.g. Generator.choice → Generator.integers).
        plain = plain_factory(self)
        result = base_method(plain, *args, **kwargs)
        cost = formula(args, kwargs, result)
        if method_name in _dtype_neutral_methods:
            billing_dtypes: tuple = ()
        elif isinstance(result, _np.ndarray):
            billing_dtypes = (result.dtype,)
        else:
            billing_dtypes = ()
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(), dtypes=billing_dtypes
        ):
            pass  # numpy already executed; deduct is post-hoc for output-dependent cost
        if isinstance(result, _np.ndarray):
            return _asflopscope(result)
        return result

    return wrapped


def _build_counted_class(
    base_cls: type,
    op_prefix: str,
    target_cls: type,
    plain_factory: Callable[..., Any],
) -> None:
    """Populate target_cls._COUNTED, _FREE, and counted method bindings.

    Reads REGISTRY entries whose op_name starts with op_prefix. counted_random_method
    entries become wrapped class methods; free_random_method entries are added to
    _FREE so the __getattribute__ gate lets them through unwrapped.

    plain_factory is called with ``self`` (the counted subclass instance) to
    produce a plain base-class instance sharing the same RNG state, so that
    internal sibling calls inside numpy's C code do not go through the gate.
    """
    counted: set[str] = set()
    free: set[str] = set()
    for op_name, entry in REGISTRY.items():
        if not op_name.startswith(op_prefix):
            continue
        short = op_name[len(op_prefix) :]
        category = entry.get("category")
        if category == "counted_random_method":
            counted.add(short)
            setattr(
                target_cls,
                short,
                _make_counted_method(
                    op_name, entry["cost_formula"], base_cls, plain_factory
                ),
            )
        elif category == "free_random_method":
            free.add(short)
    target_cls._COUNTED = frozenset(counted)
    target_cls._FREE = frozenset(free)


def _reconstruct_counted_generator(bit_generator: Any) -> _CountedGenerator:
    """Pickle helper — reconstruct a _CountedGenerator from its bit_generator."""
    return _CountedGenerator(bit_generator)


def _reconstruct_counted_random_state(state: tuple) -> _CountedRandomState:
    """Pickle helper — reconstruct a _CountedRandomState from its full state tuple."""
    # Construct with default seed; the state we just received will overwrite.
    rs = _CountedRandomState(seed=None)
    rs.set_state(state)
    return rs


class _CountedGenerator(_np.random.Generator):
    """numpy ``Generator`` subclass with FLOP-counted sampler methods.

    Each sampler method (``standard_normal``, ``normal``, ``uniform``,
    ``integers``, ``choice``, ``shuffle``, ``permutation``, ``bytes``, ...)
    deducts FLOPs from the active ``BudgetContext`` and returns
    ``FlopscopeArray``. Free attribute access (``bit_generator``, ``spawn``)
    is allowed; anything else raises ``UnsupportedFunctionError``.

    Construct via :func:`flopscope.numpy.random.default_rng` (canonical) or by
    passing a ``BitGenerator`` directly: ``Generator(np.random.PCG64(42))``.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> from flopscope import BudgetContext
    >>> rng = fnp.random.default_rng(42)
    >>> with BudgetContext(flop_budget=10**6):
    ...     x = rng.standard_normal((10,))
    >>> type(x).__name__
    'FlopscopeArray'

    Pickle / ``copy`` round-trips preserve counting:

    >>> import pickle
    >>> revived = pickle.loads(pickle.dumps(rng))
    >>> isinstance(revived, type(rng))
    True

    See Also
    --------
    fnp.random.default_rng : canonical constructor.
    """

    _COUNTED: ClassVar[frozenset[str]] = frozenset()
    _FREE: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        cls = type(self)
        if name in cls._FREE or name in cls._COUNTED:
            return super().__getattribute__(name)
        raise UnsupportedFunctionError(
            f"flopscope does not count Generator.{name}. "
            f"This is either a new numpy method or one not yet wrapped. "
            f"See https://github.com/AIcrowd/flopscope/issues/18"
        )

    # Wrap spawned children so their methods stay counted.
    # Annotated as list[Generator] to match parent's invariant return type;
    # runtime objects are _CountedGenerator (subclass of Generator).
    def spawn(self, n_children: int) -> list[_np.random.Generator]:
        children = _np.random.Generator.spawn(self, n_children)
        return [_CountedGenerator(c.bit_generator) for c in children]

    # Round-trip pickling back to _CountedGenerator (numpy's default would
    # reconstruct as np.random.Generator and lose the counting wrappers).
    def __reduce__(self) -> tuple:
        return (_reconstruct_counted_generator, (self.bit_generator,))


class _CountedRandomState(_np.random.RandomState):
    """numpy legacy ``RandomState`` subclass with FLOP-counted sampler methods.

    Mirrors :class:`_CountedGenerator` for the legacy API: each sampler
    (``randn``, ``normal``, ``uniform``, ``randint``, ``choice``,
    ``shuffle``, ``permutation``, ``bytes``, ...) deducts FLOPs from the
    active ``BudgetContext`` and returns ``FlopscopeArray``. Free methods
    (``seed``, ``get_state``, ``set_state``) pass through; everything else
    raises ``UnsupportedFunctionError``.

    Modern code should prefer :func:`flopscope.numpy.random.default_rng`;
    use this only when porting code that relies on the legacy API.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> from flopscope import BudgetContext
    >>> rs = fnp.random.RandomState(42)
    >>> with BudgetContext(flop_budget=10**6):
    ...     z = rs.randn(10)
    >>> type(z).__name__
    'FlopscopeArray'

    See Also
    --------
    fnp.random.default_rng : canonical (modern) RNG constructor.
    """

    _COUNTED: ClassVar[frozenset[str]] = frozenset()
    _FREE: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        cls = type(self)
        if name in cls._FREE or name in cls._COUNTED:
            return super().__getattribute__(name)
        raise UnsupportedFunctionError(
            f"flopscope does not count RandomState.{name}. "
            f"This is either a new numpy method or one not yet wrapped. "
            f"See https://github.com/AIcrowd/flopscope/issues/18"
        )

    # Round-trip pickling back to _CountedRandomState (numpy's default
    # __reduce__ hardcodes __randomstate_ctor which returns plain RandomState,
    # silently losing FLOP counting).
    def __reduce__(self) -> tuple:
        # get_state() is in _FREE so the gate lets it through.
        # set_state() is also in _FREE, used by the helper on the other side.
        return (_reconstruct_counted_random_state, (self.get_state(),))


# Wire counted methods at import time.
# plain_factory creates a plain base-class instance sharing the same RNG state;
# this prevents numpy's internal sibling-method calls from going through the
# counted __getattribute__ gate and double-counting FLOPs.
_build_counted_class(
    _np.random.Generator,
    "random.Generator.",
    _CountedGenerator,
    plain_factory=lambda self: _np.random.Generator(self._bit_generator),
)
_build_counted_class(
    _np.random.RandomState,
    "random.RandomState.",
    _CountedRandomState,
    plain_factory=lambda self: _np.random.RandomState(self._bit_generator),
)
