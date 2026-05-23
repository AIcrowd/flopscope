"""Subclass of ``numpy.ndarray`` that routes every operation through
flopscope's FLOP-counted ``me.*`` functions.

``FlopscopeArray`` overrides:

- **Arithmetic / comparison / bitwise / shift dunders** so ``a + b``,
  ``a @ b``, ``a == b``, ``a & b``, ``a << b`` etc. route through
  ``me.add``, ``me.matmul``, etc.
- **In-place dunders** (``__iadd__``, …, ``__imatmul__``) with a
  symmetry-corruption guard that refuses ``A_sym += B`` when the
  result would weaken or destroy ``A_sym``'s tagged symmetry. ``A @= B``
  on shape-changing matmul falls back to CPython's documented
  rebind-the-name semantics.
- **25 ndarray methods** (``sum``, ``mean``, ``dot``, ``argsort``,
  ``compress``, ``trace``, ``round``, ``clip``, ``ptp``, …) so
  ``a.sum()`` and ``fnp.sum(a)`` produce the same FLOP count.
- **In-place ``sort`` / ``partition``** which refuse on a
  ``SymmetricTensor`` (the reorder would break symmetry).

It also implements two NumPy protocols:

- ``__array_ufunc__`` (NEP 13) — ``np.add(a, b)``, ``np.add.reduce(a)``,
  ``np.add.outer(a, b)``, ``np.add.at(a, idx, val)``, multi-output
  ufuncs (``np.divmod`` / ``np.frexp`` / ``np.modf``), etc.
- ``__array_function__`` (NEP 18) — explicit allowlist routing
  ``np.<func>(flopscope, …)`` for ~108 callables (reductions, sorting,
  shape ops, linear algebra, comparisons, histograms, …) plus a
  ``_PASSTHROUGH`` set of zero-FLOP type/shape queries.

Recursion-prevention helpers (``_to_base_ndarray``,
``_to_base_ndarray_tree``) view-cast ``FlopscopeArray`` to plain
``np.ndarray`` for the inner NumPy call, breaking the cycle that
would otherwise re-dispatch through these protocols.

Because ``FlopscopeArray`` inherits from ``numpy.ndarray``,
``isinstance(x, numpy.ndarray)`` returns ``True``. All ``me.*``
functions return ``FlopscopeArray`` (or ``SymmetricTensor`` when symmetry
survives the operation). See PR #67 for the complete protocol design.
"""

from __future__ import annotations

import functools as _functools
import inspect as _inspect
from typing import Any

import numpy as _np
from numpy.typing import DTypeLike


def _me():
    """Lazy import of flopscope.numpy to avoid circular imports.

    The dunder methods need fnp.add, fnp.multiply etc. — these counted
    numpy ops live under flopscope.numpy in the JAX-style public API
    (top-level flopscope exposes only primitives like BudgetContext).
    Defer the import until first use to break the import cycle:
    flopscope.numpy → flopscope._free_ops/etc. → this module.
    """
    import flopscope.numpy as _fnp

    return _fnp


# Eagerly captured at import time so ``tests/numpy_compat`` monkeypatching
# (which replaces ``np.<name>`` with ``fnp.<name>``) doesn't accidentally
# install the flopscope-replacements into ``_PASSTHROUGH``. The set holds the
# *original* numpy callables.
_PASSTHROUGH_NAMES = (
    # Zero-FLOP type/shape queries:
    "ndim",
    "shape",
    "size",
    # Zero-FLOP type-system queries (added by Task 4 for #62/#58 followup):
    "result_type",
    "can_cast",
    "min_scalar_type",
    "promote_types",
    "find_common_type",
    "mintypecode",
    # Test-harness assertion that should not count FLOPs:
    "array_equal",
    # Zero-FLOP memory-layout queries (#72):
    "may_share_memory",
    "shares_memory",
    "byte_bounds",
    # Zero-FLOP boolean predicates (#72 audit):
    "iscomplexobj",
    "isrealobj",
    "isfortran",
    "isscalar",
    # Zero-FLOP shape arithmetic (#72 audit):
    "broadcast_shapes",
)


def _build_passthrough():
    """Build the ``_PASSTHROUGH`` set, eagerly at import time."""
    s = set()
    for name in _PASSTHROUGH_NAMES:
        fn = getattr(_np, name, None)
        if fn is not None:
            s.add(fn)
    return s


_INITIAL_PASSTHROUGH = _build_passthrough()


class FlopscopeArray(_np.ndarray):
    """A numpy ndarray subclass with FLOP-tracked operators.

    Behaves exactly like numpy.ndarray except that arithmetic and
    related operators route through flopscope's counted fnp.* functions
    so the active BudgetContext sees them.
    """

    def __new__(
        cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __array_finalize__(self, obj):
        # Called when numpy creates a view or slice of this subclass.
        # No subclass state to propagate.
        pass

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        """Honor numpy's `return_scalar` request from ufunc reductions.

        When a ufunc reduction collapses to a single value (e.g.
        ``np.bitwise_or.reduce(np.array([True], dtype=object))``), numpy
        passes ``return_scalar=True`` to ``__array_wrap__`` so the caller
        receives a Python scalar rather than a 0-d ndarray. The default
        ndarray behaviour respects this flag; we forward it explicitly so
        the same behaviour holds when the input is a FlopscopeArray.

        For non-scalar results we let numpy preserve the subclass (the
        default behaviour) so views, slices, and ufunc outputs stay
        FlopscopeArrays — keeping operator overloads and FLOP tracking
        intact for chained expressions.

        Flopscope does not guarantee ndarray flag fidelity for subclass
        results. In particular, view-casting a fresh ufunc result into a
        subclass often reports ``OWNDATA=False`` because the subclass is a
        view over the ufunc's base ndarray. We intentionally keep that
        no-copy behaviour because ndarray-subclass operations are on a hot
        path and avoiding extra copies is a higher priority than exact
        flag parity with bare ndarrays.
        """
        if return_scalar:
            return out_arr[()]
        return super().__array_wrap__(out_arr, context, return_scalar)

    # ----- numpy ufunc protocol (NEP 13) -----

    _REDUCE_TO_WHEST = {
        "add": "sum",
        "multiply": "prod",
        "maximum": "max",
        "minimum": "min",
        "logical_and": "all",
        "logical_or": "any",
    }
    _ACCUMULATE_TO_WHEST = {
        "add": "cumsum",
        "multiply": "cumprod",
    }

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Route ufunc calls through flopscope's counted functions.

        Triggered for:
        - ``np.add(flopscope, x)``           → method='__call__'
        - ``np.add.reduce(flopscope)``       → method='reduce'
        - ``np.add.accumulate(flopscope)``   → method='accumulate'
        - ``np.add.outer(flopscope, w2)``    → method='outer'
        - ``np.add.reduceat(flopscope, ix)`` → method='reduceat'
        - ``np.add.at(flopscope, ix, val)``  → method='at'

        ``out`` is passed by NumPy as a tuple of length ``ufunc.nout``.
        For single-output ufuncs we unwrap the 1-tuple to ``out=arr``;
        for multi-output ufuncs (``divmod``, ``frexp``, ``modf``) we
        forward the tuple as-is to the corresponding ``fnp.*`` wrapper,
        which knows how to handle per-slot stripping and identity.

        ``reduce`` / ``accumulate`` first try the optimized routing in
        :attr:`_REDUCE_TO_WHEST` / :attr:`_ACCUMULATE_TO_WHEST`; on a
        miss (e.g. ``np.subtract.reduce``) they fall back to a generic
        path in :mod:`flopscope._pointwise` that strips, charges
        :func:`reduction_cost`, and routes back through the raw
        ``ufunc.<method>``. ``outer`` / ``reduceat`` / ``at`` always
        use the generic path. ``ufunc.at`` refuses on SymmetricTensor
        operands (the in-place fancy-index write would break symmetry).
        """
        me = _me()

        np_target_name = None  # used to drive _filter_to_np_signature below
        whest_fn = None
        if method == "__call__":
            whest_fn = getattr(me, ufunc.__name__, None)
            np_target_name = ufunc.__name__
        elif method == "reduce":
            target = self._REDUCE_TO_WHEST.get(ufunc.__name__)
            if target is not None:
                whest_fn = getattr(me, target, None)
                np_target_name = target
                # NumPy's ufunc.reduce defaults to axis=0; flopscope's me.sum etc.
                # default to axis=None (full reduction). Force NumPy default.
                kwargs.setdefault("axis", 0)
        elif method == "accumulate":
            target = self._ACCUMULATE_TO_WHEST.get(ufunc.__name__)
            if target is not None:
                whest_fn = getattr(me, target, None)
                np_target_name = target
                kwargs.setdefault("axis", 0)

        if whest_fn is not None:
            # Optimized routing-table path: forward to the dedicated
            # ``fnp.*`` wrapper.
            #
            # Unwrap single-output ``out`` tuple.
            if out is not None:
                if isinstance(out, tuple) and len(out) == 1:
                    kwargs["out"] = out[0]
                else:
                    kwargs["out"] = out
            # Filter kwargs against the target NumPy callable's signature so
            # ufunc-internal kwargs (e.g. ``dtype=`` from np.all → np.add.reduce)
            # don't reach a function-form flopscope wrapper that doesn't accept
            # them.
            if np_target_name is not None:
                kwargs = _filter_to_np_signature(
                    getattr(_np, np_target_name, None), kwargs
                )
            return whest_fn(*inputs, **kwargs)

        # Generic ufunc-method paths for ops without a dedicated flopscope
        # equivalent. Lazy-imported to avoid the _ndarray ↔ _pointwise
        # circular-dependency loop.
        if method in ("outer", "reduce", "accumulate", "reduceat", "at"):
            from flopscope._pointwise import (
                _counted_ufunc_accumulate_generic,
                _counted_ufunc_at,
                _counted_ufunc_outer,
                _counted_ufunc_reduce_generic,
                _counted_ufunc_reduceat,
            )

            # Unwrap single-output ``out`` tuple for the generic paths
            # too. ``ufunc.at`` doesn't take ``out``.
            out_for_generic = out
            if out is not None and isinstance(out, tuple) and len(out) == 1:
                out_for_generic = out[0]

            if method == "outer":
                return _counted_ufunc_outer(
                    ufunc, *inputs, out=out_for_generic, **kwargs
                )
            if method == "reduce":
                kwargs.setdefault("axis", 0)
                return _counted_ufunc_reduce_generic(
                    ufunc, *inputs, out=out_for_generic, **kwargs
                )
            if method == "accumulate":
                kwargs.setdefault("axis", 0)
                return _counted_ufunc_accumulate_generic(
                    ufunc, *inputs, out=out_for_generic, **kwargs
                )
            if method == "reduceat":
                return _counted_ufunc_reduceat(
                    ufunc, *inputs, out=out_for_generic, **kwargs
                )
            if method == "at":
                # ``ufunc.at`` does not accept ``out=``.
                return _counted_ufunc_at(ufunc, *inputs, **kwargs)

        return NotImplemented

    # ----- numpy array-function protocol (NEP 18) -----

    # Lazy-built dispatch map. Populated on first __array_function__ call
    # because flopscope's namespace uses lazy submodule loading.
    _ARRAY_FUNCTION_DISPATCH: dict | None = None
    _PASSTHROUGH: set | None = None

    @classmethod
    def _get_array_function_dispatch(cls):
        """Build (once) and return the np-callable → flopscope-callable map."""
        if cls._ARRAY_FUNCTION_DISPATCH is not None:
            return cls._ARRAY_FUNCTION_DISPATCH

        me = _me()
        d: dict = {}

        def _bind(np_attr_path, we_attr_path):
            """Look up np.<path> and me.<path>, add to dispatch map.

            Silently skip pairs where one side is missing (e.g. linalg
            functions added in newer NumPy versions).
            """
            np_obj = _np
            for part in np_attr_path.split("."):
                np_obj = getattr(np_obj, part, None)
                if np_obj is None:
                    return
            we_obj = me
            for part in we_attr_path.split("."):
                we_obj = getattr(we_obj, part, None)
                if we_obj is None:
                    return
            if callable(np_obj) and callable(we_obj):
                d[np_obj] = we_obj

        # ----- Reductions -----
        for name in (
            "sum",
            "prod",
            "mean",
            "min",
            "max",
            "std",
            "var",
            "all",
            "any",
            "cumsum",
            "cumprod",
            "argmin",
            "argmax",
            "ptp",
            "median",
            "average",
            "percentile",
            "quantile",
            # NumPy 2.x retained ``amax``/``amin`` as exported aliases
            # for ``max``/``min``. Bind them so ``np.amax(FlopscopeArray)``
            # routes through flopscope instead of raising TypeError.
            "amax",
            "amin",
        ):
            _bind(name, name)

        # ----- Sorting / selection -----
        for name in (
            "sort",
            "argsort",
            "lexsort",
            "partition",
            "argpartition",
            "searchsorted",
            "digitize",
        ):
            _bind(name, name)

        # ----- Set / unique -----
        for name in (
            "unique",
            "unique_all",
            "unique_counts",
            "unique_inverse",
            "unique_values",
            "in1d",
            "isin",
            "intersect1d",
            "union1d",
            "setdiff1d",
            "setxor1d",
        ):
            _bind(name, name)

        # ----- Free / structural (asarray excluded) -----
        for name in (
            "where",
            "tile",
            "repeat",
            "flip",
            "roll",
            "pad",
            "triu",
            "tril",
            "diagonal",
            "broadcast_to",
            "meshgrid",
            "copy",
            "astype",
            "trace",
            "diff",
            "gradient",
            "clip",
            "round",
        ):
            _bind(name, name)

        # ----- Shape / view ops -----
        # The flopscope counterparts (``me.transpose``, ``me.swapaxes``,
        # ``me.moveaxis``, etc.) handle ``SymmetricTensor`` axis
        # remapping correctly via ``wrap_with_symmetry`` /
        # ``remap_group_axes``. Routing through the allowlist preserves
        # symmetry; PASSTHROUGH would silently strip it via
        # ``_to_base_ndarray_tree`` before the raw NumPy call.
        for name in (
            "transpose",
            "swapaxes",
            "moveaxis",
            "reshape",
            "ravel",
            "expand_dims",
            "squeeze",
            "concatenate",
            "stack",
            "vstack",
            "hstack",
            "column_stack",
            "split",
            "hsplit",
            "vsplit",
            "dsplit",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "broadcast_to",
            "matrix_transpose",  # numpy 2.x ufunc-like
        ):
            _bind(name, name)

        # ----- Linear algebra -----
        for name in (
            "dot",
            "matmul",
            "einsum",
            "tensordot",
            "inner",
            "outer",
            "cross",
        ):
            _bind(name, name)

        # ----- Comparisons -----
        for name in (
            "allclose",
            "isclose",
            "array_equiv",
            # NOTE: array_equal is in _PASSTHROUGH instead.
        ):
            _bind(name, name)

        # ----- Histograms / counts -----
        for name in (
            "histogram",
            "histogram2d",
            "histogramdd",
            "histogram_bin_edges",
            "bincount",
            "vander",
            "apply_over_axes",
            "piecewise",
        ):
            _bind(name, name)

        # ----- linalg submodule -----
        for name in (
            "norm",
            "solve",
            "det",
            "inv",
            "pinv",
            "eig",
            "eigh",
            "eigvals",
            "eigvalsh",
            "svd",
            "qr",
            "cholesky",
            "matrix_rank",
            "lstsq",
            "multi_dot",
            "matrix_power",
            "slogdet",
        ):
            _bind(f"linalg.{name}", f"linalg.{name}")

        cls._ARRAY_FUNCTION_DISPATCH = d
        return d

    @classmethod
    def _get_passthrough(cls):
        """Return the eagerly-captured passthrough set."""
        if cls._PASSTHROUGH is not None:
            return cls._PASSTHROUGH
        cls._PASSTHROUGH = set(_INITIAL_PASSTHROUGH)
        return cls._PASSTHROUGH

    def __array_function__(self, func, types, args, kwargs):
        """Route ``np.<func>(flopscope, ...)`` calls through flopscope.

        Two distinct paths:

        - **Inside an fnp wrapper (depth > 0):** A wrapper forgot to strip a
          ``FlopscopeArray`` before calling raw numpy, and numpy's NEP 18
          dispatch caught us. This is a flopscope bug (the polyval class from
          issue #69). Raise ``RuntimeError`` at the leak site so the bug is
          impossible to miss.

        - **Top-level call (depth == 0):** A user wrote ``np.<func>(whest)``
          directly. PASSTHROUGH set is checked first for zero-FLOP queries.
          Otherwise we route through the allowlist to ``fnp.<func>`` AND emit
          a ``UserWarning`` so the user knows to call ``fnp.<func>`` directly.
          Warning is de-duped per call site by Python's ``warnings`` module.
        """
        from flopscope._budget import _called_from_wrapper

        if _called_from_wrapper():
            raise RuntimeError(
                f"WhestArray reached numpy.{func.__name__} from inside an fnp "
                f"wrapper — missing _to_base_ndarray() strip. Check the "
                f"calling fnp wrapper and add a strip before the numpy call."
            )

        # PASSTHROUGH check first: zero-FLOP queries bypass dispatch.
        if func in self._get_passthrough():
            stripped_args = _to_base_ndarray_tree(args)
            stripped_kwargs = {k: _to_base_ndarray_tree(v) for k, v in kwargs.items()}
            return func(*stripped_args, **stripped_kwargs)

        dispatch = self._get_array_function_dispatch()
        we_func = dispatch.get(func)
        if we_func is None:
            return NotImplemented

        import warnings as _warnings
        _warnings.warn(
            f"np.{func.__name__}(WhestArray) auto-routed to fnp.{func.__name__}; "
            f"call fnp.{func.__name__} directly to avoid this warning.",
            UserWarning,
            stacklevel=2,
        )
        return we_func(*args, **kwargs)

    # ----- ndarray method overrides (route through me.* for budget parity) -----
    # We forward *args, **kwargs to be forward-compatible with NumPy's
    # evolving method signatures (dtype, out, where, casting, keepdims, axis
    # as positional or keyword).

    def sum(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().sum(self, *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().mean(self, *args, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().prod(self, *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().min(self, *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().max(self, *args, **kwargs)

    def std(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().std(self, *args, **kwargs)

    def var(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().var(self, *args, **kwargs)

    def all(self, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().all(self, *args, **kwargs)

    def any(self, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().any(self, *args, **kwargs)

    def cumsum(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().cumsum(self, *args, **kwargs)

    def cumprod(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().cumprod(self, *args, **kwargs)

    def argmin(self, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().argmin(self, *args, **kwargs)

    def argmax(self, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().argmax(self, *args, **kwargs)

    def ptp(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().ptp(self, *args, **kwargs)

    def trace(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().trace(self, *args, **kwargs)

    def round(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().round(self, *args, **kwargs)

    def clip(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().clip(self, *args, **kwargs)

    def astype(  # type: ignore[override]
        self,
        dtype: DTypeLike,
        order: Any = "K",
        casting: Any = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> FlopscopeArray:
        return _np.ndarray.astype(
            self, dtype, order=order, casting=casting, subok=subok, copy=copy
        )  # type: ignore[return-value]

    # ----- Other ndarray methods -----

    def dot(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().dot(self, *args, **kwargs)

    def conj(self) -> FlopscopeArray:
        return _me().conjugate(self)

    def conjugate(self) -> FlopscopeArray:
        return _me().conjugate(self)

    def argsort(self, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().argsort(self, *args, **kwargs)

    def argpartition(self, kth: Any, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().argpartition(self, kth, *args, **kwargs)

    def take(self, indices: Any, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().take(self, indices, *args, **kwargs)

    def repeat(self, repeats: Any, *args: Any, **kwargs: Any) -> FlopscopeArray:
        return _me().repeat(self, repeats, *args, **kwargs)

    def searchsorted(self, v: Any, *args: Any, **kwargs: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().searchsorted(self, v, *args, **kwargs)

    def compress(self, condition: Any, *args: Any, **kwargs: Any) -> FlopscopeArray:
        # ndarray.compress(condition) -> np.compress(condition, arr)
        return _me().compress(condition, self, *args, **kwargs)

    # In-place sort/partition: NumPy mutates self and returns None.
    # Charge FLOPs through me.sort/partition, then copy result into self.
    # Guard against in-place mutation that would silently break symmetry.

    def _check_inplace_breaks_symmetry(self, op_name):
        """Refuse in-place ops that would invalidate SymmetricTensor metadata.

        ``self._symmetry`` is set by SymmetricTensor; plain FlopscopeArrays
        do not have it (or it's None). Guarding via getattr keeps this
        method valid on both subclasses without a forward reference.
        """
        sym = getattr(self, "_symmetry", None)
        if sym is not None:
            raise ValueError(
                f"in-place {op_name} on a SymmetricTensor would break "
                f"symmetry on axes {sym.axes}; call fnp.{op_name}(arr) for "
                f"an unsymmetric copy instead."
            )

    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._check_inplace_breaks_symmetry("sort")
        result = _me().sort(self, *args, **kwargs)
        # Strip both self and result before np.copyto: keeps the
        # invariant ("never pass a flopscope subclass to a raw NumPy call")
        # explicit even though np.copyto is currently NOT in the
        # __array_function__ allowlist.
        _np.copyto(_to_base_ndarray(self), _to_base_ndarray(result))

    def partition(self, kth: Any, *args: Any, **kwargs: Any) -> None:
        self._check_inplace_breaks_symmetry("partition")
        result = _me().partition(self, kth, *args, **kwargs)
        _np.copyto(_to_base_ndarray(self), _to_base_ndarray(result))

    def _inplace_from_result(self, result, op_name):
        """Apply ``result`` into ``self`` in place; refuse if the
        operation would destroy or weaken symmetry metadata.

        Compare ``self.symmetry`` and ``result.symmetry`` directly via
        ``SymmetryGroup.__eq__`` (PR #51 made these value-equal with an
        identity short-circuit and per-instance canonical-action cache).
        Scalar in-place ops (``a += 1.0``) keep every group identically,
        so the comparison passes via the identity short-circuit and the
        copy proceeds cleanly.

        ``_to_base_ndarray(self)`` is required around ``np.copyto``
        because ``np.copyto`` could otherwise dispatch via
        ``__array_function__`` if it ever lands in the allowlist --
        without the strip, the call would recurse back through flopscope.
        """
        self_sym = getattr(self, "_symmetry", None)
        result_sym = getattr(result, "_symmetry", None)
        if self_sym is not None:
            if self_sym != result_sym:
                raise ValueError(
                    f"in-place {op_name} would destroy or weaken symmetry "
                    f"metadata on axes {self_sym.axes} (result symmetry: "
                    f"{result_sym.axes if result_sym is not None else None}); "
                    f"use ``self = fnp.{op_name}(self, other)`` to accept the "
                    f"new result explicitly."
                )
        _np.copyto(_to_base_ndarray(self), _to_base_ndarray(result))
        return self

    # ----- Binary arithmetic -----

    def __add__(self, other: Any) -> FlopscopeArray:
        return _me().add(self, other)

    def __radd__(self, other: Any) -> FlopscopeArray:
        return _me().add(other, self)

    def __iadd__(self, other: Any) -> FlopscopeArray:
        result = _me().add(self, other)
        return self._inplace_from_result(result, "add")

    def __sub__(self, other: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().subtract(self, other)

    def __rsub__(self, other: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().subtract(other, self)

    def __isub__(self, other: Any) -> FlopscopeArray:
        result = _me().subtract(self, other)
        return self._inplace_from_result(result, "subtract")

    def __mul__(self, other: Any) -> FlopscopeArray:
        return _me().multiply(self, other)

    def __rmul__(self, other: Any) -> FlopscopeArray:
        return _me().multiply(other, self)

    def __imul__(self, other: Any) -> FlopscopeArray:
        result = _me().multiply(self, other)
        return self._inplace_from_result(result, "multiply")

    def __truediv__(self, other: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().true_divide(self, other)

    def __rtruediv__(self, other: Any) -> FlopscopeArray:
        return _me().true_divide(other, self)

    def __itruediv__(self, other: Any) -> FlopscopeArray:
        result = _me().true_divide(self, other)
        return self._inplace_from_result(result, "true_divide")

    def __floordiv__(self, other: Any) -> FlopscopeArray:  # type: ignore[override]
        return _me().floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> FlopscopeArray:
        return _me().floor_divide(other, self)

    def __ifloordiv__(self, other: Any) -> FlopscopeArray:
        result = _me().floor_divide(self, other)
        return self._inplace_from_result(result, "floor_divide")

    def __mod__(self, other: Any) -> FlopscopeArray:
        return _me().mod(self, other)

    def __rmod__(self, other: Any) -> FlopscopeArray:
        return _me().mod(other, self)

    def __imod__(self, other: Any) -> FlopscopeArray:
        result = _me().mod(self, other)
        return self._inplace_from_result(result, "mod")

    def __pow__(self, other: Any) -> FlopscopeArray:
        return _me().power(self, other)

    def __rpow__(self, other: Any) -> FlopscopeArray:
        return _me().power(other, self)

    def __ipow__(self, other: Any) -> FlopscopeArray:
        result = _me().power(self, other)
        return self._inplace_from_result(result, "power")

    def __matmul__(self, other: Any) -> FlopscopeArray:
        return _me().matmul(self, other)

    def __rmatmul__(self, other: Any) -> FlopscopeArray:
        return _me().matmul(other, self)

    def __imatmul__(self, other: Any) -> FlopscopeArray:
        # __imatmul__ is special: matmul output shape may differ from
        # self.shape, in which case in-place mutation is impossible.
        # CPython's documented in-place fallback rebinds the name to the
        # new (out-of-place) result. NumPy raises ValueError on shape
        # mismatch; we follow the CPython fallback so typical pipelines
        # using ``A @= B`` to grow state work cleanly.
        result = _me().matmul(self, other)
        if result.shape != self.shape:
            return result
        return self._inplace_from_result(result, "matmul")

    # ----- Unary arithmetic -----

    def __neg__(self) -> FlopscopeArray:
        return _me().negative(self)

    def __pos__(self) -> FlopscopeArray:
        return _me().positive(self)

    def __abs__(self) -> FlopscopeArray:
        return _me().abs(self)

    def __invert__(self) -> FlopscopeArray:
        return _me().invert(self)

    # ----- Comparison -----

    def __eq__(self, other: Any) -> FlopscopeArray:
        return _me().equal(self, other)

    def __ne__(self, other: Any) -> FlopscopeArray:
        return _me().not_equal(self, other)

    def __lt__(self, other: Any) -> FlopscopeArray:
        return _me().less(self, other)

    def __le__(self, other: Any) -> FlopscopeArray:
        return _me().less_equal(self, other)

    def __gt__(self, other: Any) -> FlopscopeArray:
        return _me().greater(self, other)

    def __ge__(self, other: Any) -> FlopscopeArray:
        return _me().greater_equal(self, other)

    def __hash__(self) -> int:  # type: ignore[override]
        # numpy ndarray is unhashable; preserve that.
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    # ----- Bitwise -----

    def __and__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_and(self, other)

    def __rand__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_and(other, self)

    def __iand__(self, other: Any) -> FlopscopeArray:
        result = _me().bitwise_and(self, other)
        return self._inplace_from_result(result, "bitwise_and")

    def __or__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_or(self, other)

    def __ror__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_or(other, self)

    def __ior__(self, other: Any) -> FlopscopeArray:
        result = _me().bitwise_or(self, other)
        return self._inplace_from_result(result, "bitwise_or")

    def __xor__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_xor(self, other)

    def __rxor__(self, other: Any) -> FlopscopeArray:
        return _me().bitwise_xor(other, self)

    def __ixor__(self, other: Any) -> FlopscopeArray:
        result = _me().bitwise_xor(self, other)
        return self._inplace_from_result(result, "bitwise_xor")

    def __lshift__(self, other: Any) -> FlopscopeArray:
        return _me().left_shift(self, other)

    def __rlshift__(self, other: Any) -> FlopscopeArray:
        return _me().left_shift(other, self)

    def __ilshift__(self, other: Any) -> FlopscopeArray:
        result = _me().left_shift(self, other)
        return self._inplace_from_result(result, "left_shift")

    def __rshift__(self, other: Any) -> FlopscopeArray:
        return _me().right_shift(self, other)

    def __rrshift__(self, other: Any) -> FlopscopeArray:
        return _me().right_shift(other, self)

    def __irshift__(self, other: Any) -> FlopscopeArray:
        result = _me().right_shift(self, other)
        return self._inplace_from_result(result, "right_shift")


def wrap_module_returns(module, skip_names=None, check_module=True):
    """Patch all public callables in a module to wrap return values.

    Walks the module's namespace, finds public functions defined in
    that module, and replaces them with wrappers that convert any
    numpy.ndarray return value into a FlopscopeArray (zero-copy view).

    Tuple/list of arrays are also handled element-wise.

    Args:
        module: The module object to patch.
        skip_names: Optional set of function names to leave unwrapped
                    (e.g. functions that return scalars or shape tuples).
        check_module: If True (default), only wrap functions whose
                      __module__ matches the module being patched.
                      Set to False for modules that re-export from
                      sub-modules (e.g. flopscope.numpy.linalg).
    """
    import functools

    skip = set(skip_names or ())

    for name in list(vars(module)):
        if name.startswith("_") or name in skip:
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        if check_module:
            if not hasattr(obj, "__module__") or obj.__module__ != module.__name__:
                continue

        original = obj

        @functools.wraps(original)
        def wrapped(*args, _orig=original, **kwargs):
            result = _orig(*args, **kwargs)
            if isinstance(result, _np.ndarray):
                return _asflopscope(result)
            if isinstance(result, tuple):
                wrapped_elems = [
                    _asflopscope(r) if isinstance(r, _np.ndarray) else r for r in result
                ]
                # Preserve named tuple type (e.g. UniqueAllResult).
                if type(result) is not tuple and hasattr(type(result), "_fields"):
                    return type(result)(*wrapped_elems)
                return tuple(wrapped_elems)
            if isinstance(result, list):
                return [
                    _asflopscope(r) if isinstance(r, _np.ndarray) else r for r in result
                ]
            return result

        wrapped.__wrapped__ = original

        setattr(module, name, wrapped)


def _asflopscope(x):
    """Convert any array-like to FlopscopeArray without forcing ownership copies.

    - FlopscopeArray: returned as-is
    - numpy.ndarray subclass (e.g. SymmetricTensor): returned as-is to
      preserve subclass metadata
    - plain numpy.ndarray: view-cast to FlopscopeArray (zero-copy)
    - other: np.asarray first, then view-cast (also zero-copy with
      respect to the ndarray returned by ``np.asarray``)

    Flopscope deliberately does not promise ``OWNDATA`` parity for subclass
    results. Avoiding extra copies is preferred because this conversion
    sits on a hot path for many small-array operations.
    """
    if isinstance(x, FlopscopeArray):
        return x
    if type(x) is not _np.ndarray and isinstance(x, _np.ndarray):
        # Other ndarray subclass (e.g. SymmetricTensor) — preserve as-is.
        return x
    if isinstance(x, _np.ndarray):
        return x.view(FlopscopeArray)
    arr = _np.asarray(x)
    return arr.view(FlopscopeArray)


def _asplainflopscope(x):
    """Convert any array-like to a base FlopscopeArray.

    Unlike :func:`_asflopscope`, this always drops ndarray subclasses so callers can
    explicitly return a plain tracked array after metadata becomes invalid.
    As with :func:`_asflopscope`, this is intentionally no-copy: the result may
    report ``OWNDATA=False`` even when the underlying base ndarray owns the
    data.
    """
    arr = _np.asarray(x)
    return arr.view(FlopscopeArray)


def _to_base_ndarray(x):
    """View a flopscope array as a plain ``np.ndarray`` (zero-copy).

    Distinct from :func:`_asplainflopscope` (which returns a ``FlopscopeArray`` —
    still a numpy subclass that triggers ``__array_function__``).
    Required before calling ``_np.<func>(x)`` from inside our own
    ``__array_ufunc__`` / ``__array_function__`` handlers, so that
    NumPy's protocol dispatch (which would route FlopscopeArray inputs back
    through ``me.<func>``) does not recurse infinitely.

    Only ``FlopscopeArray`` instances (and subclasses like
    ``SymmetricTensor``) are stripped — other ``numpy.ndarray``
    subclasses (e.g. ``np.matrix``, user-defined ``ArraySubclass``)
    pass through unchanged so NumPy's standard subclass-propagation
    behaviour preserves their type on the result.

    Non-array inputs (Python scalars, lists) pass through unchanged so
    that NEP 50 weak-typing rules continue to apply when flopscope wrappers
    forward these to NumPy.

    Examples
    --------
    >>> a = FlopscopeArray((4,), dtype=float)
    >>> type(_to_base_ndarray(a)) is _np.ndarray
    True
    >>> _to_base_ndarray(2.0) == 2.0
    True
    """
    if isinstance(x, FlopscopeArray):
        return x.view(_np.ndarray)
    return x


def _to_base_ndarray_tree(x):
    """Recursively strip flopscope subclasses from arrays inside tuples/lists.

    Use for arguments that accept *containers* of arrays passed through
    ``__array_function__``:
    - ``np.lexsort(keys)`` — keys is a sequence of arrays.
    - ``np.meshgrid(*xi)`` — xi is a tuple of arrays.
    - ``np.concatenate(arrays)`` / ``np.stack(arrays)`` — arrays is a sequence.
    - ``out=(out1, out2)`` — multi-output ufunc out tuples.

    Plain ``_to_base_ndarray`` only handles a single array; using it on
    a list of FlopscopeArrays leaves the inner FlopscopeArrays intact and
    recursion can still happen through them.

    Preserves namedtuples (e.g. ``UniqueAllResult``) by re-constructing
    the type with stripped fields.

    Intentional scope: tuples, lists, namedtuples, and bare arrays only.
    Dicts and other generic mappings are NOT recursed into.

    Only ``FlopscopeArray`` instances (and subclasses like
    ``SymmetricTensor``) are stripped — other ``numpy.ndarray``
    subclasses (e.g. ``np.matrix``, user-defined ``ArraySubclass``)
    pass through unchanged so NumPy's standard subclass-propagation
    behaviour preserves their type on the result.
    """
    if isinstance(x, FlopscopeArray):
        return x.view(_np.ndarray)
    if isinstance(x, tuple):
        stripped = tuple(_to_base_ndarray_tree(e) for e in x)
        # Preserve namedtuple type if present.
        if type(x) is not tuple and hasattr(type(x), "_fields"):
            return type(x)(*stripped)
        return stripped
    if isinstance(x, list):
        return [_to_base_ndarray_tree(e) for e in x]
    return x


@_functools.cache
def _signature_kwargs_accepted(np_func):
    """Return frozenset of kwarg names accepted by ``np_func``.

    Returns:
    - ``None`` if ``np_func`` is ``None`` or signature inspection fails.
    - Empty frozenset if ``np_func`` accepts ``**kwargs`` (sentinel meaning
      "accepts every kwarg name; do not filter").
    - Otherwise a frozenset of accepted parameter names.

    Cached: NumPy callable identities are stable for the process
    lifetime, so ``@functools.cache`` is safe and necessary on this hot
    path. PR #51 memoized similar per-call introspection
    (``unique_elements_for_shape``, ``_canonical_axis_action``); this
    follows the same pattern. See PR #51 perf comment for why per-call
    ``inspect.signature`` is unacceptable.
    """
    if np_func is None:
        return None
    try:
        sig = _inspect.signature(np_func)
    except (ValueError, TypeError):
        return None
    for p in sig.parameters.values():
        if p.kind == _inspect.Parameter.VAR_KEYWORD:
            return frozenset()  # sentinel: accepts everything
    return frozenset(
        n
        for n, p in sig.parameters.items()
        if p.kind != _inspect.Parameter.VAR_POSITIONAL
    )


def _filter_to_np_signature(np_func, kwargs):
    """Drop kwargs that ``np_func`` does not accept.

    NumPy's ``ufunc.reduce`` always supplies ``dtype=`` / ``keepdims=`` /
    ``out=``, but the equivalent function-form wrapper (``np.all``,
    ``np.any``, etc.) accepts only a subset. Forwarding the full
    ufunc-reduce kwarg set to those function-form wrappers raises
    ``TypeError``.

    Falls back to the original kwargs when ``inspect.signature`` cannot
    introspect (e.g. C-implemented functions). Per-function signature
    lookup is cached via :func:`_signature_kwargs_accepted` — this is on
    the per-ufunc-call hot path; uncached lookup would be a perf cliff.
    """
    accepted = _signature_kwargs_accepted(np_func)
    if accepted is None:
        return kwargs
    if not accepted:  # empty frozenset = sentinel for **kwargs accepted
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}
