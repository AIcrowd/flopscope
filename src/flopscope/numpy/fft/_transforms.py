# src/flopscope/fft/_transforms.py
"""FFT transform wrappers with FLOP counting.

Cost model: 5 * N * log2(N) for complex DFT of length N (radix-2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import fft_billing_dtype
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._validation import _normalize_out, require_budget

# numpy 2.1 reversed the remaining-axis execution order that rfftn/rfft2 use
# after the real FFT on the last axis (forward on 2.0.x, reversed from 2.1
# onward); staged_fftn_cost's r2c branch replays that order exactly, so it
# must branch on the installed numpy version to stay correct across 2.0-2.4.
_NUMPY_GE_2_1 = tuple(int(x) for x in _np.__version__.split(".")[:2]) >= (2, 1)


def fft_cost(n: int) -> int:
    """FLOP cost of a 1-D complex FFT.

    Parameters
    ----------
    n : int
        Transform length.

    Returns
    -------
    int
        Estimated FLOP count: 5 * n * ceil(log2(n)).

    Notes
    -----
    Assumes radix-2 algorithm.
    """
    if n <= 1:
        return 0
    return 5 * n * math.ceil(math.log2(n))


def rfft_cost(n: int) -> int:
    """FLOP cost of a 1-D real FFT.

    Parameters
    ----------
    n : int
        Input length (real-valued).

    Returns
    -------
    int
        Estimated FLOP count: 5 * (n // 2) * ceil(log2(n)).

    Notes
    -----
    The real FFT exploits conjugate symmetry, roughly halving the work
    compared to a full complex FFT.
    """
    if n <= 1:
        return 0
    return 5 * (n // 2) * math.ceil(math.log2(n))


def fftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of an N-D complex FFT.

    Parameters
    ----------
    shape : tuple of int
        Transform shape along each axis.

    Returns
    -------
    int
        Estimated FLOP count: 5 * N * sum(ceil(log2(d_i))), where N = prod(shape).

    Notes
    -----
    The multi-dimensional FFT is computed as successive 1-D FFTs along each
    axis. The cost along axis *i* is 5 * (N / d_i) * d_i * ceil(log2(d_i))
    = 5 * N * ceil(log2(d_i)). Summing over axes gives
    5 * N * sum(ceil(log2(d_i))).
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    log_sum = sum(math.ceil(math.log2(d)) for d in shape if d > 1)
    return 5 * N * log_sum


def rfftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of an N-D real FFT.

    Parameters
    ----------
    shape : tuple of int
        Transform shape along each axis.

    Returns
    -------
    int
        Estimated FLOP count: 5 * (N // 2) * sum(ceil(log2(d_i))), where N = prod(shape).

    Notes
    -----
    Exploits conjugate symmetry along the last axis, roughly halving
    the work compared to a full complex N-D FFT.
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    log_sum = sum(math.ceil(math.log2(d)) for d in shape if d > 1)
    return 5 * (N // 2) * log_sum


def staged_fftn_cost(
    input_shape: tuple[int, ...],
    s_resolved: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    kind: str,
) -> int:
    """FLOP cost of an N-D FFT, replaying numpy's exact 1-D execution cascade.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input array.
    s_resolved : tuple of int
        Resolved transform length for each entry in *axes* (same order,
        same length; sentinel values such as `None`/`-1` already resolved
        by the caller).
    axes : tuple of int
        Transform axes, normalized to non-negative positions, same order
        and length as *s_resolved*.
    kind : str
        One of ``"c2c"`` (complex-to-complex: fftn/ifftn/fft2/ifft2),
        ``"r2c"`` (real-to-complex: rfftn/rfft2), or ``"c2r"``
        (complex-to-real: irfftn/irfft2).

    Returns
    -------
    int
        Sum of the per-stage 1-D FLOP costs, computed over the same
        sequence of intermediate array shapes numpy's pocketfft backend
        actually produces: one 1-D transform per axis, applied in numpy's
        real execution order (reversed for c2c; real axis first for r2c;
        real axis last for c2r), with the batch size at each stage taken
        from the *current* (not final) shape.

    Notes
    -----
    numpy computes an N-D FFT as a cascade of 1-D FFTs over evolving
    intermediate shapes rather than a single N-D butterfly. Pricing from
    only the final transform shape misprices any case where an axis is
    resized (via `s`), because the batch count at each stage depends on
    the shape *at that point in the cascade*, not the final shape. This
    helper replays the cascade with the same `fft_cost`/`rfft_cost`
    primitives the 1-D wrappers use, so the fused N-D bill equals the sum
    of the equivalent explicit 1-D calls by construction.
    """
    current = [int(d) for d in input_shape]

    def stage(axis: int, n: int, real: bool) -> int:
        batch = 1
        for i, d in enumerate(current):
            if i != axis:
                batch *= d
        return batch * (rfft_cost(n) if real else fft_cost(n))

    total = 0
    if kind == "c2c":  # numpy _raw_fftnd: reverse axis order
        for pos in reversed(range(len(axes))):
            ax, n = axes[pos], s_resolved[pos]
            total += stage(ax, n, False)
            current[ax] = n
    elif kind == "r2c":  # rfft last axis first, then remaining c2c axes
        # (reversed from numpy 2.1 onward; forward on 2.0.x -- see
        # _NUMPY_GE_2_1)
        ax, n = axes[-1], s_resolved[-1]
        total += stage(ax, n, True)
        current[ax] = n // 2 + 1
        remaining = range(len(axes) - 1)
        remaining = reversed(remaining) if _NUMPY_GE_2_1 else remaining
        for pos in remaining:
            ax, n = axes[pos], s_resolved[pos]
            total += stage(ax, n, False)
            current[ax] = n
    else:  # c2r: c2c forward, irfft last axis last
        for pos in range(len(axes) - 1):
            ax, n = axes[pos], s_resolved[pos]
            total += stage(ax, n, False)
            current[ax] = n
        ax, n = axes[-1], s_resolved[-1]
        total += stage(ax, n, True)
        current[ax] = n
    return total


def hfft_cost(n_out: int) -> int:
    """FLOP cost of a Hermitian FFT.

    Parameters
    ----------
    n_out : int
        Output length.

    Returns
    -------
    int
        Estimated FLOP count: 5 * (n_out // 2) * ceil(log2(n_out)).

    Notes
    -----
    hfft(a, n) == irfft(conj(a), n): real-output c2r transform.
    Exploits conjugate symmetry, roughly halving the work vs a full complex FFT.
    """
    if n_out <= 1:
        return 0
    return 5 * (n_out // 2) * math.ceil(math.log2(n_out))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _batch_count_1d(a: _np.ndarray, axis: int) -> int:
    """Number of independent 1-D transforms along *axis*."""
    if a.ndim == 0 or a.shape[axis] == 0:
        return 1
    return a.size // a.shape[axis]


# 1-D transforms
@_counted_wrapper
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.fft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * fft_cost(n)
    with budget.deduct(
        "fft.fft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.fft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    fft, _np.fft.fft, "fft", "5 \u00d7 n \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


@_counted_wrapper
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.ifft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * fft_cost(n)
    with budget.deduct(
        "fft.ifft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.ifft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    ifft, _np.fft.ifft, "fft", "5 \u00d7 n \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


@_counted_wrapper
def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.rfft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * rfft_cost(n)
    with budget.deduct(
        "fft.rfft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.rfft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    rfft, _np.fft.rfft, "fft", "5 \u00d7 (n/2) \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


@_counted_wrapper
def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.irfft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = _batch_count_1d(a, axis) * rfft_cost(n)
    with budget.deduct(
        "fft.irfft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.irfft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    irfft, _np.fft.irfft, "fft", "5 \u00d7 (n/2) \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


# 2-D transforms
@_counted_wrapper
def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.fft2")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # fftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.fft2",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.fft2,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    fft2,
    _np.fft.fft2,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.ifft2")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # fftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.ifft2",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.ifft2,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    ifft2,
    _np.fft.ifft2,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.rfft2")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # rfftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="r2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.rfft2",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.rfft2,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    rfft2,
    _np.fft.rfft2,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.irfft2")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        # Hermitian reconstruction: every transform axis uses the input size
        # except the last, which doubles (2*(m-1)) to recover the real
        # output length numpy assumes when `s` is omitted entirely.
        last = len(eff) - 1
        s_for_cost = tuple(
            int(a.shape[ax]) if i < last else 2 * (int(a.shape[ax]) - 1)
            for i, ax in enumerate(eff)
        )
    else:
        # `None` in the last position still means "omitted" -> doubles, same
        # as numpy's irfft(n=None) default. `-1` (any position, including
        # last) means "use the input size as-is" -- numpy resolves it
        # BEFORE the Hermitian reconstruction, so it is never doubled (see
        # numpy.fft._pocketfft._cook_nd_args). Passing either sentinel
        # through unresolved collapses rfftn_cost's `prod(shape)` to <= 1
        # and bills 0.
        last = len(s) - 1
        s_for_cost = tuple(
            int(sv)
            if (sv is not None and sv >= 0)
            else (
                2 * (int(a.shape[ax]) - 1)
                if (sv is None and i == last)
                else int(a.shape[ax])
            )
            for i, (sv, ax) in enumerate(zip(s, eff, strict=True))
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2r")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.irfft2",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.irfft2,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    irfft2,
    _np.fft.irfft2,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


# N-D transforms
@_counted_wrapper
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.fftn")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # fftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.fftn",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.fftn,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    fftn,
    _np.fft.fftn,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.ifftn")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # fftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.ifftn",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.ifftn,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    ifftn,
    _np.fft.ifftn,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.rfftn")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        s_for_cost = tuple(int(a.shape[ax]) for ax in eff)
    else:
        # numpy resolves both `None` and `-1` in `s` to the input size along
        # the corresponding transform axis (NumPy >= 2.0 treats `-1` as "use
        # the whole input"); a raw `-1`/`None` passed straight into
        # rfftn_cost's `prod(shape)` collapses N to <= 1 and bills 0.
        s_for_cost = tuple(
            int(a.shape[ax]) if (sv is None or sv < 0) else int(sv)
            for sv, ax in zip(s, eff, strict=True)
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="r2c")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.rfftn",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.rfftn,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    rfftn,
    _np.fft.rfftn,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


@_counted_wrapper
def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.irfftn")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if axes is None:
        eff = (
            tuple(range(a.ndim)) if s is None else tuple(range(a.ndim - len(s), a.ndim))
        )
    else:
        eff = tuple(axes)
    if s is None:
        # Hermitian reconstruction: every transform axis uses the input size
        # except the last, which doubles (2*(m-1)) to recover the real
        # output length numpy assumes when `s` is omitted entirely.
        last = len(eff) - 1
        s_for_cost = tuple(
            int(a.shape[ax]) if i < last else 2 * (int(a.shape[ax]) - 1)
            for i, ax in enumerate(eff)
        )
    else:
        # `None` in the last position still means "omitted" -> doubles, same
        # as numpy's irfft(n=None) default. `-1` (any position, including
        # last) means "use the input size as-is" -- numpy resolves it
        # BEFORE the Hermitian reconstruction, so it is never doubled (see
        # numpy.fft._pocketfft._cook_nd_args). Passing either sentinel
        # through unresolved collapses rfftn_cost's `prod(shape)` to <= 1
        # and bills 0.
        last = len(s) - 1
        s_for_cost = tuple(
            int(sv)
            if (sv is not None and sv >= 0)
            else (
                2 * (int(a.shape[ax]) - 1)
                if (sv is None and i == last)
                else int(a.shape[ax])
            )
            for i, (sv, ax) in enumerate(zip(s, eff, strict=True))
        )
    axes_nn = tuple(ax % a.ndim for ax in eff)
    cost = staged_fftn_cost(a.shape, s_for_cost, axes_nn, kind="c2r")  # type: ignore[reportArgumentType]
    with budget.deduct(
        "fft.irfftn",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.irfftn,
            _to_base_ndarray(a),
            s=s,
            axes=axes,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    irfftn,
    _np.fft.irfftn,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


# Hermitian transforms
@_counted_wrapper
def hfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.hfft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = _batch_count_1d(a, axis) * hfft_cost(n)
    with budget.deduct(
        "fft.hfft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.hfft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    hfft,
    _np.fft.hfft,
    "fft",
    "5 \u00d7 (n_out/2) \u00d7 \u2308log\u2082(n_out)\u2309 FLOPs",
)


@_counted_wrapper
def ihfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: ArrayLike | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "fft.ihfft")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * hfft_cost(n)
    with budget.deduct(
        "fft.ihfft",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(fft_billing_dtype(a.dtype),),
    ):
        result = _call_numpy(
            _np.fft.ihfft,
            _to_base_ndarray(a),
            n=n,
            axis=axis,
            norm=norm,  # type: ignore[reportArgumentType]
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[reportArgumentType]
        )
    return out if out is not None else result  # type: ignore[reportReturnType]


attach_docstring(
    ihfft,
    _np.fft.ihfft,
    "fft",
    "5 \u00d7 (n/2) \u00d7 \u2308log\u2082(n)\u2309 FLOPs",
)

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
