"""Exception and warning classes for flopscope."""

from __future__ import annotations

import os
import sys as _sys
import warnings as _warnings

_DEFAULT_DOCS_ROOT = "https://aicrowd.github.io/flopscope/docs"
_BUDGET_DOCS_PATH = "/guides/budget-planning"
_COMPETITION_DOCS_PATH = "/getting-started/competition"
_SYMMETRY_DOCS_PATH = "/guides/symmetry"


def _docs_url(path: str) -> str:
    root = os.environ.get("FLOPSCOPE_DOCS_ROOT", "").strip() or _DEFAULT_DOCS_ROOT
    return f"{root.rstrip('/')}{path}"


class FlopscopeError(Exception):
    """Base exception for all flopscope errors."""


class BudgetExhaustedError(FlopscopeError):
    """Raised when an operation would exceed the FLOP budget."""

    def __init__(self, op_name: str, *, flop_cost: int, flops_remaining: int):
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.flops_remaining = flops_remaining
        super().__init__(
            f"{op_name} would cost {flop_cost:,} FLOPs but only "
            f"{flops_remaining:,} remain. "
            f"See: {_docs_url(_BUDGET_DOCS_PATH)}"
        )


class TimeExhaustedError(FlopscopeError):
    """Raised when an operation exceeds the wall-clock time limit."""

    def __init__(self, op_name: str, *, elapsed_s: float, limit_s: float):
        self.op_name = op_name
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        super().__init__(
            f"{op_name}: wall-clock time {elapsed_s:.3f}s exceeds "
            f"limit {limit_s:.3f}s. "
            f"See: {_docs_url(_BUDGET_DOCS_PATH)}"
        )


class NoBudgetContextError(FlopscopeError):
    """Raised when a counted operation is called outside a BudgetContext."""

    def __init__(self):
        super().__init__(
            "No active BudgetContext. "
            "Wrap your code in `with flopscope.BudgetContext(...):`  "
            f"See: {_docs_url(_COMPETITION_DOCS_PATH)}"
        )


class SymmetryError(FlopscopeError):
    """Raised when a claimed tensor symmetry does not hold."""

    def __init__(
        self,
        axes: tuple[int, ...],
        max_deviation: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ):
        self.axes = axes
        self.max_deviation = max_deviation
        self.atol = atol
        self.rtol = rtol
        super().__init__(
            f"Tensor not symmetric along axes ({', '.join(str(d) for d in axes)}): "
            f"max deviation = {max_deviation} "
            f"(tolerance: atol={atol}, rtol={rtol}). "
            f"See: {_docs_url(_SYMMETRY_DOCS_PATH)}"
        )


class UnsupportedFunctionError(FlopscopeError):
    """Raised when calling a function not available in the installed NumPy.

    Use ``min_version`` for functions that require a newer numpy than installed
    (e.g. ``bitwise_count`` requires numpy >= 2.1). Use ``max_version`` for
    functions that have been removed in a newer numpy than installed
    (e.g. ``in1d`` was removed in numpy 2.4). ``replacement`` optionally names
    the function users should call instead.
    """

    def __init__(
        self,
        func_name: str,
        *,
        min_version: str | None = None,
        max_version: str | None = None,
        replacement: str | None = None,
    ):
        import numpy as _np

        self.func_name = func_name
        self.min_version = min_version
        self.max_version = max_version
        self.replacement = replacement

        if max_version is not None:
            if replacement is not None:
                msg = (
                    f"numpy.{func_name} was removed in numpy {max_version}; "
                    f"use `{replacement}` instead "
                    f"(you have numpy {_np.__version__})."
                )
            else:
                msg = (
                    f"numpy.{func_name} was removed in numpy {max_version} "
                    f"(you have numpy {_np.__version__})."
                )
        elif min_version is not None:
            msg = (
                f"numpy.{func_name} requires numpy >= {min_version} "
                f"(you have numpy {_np.__version__}). "
                f"To use it: uv pip install 'numpy>={min_version}'"
            )
        else:
            msg = f"numpy.{func_name} is not supported by flopscope."

        super().__init__(msg)


class UnsupportedDtypeError(TypeError):
    """An operation's dtype cannot be billed.

    Raised in three cases: (1) a participating dtype is non-numeric --
    ``dtype.kind`` outside the numeric allowlist ``"biufc"`` (bool, integer,
    float, complex), which covers object as well as string, bytes,
    structured/void, datetime64, and timedelta64 -- refused regardless of
    which weight table is loaded; (2) production dtype rates are active and
    the resolved calculation dtype is a numeric type absent from the
    supported table (a future dtype numpy or an extension package might
    introduce); or (3) a complex-dtype call reaches an op marked
    complex-illegal.
    """


class UnsupportedReturnType(FlopscopeError):
    """Raised when an op's result cannot be serialized across the client/server boundary.

    The op executed successfully on the server, but its return type is not one
    the response packer knows how to send back (and is not msgpack-native).
    Surfaced loudly — rather than silently degrading to ``str()`` — so such gaps
    are caught by the conformance tests, not by participants.
    """


class UnauthorizedControlError(FlopscopeError):
    """Raised when a session-lifecycle op (budget_open/budget_close) is sent
    without the trusted control token. Session lifecycle is grader-controlled;
    participant code cannot open, close, or reset a budget."""


class FlopscopeWarning(UserWarning):
    """Warning issued when flopscope detects potential numerical issues."""


class SymmetryLossWarning(FlopscopeWarning):
    """Warning issued when an operation causes loss of symmetry metadata."""


class CostFallbackWarning(FlopscopeWarning):
    """Warning issued when flopscope skips its symmetry-aware cost adjustment.

    The output's symmetry group is too large for the placeholder cost
    model's Burnside enumeration — its order ``|G|`` exceeds the configured
    ``dimino_budget``. The op runs correctly with the dense cost charged
    instead — only the FLOP accounting is conservative; the data is
    unaffected. Common trigger: ``np.ones((1,)*n)`` for large ``n`` or
    other auto-inferred ``S_n`` symmetries on degenerate shapes.

    Tune the threshold with ``flops.configure(dimino_budget=...)``. Suppress
    the warning with ``flops.configure(symmetry_warnings=False)`` (shares
    the flag with :class:`SymmetryLossWarning` since both are
    symmetry-related diagnostics).
    """


class RemoteCallbackWarning(FlopscopeWarning):
    """Warning issued when a callback-taking op (e.g. ``apply_along_axis``) is
    called in-process. The op works in-process but raises
    :class:`RemoteCallbackError` on the remote (client/server) backend used for
    AIcrowd submissions. Suppress with
    ``flops.configure(callback_warnings=False)``.
    """


class ConfigureNoOpWarning(FlopscopeWarning):
    """Warning issued when ``flops.configure()`` is called. ``configure``
    affects the in-process flopscope backend only; it is a no-op on
    ``flopscope-client`` and the evaluation servers, so its settings do not
    carry into a graded submission.
    """


# Used by _user_stacklevel() to skip frames inside the flopscope package.
_FLOPSCOPE_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_stacklevel() -> int:
    """Stacklevel for :func:`warnings.warn` that points to the first frame
    outside the ``flopscope`` package — i.e. the user's call site.

    Walks the active call stack starting from the caller of
    :func:`_warn_symmetry_loss`. Robust to changes in the number of
    decorator/wrapper layers between user code and the warn site.
    """
    frame = _sys._getframe(2)
    level = 2
    while frame is not None:
        if not frame.f_code.co_filename.startswith(_FLOPSCOPE_PKG_DIR):
            return level
        frame = frame.f_back
        level += 1
    return 3


def _warn_symmetry_loss(
    lost_dims: list[tuple[int, ...]],
    reason: str,
) -> None:
    """Emit a :class:`SymmetryLossWarning` if warnings are enabled."""
    from flopscope._config import get_setting

    if not get_setting("symmetry_warnings"):
        return
    dim_str = ", ".join(str(g) for g in lost_dims)
    _warnings.warn(
        f"Symmetry lost along dims {dim_str}: {reason}. "
        "Use as_symmetric() to re-tag if you know the result is symmetric. "
        "Suppress with flops.configure(symmetry_warnings=False).",
        SymmetryLossWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_remote_callback(op_name: str) -> None:
    """Emit a :class:`RemoteCallbackWarning` if ``callback_warnings`` is enabled."""
    from flopscope._config import get_setting

    if not get_setting("callback_warnings"):
        return
    _warnings.warn(
        f"{op_name}() runs a Python callback in-process, but raises "
        f"RemoteCallbackError on the remote (client/server) backend used for "
        f"AIcrowd submissions. Precompute the result or avoid this op in "
        f"submitted code. Suppress with flops.configure(callback_warnings=False).",
        RemoteCallbackWarning,
        stacklevel=_user_stacklevel(),
    )
