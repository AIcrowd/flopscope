"""Configurable per-operation FLOP weights for flopscope.

Flopscope supports official packaged weights, custom overrides via
``FLOPSCOPE_WEIGHTS_FILE``, and complete disabling via ``FLOPSCOPE_DISABLE_WEIGHTS=1``.
Packaged official weights are loaded automatically on import unless explicitly
overridden or disabled.

The JSON payload must have a ``"weights"`` key mapping
``op_name -> float multiplier``. It may also have a ``"dtype_rates"`` key
mapping ``dtype_name -> float rate`` (see :func:`get_dtype_rate`).

Active weights are loaded from the packaged ``data/default_weights.json`` — the
single source of truth for billed weights. The sibling ``data/weights.json`` and
``data/weights.csv`` are frozen calibration evidence and are NOT loaded at
runtime (see ``data/README.md``).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from importlib import resources

_ACTIVE_WEIGHTS: dict[str, float] = {}
_ACTIVE_DTYPE_RATES: dict[str, float] = {}
_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    _WARNED_MESSAGES.add(message)


def _weights_disabled() -> bool:
    value = os.environ.get("FLOPSCOPE_DISABLE_WEIGHTS")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_weights(data: dict, *, source: str) -> dict[str, float]:
    weights = data.get("weights")
    if not isinstance(weights, dict):
        raise ValueError(f"{source} is missing a valid 'weights' mapping")
    validated: dict[str, float] = {}
    for op_name, value in weights.items():
        if not isinstance(op_name, str):
            raise ValueError(f"{source} has a non-string operation name: {op_name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{source} weight for {op_name!r} must be a non-negative finite number"
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"{source} weight for {op_name!r} must be a non-negative finite number"
            )
        validated[op_name] = numeric_value
    return validated


def _extract_dtype_rates(data: dict, *, source: str) -> dict[str, float] | None:
    rates = data.get("dtype_rates")
    if rates is None:
        return None
    if not isinstance(rates, dict):
        raise ValueError(f"{source} has an invalid 'dtype_rates' mapping")
    validated: dict[str, float] = {}
    for name, value in rates.items():
        if not isinstance(name, str):
            raise ValueError(f"{source} has a non-string dtype name: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{source} dtype rate for {name!r} must be a non-negative finite number"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"{source} dtype rate for {name!r} must be a non-negative finite number"
            )
        validated[name] = numeric
    return validated


def _read_weights_file(path: str, *, source: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_packaged_weights_data() -> dict | None:
    """Read and parse the packaged ``default_weights.json`` resource.

    Returns the raw parsed payload (both the ``weights`` and ``dtype_rates``
    keys, unvalidated) so the caller can extract both. Returns ``None`` — and
    warns once — if the resource is missing or malformed as JSON.
    """
    try:
        resource = resources.files("flopscope").joinpath("data/default_weights.json")
        with resource.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive fallback path
        _warn_once(
            "Flopscope could not load packaged official weights "
            f"({exc}); falling back to unit weights (1.0 for all operations)."
        )
        return None


# Modern-Generator → legacy-module-level rename map (issue #18 follow-up).
# Methods whose names changed between the legacy API and the modern Generator API.
# Everything else uses simple prefix-stripping (no entry needed).
_RANDOM_METHOD_RENAMES: dict[str, str] = {
    "integers": "randint",  # Generator.integers → random.randint
    "random": "random_sample",  # Generator.random   → random.random_sample
}

# NumPy 2.x ufunc aliases → canonical name. Each alias IS the same ufunc object
# as its canonical twin (np.acos is np.arccos), so it does identical work and
# must bill the canonical's weight. Without this, an alias with no explicit JSON
# entry falls through to the 1.0 default — a bit-identical 16x substitution
# exploit (acos vs arccos). Resolving to the canonical keeps a single source of
# truth, so future alias additions stay in sync automatically.
_UFUNC_ALIAS_RENAMES: dict[str, str] = {
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atanh": "arctanh",
    "atan2": "arctan2",
    "pow": "power",
    "divmod": "floor_divide",  # divmod >= floor_divide work; 16.0 is a conservative floor
}

# Generic ufunc-method op names are synthesized at billing time as
# f"{ufunc.__name__}.{method}" (see flopscope._pointwise) and are deliberately
# NOT rows in the weights table: each method applies the BASE ufunc's
# arithmetic once per billed unit, so it inherits the base's per-application
# weight. Single-sourced here so the weight and the complex factor can never
# disagree about what counts as a ufunc method.
_UFUNC_METHOD_SUFFIXES: tuple[str, ...] = (
    ".reduce",
    ".accumulate",
    ".reduceat",
    ".outer",
    ".at",
)


def get_weight(op_name: str) -> float:
    """Return the FLOP weight multiplier for an operation.

    Parameters
    ----------
    op_name : str
        Operation name as passed to ``BudgetContext.deduct()``,
        e.g. ``"exp"``, ``"linalg.cholesky"``, ``"fft.fft"``.

    Returns
    -------
    float
        Multiplicative weight. Defaults to 1.0 if not configured.

    Notes
    -----
    For method-level random ops (``random.Generator.<m>`` and
    ``random.RandomState.<m>``), if no explicit weight is set, falls
    through to the legacy module-level weight via name-stripping plus
    ``_RANDOM_METHOD_RENAMES`` for the few renamed methods. This keeps
    same-physics ops at the same FLOP cost without requiring 94 explicit
    JSON entries (issue #18 follow-up).
    """
    # 1. Explicit entry — primary path.
    if op_name in _ACTIVE_WEIGHTS:
        return _ACTIVE_WEIGHTS[op_name]

    # 2. Method-level alias fallback for random.Generator.* / random.RandomState.*
    for prefix in ("random.Generator.", "random.RandomState."):
        if op_name.startswith(prefix):
            short = op_name[len(prefix) :]
            legacy = _RANDOM_METHOD_RENAMES.get(short, short)
            legacy_op = f"random.{legacy}"
            if legacy_op in _ACTIVE_WEIGHTS:
                return _ACTIVE_WEIGHTS[legacy_op]
            break

    # 3. Generic ufunc-method fallback. ``.at``/``.outer`` apply the base ufunc
    #    once per output cell; ``.reduce``/``.accumulate``/``.reduceat`` once
    #    per input element -- in every case the base ufunc's per-application
    #    weight IS the rate. Without this a heavier ufunc silently bills at the
    #    neutral 1.0 when reached through a method (the same substitution the
    #    alias fallback below exists to prevent).
    for suffix in _UFUNC_METHOD_SUFFIXES:
        if op_name.endswith(suffix):
            base = op_name[: -len(suffix)]
            if base in _ACTIVE_WEIGHTS or base in _UFUNC_ALIAS_RENAMES:
                return get_weight(base)
            break

    # 4. NumPy 2.x ufunc-alias fallback (acos→arccos, pow→power, ...). The alias
    #    is the same ufunc as its canonical twin, so it must bill the same weight
    #    rather than fall through to 1.0 (a 16x substitution exploit otherwise).
    canonical = _UFUNC_ALIAS_RENAMES.get(op_name)
    if canonical is not None and canonical in _ACTIVE_WEIGHTS:
        return _ACTIVE_WEIGHTS[canonical]

    # 5. Default fallback (unchanged).
    return 1.0


def get_dtype_rate(dtype_name: str) -> float:
    """Billing rate for a resolved dtype name.

    Unit mode (empty table — disabled weights, or test reset) rates
    everything 1.0. With a loaded table, unknown dtypes fail closed.
    """
    from flopscope.errors import UnsupportedDtypeError

    if not _ACTIVE_DTYPE_RATES:
        return 1.0
    try:
        return _ACTIVE_DTYPE_RATES[dtype_name]
    except KeyError:
        raise UnsupportedDtypeError(
            f"dtype {dtype_name!r} has no billing rate; supported dtypes: "
            f"{sorted(_ACTIVE_DTYPE_RATES)}"
        ) from None


def load_weights(path: str | None = None, *, use_packaged_default: bool = True) -> None:
    """Resolve and load active FLOP weights and per-dtype billing rates.

    Parameters
    ----------
    path : str or None
        Explicit path to a weights JSON file. If None, reads from the
        ``FLOPSCOPE_WEIGHTS_FILE`` environment variable when present.
    use_packaged_default : bool, optional
        Whether to fall back to the packaged official weights when no valid
        override is available.

    Notes
    -----
    Resolution order:

    1. If ``FLOPSCOPE_DISABLE_WEIGHTS=1``, all weights and dtype rates are
       disabled (unit mode).
    2. If an explicit path or ``FLOPSCOPE_WEIGHTS_FILE`` is provided and valid,
       it is used. If its payload has no ``dtype_rates`` key, a warning is
       emitted and dtype rates stay in unit mode.
    3. If the override is unusable, a warning is emitted and Flopscope falls back
       to the packaged official weights when enabled, otherwise to unit
       weights.
    4. If packaged weights are enabled but unavailable, a warning is emitted
       and Flopscope falls back to unit weights.
    """
    _ACTIVE_WEIGHTS.clear()
    _ACTIVE_DTYPE_RATES.clear()

    if _weights_disabled():
        return

    override_path = (
        path if path is not None else os.environ.get("FLOPSCOPE_WEIGHTS_FILE")
    )

    if override_path is not None:
        source = f"Custom weights file '{override_path}'"
        try:
            data = _read_weights_file(override_path, source=source)
            weights = _extract_weights(data, source=source)
            rates = _extract_dtype_rates(data, source=source)
        except Exception as exc:
            fallback_target = (
                "packaged official weights" if use_packaged_default else "unit weights"
            )
            _warn_once(
                f"Flopscope could not load custom weights from '{override_path}' "
                f"({exc}); falling back to {fallback_target}."
            )
        else:
            _ACTIVE_WEIGHTS.update(weights)
            if rates is None:
                _warn_once(
                    f"{source} has no 'dtype_rates'; using unit dtype rates (1.0)."
                )
            else:
                _ACTIVE_DTYPE_RATES.update(rates)
            return

    if not use_packaged_default:
        return

    packaged_data = _load_packaged_weights_data()
    if packaged_data is None:
        return

    try:
        packaged_weights = _extract_weights(
            packaged_data, source="Packaged official weights"
        )
        packaged_rates = _extract_dtype_rates(
            packaged_data, source="Packaged official weights"
        )
    except Exception as exc:  # pragma: no cover - defensive fallback path
        _warn_once(
            "Flopscope could not load packaged official weights "
            f"({exc}); falling back to unit weights (1.0 for all operations)."
        )
        return

    _ACTIVE_WEIGHTS.update(packaged_weights)
    if packaged_rates is not None:
        _ACTIVE_DTYPE_RATES.update(packaged_rates)


def reset_weights() -> None:
    """Clear all loaded weights and dtype rates. For testing."""
    _ACTIVE_WEIGHTS.clear()
    _ACTIVE_DTYPE_RATES.clear()
    _WARNED_MESSAGES.clear()


load_weights()
