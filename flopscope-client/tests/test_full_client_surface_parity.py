"""Guard: every name CALLABLE in full flopscope.numpy must be callable in the
client OR raise a CLEAR (not opaque) error. This is the check that would have
caught the dtype-as-string bug (float32 callable in full, a non-callable str in
the client). Compares against a committed snapshot because the numpy-free client
cannot import full flopscope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flopscope.numpy as fnp

_SNAPSHOT = Path(__file__).parent / "fixtures" / "full_numpy_surface.json"


def _full_surface() -> dict[str, dict[str, bool]]:
    return json.loads(_SNAPSHOT.read_text())


# Names callable in full that we ACCEPT diverging with no clear error required.
# Keep empty; add only with explicit justification.
_ACCEPTED: frozenset[str] = frozenset()

# Substrings that mark an INTENTIONAL, actionable client exclusion.
_CLEAR_ERROR_MARKERS = (
    "not supported in the flopscope client",
    "server-side",
    "abstract scalar type",
    "registered but not yet implemented",
)

_CALLABLE_IN_FULL = sorted(
    name for name, meta in _full_surface().items() if meta["callable"]
)


@pytest.mark.parametrize("name", _CALLABLE_IN_FULL)
def test_full_callable_is_callable_or_clear_in_client(name):
    if name in _ACCEPTED:
        pytest.skip("accepted divergence")
    try:
        obj = getattr(fnp, name)
    except AttributeError as exc:
        msg = str(exc)
        assert any(m in msg for m in _CLEAR_ERROR_MARKERS), (
            f"fnp.{name} is callable in full flopscope but the client raises an "
            f"OPAQUE error: {msg!r}. Expose it, or give a clear, actionable error."
        )
        return
    assert callable(obj), (
        f"fnp.{name} is callable in full flopscope but the client exposes a "
        f"non-callable {type(obj).__name__} ({obj!r}). This is the dtype-as-"
        f"string class of bug."
    )


def test_snapshot_marks_float32_callable():
    # Tripwire: if the snapshot ever says float32 is non-callable, the guard
    # above would be a no-op for it.
    assert _full_surface()["float32"]["callable"] is True
