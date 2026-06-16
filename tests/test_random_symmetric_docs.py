"""Regression test for issue #50.

``flopscope.numpy.random.symmetric`` is flopscope-only: upstream NumPy has no
``numpy.random.symmetric``. The doc builder must therefore extract the
summary/sections from the REAL flopscope callable, not from the ``None``
upstream object. ``inspect.getdoc(None)`` returns ``NoneType``'s docstring
("The type of the None singleton."), which previously shadowed the real
docstring and broke ``generate_api_docs.py --verify``.

This builds a one-entry operation registry so the assertion stays fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "gen", ROOT / "scripts" / "generate_api_docs.py"
)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


def _build_one(name: str) -> object:
    """Build the operation doc record for a single op via the real builder."""
    registry = gen.load_registry()
    assert name in registry, f"{name} missing from operation registry"
    one = {name: registry[name]}
    records = gen.build_operation_doc_records(one, workers=1)
    assert len(records) == 1, f"expected one record for {name}, got {len(records)}"
    return records[0]


def test_random_symmetric_record_uses_real_docstring() -> None:
    record = _build_one("random.symmetric")

    summary = getattr(record, "summary", "") or ""
    assert "None singleton" not in summary, (
        "summary resolved to NoneType docstring instead of the flopscope callable; "
        f"got {summary!r}"
    )
    assert "symmetry group" in summary.lower(), (
        f"expected the real flopscope summary, got {summary!r}"
    )

    # The real docstring carries Parameters/Returns/Examples; the NoneType
    # docstring carries none of them. Asserting the structured sections land
    # guards the full --verify contract, not just the summary line.
    parameters = getattr(record, "parameters", None) or []
    assert parameters, "expected a non-empty Parameters section from the real docstring"
