"""No user-facing string may name the predecessor brand.

flopscope was renamed from `whest`, but several messages a participant can
actually see still said `WhestArray` — a class that is not in the public API
and that nobody using this package can look up. This guard is a grep, not a
behavioural test, because the failure mode is textual: the names survive in
strings that no type checker or import ever touches.

Internal identifiers are deliberately out of scope here; this pins only what
reaches a caller's terminal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import flopscope

SRC = Path(flopscope.__file__).parent
FORBIDDEN = ("whest", "Whest", "WHEST")


def _string_literals(path: Path):
    """Every string literal in the module, with its line number.

    Docstrings are included: they are rendered by `help()` and by the docs
    site, so they are user-facing too.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


SOURCE_FILES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_predecessor_brand_in_string_literals(path: Path):
    hits = [
        (lineno, text.strip()[:90])
        for lineno, text in _string_literals(path)
        for bad in FORBIDDEN
        # `whestbench` is a real, separate package; naming it is correct.
        if bad in text and "whestbench" not in text.lower()
    ]
    assert not hits, (
        f"{path.relative_to(SRC)} has user-facing text naming the predecessor "
        f"brand (use FlopscopeArray / flopscope): {hits}"
    )
