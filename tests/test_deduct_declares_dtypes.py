"""Every deduct()/deduct_after() call site declares billing dtypes."""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "flopscope"


def _violations():
    out = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in {"deduct", "deduct_after"}
            ):
                continue
            if not any(kw.arg == "dtypes" for kw in node.keywords):
                out.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno}")
    return out


def test_all_deduct_sites_declare_dtypes():
    assert _violations() == []
