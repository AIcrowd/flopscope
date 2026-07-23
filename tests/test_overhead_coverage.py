"""AST-walk lint tests enforcing flopscope_overhead instrumentation coverage.

Two tests:
  - test_every_wrapper_with_budget_deduct_is_decorated: every function whose
    body calls budget.deduct(...) must carry @_counted_wrapper.
  - test_every_direct_numpy_call_in_wrapper_uses_call_numpy: every direct
    np.X(...) (or _np.X(...) etc.) call inside a `with budget.deduct(...):`
    block must go through _call_numpy(np_fn, ...).

Both tests fail initially. They will pass after Tasks 17-31 of the
flopscope_overhead migration plan land.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "flopscope"


def _functions_with_budget_deduct(path: Path) -> list[ast.FunctionDef]:
    """Return every FunctionDef in `path` whose body contains a call to
    ``budget.deduct(...)`` (any name, but resolves to a `deduct` attr call)."""
    tree = ast.parse(path.read_text())
    out: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "deduct"
            ):
                out.append(node)
                break
    return out


def _has_counted_wrapper_decorator(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "_counted_wrapper":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "_counted_wrapper":
            return True
    return False


def test_every_wrapper_with_budget_deduct_is_decorated():
    """Every function whose body calls budget.deduct(...) must carry
    @_counted_wrapper. Exceptions: BudgetContext.deduct itself, _call_numpy,
    and _einsum_routed_binary (a shared cost-routing helper invoked *by*
    already-decorated wrappers, so it must not be wrapped a second time)."""
    missing: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        for fn in _functions_with_budget_deduct(path):
            if fn.name in (
                "deduct",
                "_call_numpy",
                "_einsum_routed_binary",
                # Factory functions — their inner `wrapper` (decorated with
                # @_counted_wrapper) calls budget.deduct; the outer factory
                # is a closure builder, not itself a counted wrapper.
                "_counted_variance",
                "_counted_mean",
                # Inner per-element helper inside atleast_1d/2d/3d — the outer
                # function carries @_counted_wrapper; _one is a closure that
                # issues one deduct per variadic argument.
                "_one",
                # Shared save/savez/savez_compressed egress-billing formula
                # (src/flopscope/_io.py), called by those already-decorated
                # wrappers AND by flopscope-server's request handler (a plain
                # function call outside any wrapper) -- must not be wrapped
                # a second time, same shape as _einsum_routed_binary above.
                "_bill_save_egress",
            ):
                continue
            if not _has_counted_wrapper_decorator(fn):
                rel = path.relative_to(SRC_ROOT.parent.parent)
                missing.append(f"{rel}:{fn.lineno} {fn.name}")
    assert not missing, (
        "Wrappers calling budget.deduct(...) without @_counted_wrapper:\n  "
        + "\n  ".join(missing)
    )


_NUMPY_MODULE_NAMES = {"_np", "np", "_npr"}
_ALLOWED_CALLERS = {
    "_call_numpy",
    "_call_with_optional_out",
    "_call_with_optional_multi_out",
    "_execute_pairwise",
    # Pure-Python context managers / non-compute helpers — allowed inside
    # `with budget.deduct(...):` blocks because they don't represent the
    # numpy compute being timed:
    "errstate",  # numpy.errstate() — fp-error context manager
}


def _direct_numpy_calls_inside_with_deduct(path: Path) -> list[tuple[int, str]]:
    """Find ``_np.X(...)`` style calls inside ``with budget.deduct(...):`` blocks
    that aren't routed through ``_call_numpy``. Returns [(lineno, func_repr)]."""
    tree = ast.parse(path.read_text())
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        # Is at least one of the with-items a budget.deduct(...) call?
        is_deduct_block = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "deduct"
            for item in node.items
        )
        if not is_deduct_block:
            continue
        # Walk the body, find Call nodes whose func resolves to a numpy module.
        body_module = ast.Module(body=node.body, type_ignores=[])
        for sub in ast.walk(body_module):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            # Allow the listed helpers.
            if isinstance(f, ast.Name) and f.id in _ALLOWED_CALLERS:
                continue
            if isinstance(f, ast.Attribute) and f.attr in _ALLOWED_CALLERS:
                continue
            # Walk the attribute chain to find the leftmost Name; flag if it's
            # a numpy-module alias.
            root = f
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _NUMPY_MODULE_NAMES:
                out.append((sub.lineno, ast.unparse(f)))
    return out


def test_every_direct_numpy_call_in_wrapper_uses_call_numpy():
    """Every numpy invocation inside `with budget.deduct(...):` must go through
    `_call_numpy(np_fn, ...)` so its wall time is correctly attributed to
    flopscope backend time. Direct `_np.X(...)` is rejected."""
    bad: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        for lineno, repr_ in _direct_numpy_calls_inside_with_deduct(path):
            rel = path.relative_to(SRC_ROOT.parent.parent)
            bad.append(f"{rel}:{lineno}  {repr_}(...)")
    assert not bad, (
        "Direct numpy calls inside `with budget.deduct(...):` not using "
        "_call_numpy:\n  " + "\n  ".join(bad)
    )
