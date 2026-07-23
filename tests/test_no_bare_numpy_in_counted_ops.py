"""Regression guard: counted-op wrappers must not make bare result-producing
`_np.<fn>(...)` calls for real numpy work. Such work must flow through
`_call_numpy` (inside `deduct`/`deduct_after`) or `_call_user_code`, so it is
timed as backend / residual rather than misattributed to flopscope overhead.

This encodes the fix for the timing-misattribution bug (callbacks -> residual,
data-movement -> backend). Cheap O(1) dispatch/view/query/alloc calls are
allowlisted: they do negligible work, so leaving them bare is harmless.

The AST distinguishes the two forms exactly:
  - bare call (flagged):   `_np.tile(a, reps)`          -> ast.Call(func=Attribute)
  - helper arg (allowed):  `_call_numpy(_np.tile, ...)` -> _np.tile is a bare
    Attribute passed as an argument (no Call node on `_np.tile`).
"""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "flopscope"

# Counted-op modules whose @_counted_wrapper functions are scanned.
_MODULES = [
    "_counting_ops.py",
    "_array_ops.py",
    "_pointwise.py",
    "_sorting_ops.py",
    "_polynomial.py",
    "_window.py",
]

# numpy calls that may remain bare inside a counted wrapper: O(1) dispatch,
# shape/dtype queries, view ops, allocation, predicates, accounting helpers,
# and I/O constructors (whose time is acceptable as residual, out of scope).
# Real O(n)+ data-movement / compute must route through _call_numpy.
_ALLOWLIST = {
    # input normalization / construction
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "array",
    "require",
    # shape / dtype queries and casts metadata
    "ndim",
    "shape",
    "size",
    "dtype",
    "result_type",
    "promote_types",
    "min_scalar_type",
    "mintypecode",
    "common_type",
    "can_cast",
    "broadcast_shapes",
    "broadcast",
    "prod",
    # predicates
    "iterable",
    "isscalar",
    "issubdtype",
    "isdtype",
    "isfortran",
    "isfinite",
    "isin",
    "all",
    "shares_memory",
    "may_share_memory",
    "count_nonzero",
    # O(1) views / axis reorders
    "reshape",
    "transpose",
    "matrix_transpose",
    "permute_dims",
    "swapaxes",
    "squeeze",
    "expand_dims",
    "moveaxis",
    "rollaxis",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    "ravel",
    "broadcast_arrays",
    "broadcast_to",
    "hsplit",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    # allocation / index helpers (lazy/cheap or negligible at scale)
    "zeros",
    "ones",
    "empty",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full",
    "eye",
    "identity",
    "tri",
    "tril_indices",
    "triu_indices",
    "tril_indices_from",
    "triu_indices_from",
    "diag_indices",
    "diag_indices_from",
    "mask_indices",
    "ix_",
    "unravel_index",
    "ravel_multi_index",
    # misc cheap
    "errstate",
    "copyto",
    "from_dlpack",
    "frombuffer",
    "binary_repr",
    "base_repr",
    "typename",
    # I/O constructors (time acceptable as residual; out of Class-B scope)
    "fromfile",
    "fromregex",
    "fromstring",
}


def _is_counted_wrapper(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(d, ast.Name) and d.id == "_counted_wrapper"
        for d in node.decorator_list
    )


def _bare_np_calls(fn: ast.FunctionDef):
    """Yield (attr, lineno) for every `_np.<attr>(...)` CALL inside fn."""
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "_np"
            and n.func.attr not in _ALLOWLIST
        ):
            yield n.func.attr, n.lineno


@pytest.mark.parametrize("module", _MODULES)
def test_no_bare_numpy_compute_in_counted_ops(module):
    path = _SRC / module
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_counted_wrapper(node):
            for attr, lineno in _bare_np_calls(node):
                offenders.append(f"{module}:{lineno} {node.name}() -> _np.{attr}(")
    assert not offenders, (
        "Bare _np.<compute>() call(s) in counted op(s). Route real numpy work "
        "through _call_numpy (inside deduct/deduct_after) or _call_user_code; "
        "if the call is genuinely O(1) dispatch, add it to _ALLOWLIST.\n"
        + "\n".join(offenders)
    )
