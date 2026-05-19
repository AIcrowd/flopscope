"""flopscope's slim adapter over opt_einsum.

Re-exports contract_path from upstream and the flopscope PathInfo/StepInfo
adapters. Path search algorithms come from upstream; per-step FLOP costs are
recomputed using flopscope's FMA convention (default 1, configurable via the
``fma_cost`` setting).

See LICENSE and NOTICE in this directory for attribution to opt_einsum.
"""

import opt_einsum.path_random as _path_random_upstream
from opt_einsum import contract_path as _upstream_contract_path
from opt_einsum.parser import get_shape as _get_shape
from opt_einsum.parser import parse_einsum_input
from opt_einsum.paths import (
    _AUTO_CHOICES,
    _AUTO_HQ_CHOICES,
    BranchBound,
    DynamicProgramming,
    PathOptimizer,
    get_path_fn,
    greedy,
    register_path_fn,
)
from opt_einsum.paths import (
    _PATH_OPTIONS as _upstream_path_options,
)

from ._contract import PathInfo, StepInfo, build_path_info
from ._helpers import flop_count


def _resolve_optimizer_name(optimize, num_ops: int) -> str:
    """Resolve the effective optimizer name for display in PathInfo.

    Mirrors the resolution logic from the old local contract_path:
    - 'auto' and 'auto-hq' resolve to the inner function name based on num_ops
    - Explicit string names are returned as-is
    - num_ops <= 2 returns 'trivial'
    - PathOptimizer instances return their class name
    """
    if num_ops <= 2:
        return "trivial"
    if isinstance(optimize, PathOptimizer):
        return type(optimize).__name__
    if not isinstance(optimize, str):
        return ""
    if optimize in (True, "auto", None):
        inner_fn = _AUTO_CHOICES.get(num_ops, greedy)
        return getattr(inner_fn, "__name__", None) or getattr(
            getattr(inner_fn, "func", None), "__name__", str(inner_fn)
        )
    if optimize == "auto-hq":
        inner_fn = _AUTO_HQ_CHOICES.get(
            num_ops, _path_random_upstream.random_greedy_128
        )
        return getattr(inner_fn, "__name__", None) or getattr(
            getattr(inner_fn, "func", None), "__name__", str(inner_fn)
        )
    return optimize


def _resolve_local_path(optimize, args, kwargs):
    """Resolve a path locally for optimizers not known to upstream opt_einsum.

    Used when ``optimize`` is a local PathOptimizer instance or a string key
    registered only in flopscope's local registry (e.g. a custom key
    added via ``register_path_fn``).

    Returns ``(resolved_path_list, optimizer_name_str)``.
    """
    from . import _helpers as helpers

    operands_ = [args[0]] + list(args[1:])
    shapes = kwargs.get("shapes", False)
    input_subscripts, output_subscript, operands_prepped = parse_einsum_input(
        operands_, shapes=shapes
    )
    input_list = input_subscripts.split(",")
    input_sets = [frozenset(x) for x in input_list]
    if shapes:
        input_shapes = list(operands_prepped)
    else:
        input_shapes = [_get_shape(x) for x in operands_prepped]
    output_set = frozenset(output_subscript)
    size_dict: dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])
            if char not in size_dict:
                size_dict[char] = dim
            elif size_dict[char] == 1:
                size_dict[char] = dim

    memory_limit = kwargs.get("memory_limit", None)
    size_list_mem = [
        helpers.compute_size_by_dict(t, size_dict)
        for t in input_list + [output_subscript]
    ]
    memory_arg = None
    if memory_limit == "max_input":
        memory_arg = max(size_list_mem)
    elif isinstance(memory_limit, int) and memory_limit > 0:
        memory_arg = memory_limit

    num_ops = len(input_list)
    if isinstance(optimize, PathOptimizer):
        resolved_path = optimize(input_sets, output_set, size_dict, memory_arg)
        optimizer_name = _resolve_optimizer_name(optimize, num_ops)
    else:
        # String key in registry
        path_fn = get_path_fn(optimize)
        resolved_path = path_fn(input_sets, output_set, size_dict, memory_arg)
        optimizer_name = _resolve_optimizer_name(optimize, num_ops)

    return list(resolved_path), optimizer_name


def _count_num_ops(args, kwargs):
    """Count the number of operands from args/kwargs without full parse."""
    operands_ = [args[0]] + list(args[1:])
    shapes = kwargs.get("shapes", False)
    input_subscripts, _, _ = parse_einsum_input(operands_, shapes=shapes)
    return len(input_subscripts.split(","))


def contract_path(*args, **kwargs):
    """Run upstream opt_einsum.contract_path, then adapt the result to
    flopscope's PathInfo (with FMA-aware per-step costs).

    All arguments are forwarded to upstream. The return value's PathInfo is
    flopscope's dataclass form; the path itself is unchanged.

    If ``optimize`` is a PathOptimizer instance, or a string key registered
    only in flopscope's local registry, the path is resolved locally
    first and then forwarded to upstream as an explicit path list so that
    upstream produces the contraction_list we need for build_path_info.
    """
    optimize = kwargs.get("optimize", True)

    needs_local_resolve = False
    if isinstance(optimize, PathOptimizer):
        needs_local_resolve = True
    elif (
        isinstance(optimize, str)
        and optimize in _upstream_path_options
        and optimize not in _upstream_path_options
    ):
        # This branch is structurally unreachable but kept for symmetry.
        needs_local_resolve = True

    if needs_local_resolve:
        resolved_path, optimizer_name = _resolve_local_path(optimize, args, kwargs)
        # Re-issue upstream call with the resolved path so contraction_list is built.
        new_kwargs = {k: v for k, v in kwargs.items() if k != "optimize"}
        new_kwargs["optimize"] = resolved_path
        upstream_path, upstream_info = _upstream_contract_path(*args, **new_kwargs)
    else:
        upstream_path, upstream_info = _upstream_contract_path(*args, **kwargs)
        num_ops = (
            len(upstream_info.input_subscripts.split(","))
            if upstream_info.input_subscripts
            else 2
        )
        if optimize is True or optimize is None:
            optimize = "auto"
        optimizer_name = _resolve_optimizer_name(optimize, num_ops)

    return list(upstream_path), build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
        optimizer_used=optimizer_name,
    )


__all__ = [
    "PathInfo",
    "StepInfo",
    "contract_path",
    "flop_count",
    "build_path_info",
    "parse_einsum_input",
    "BranchBound",
    "DynamicProgramming",
    "register_path_fn",
]


def __getattr__(name: str) -> object:
    """Lazy package-level attribute hook (PEP 562).

    Exposes upstream opt_einsum sub-modules under the private names used by the
    vendored test suite (``oe._helpers``, ``oe._paths``, ``oe._path_random``)
    WITHOUT registering them in the package ``__dict__``, so they never shadow
    the real ``flopscope._opt_einsum._helpers`` submodule that lives on disk.
    """
    if name == "_helpers":
        import opt_einsum.helpers as _m
        return _m
    if name == "_paths":
        import opt_einsum.paths as _m  # type: ignore[no-redef]
        return _m
    if name == "_path_random":
        return _path_random_upstream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
