"""Client API-contract guard.

Runs in the participant (client) venv — pyzmq present, numpy absent, no server
needed (only attribute/import checks). Protects the submission + evaluator-runner
contract: `import flopscope.numpy as fnp` must work and expose the names the
evaluator's `_child_entry.py` and starter-kit submissions rely on. A divergence
here breaks every submission before participant code runs.
"""

import importlib


def test_flopscope_numpy_importable():
    fnp = importlib.import_module("flopscope.numpy")
    assert fnp is not None


def test_runner_contract_names_present():
    # The exact names whestbench-evaluator/worker/_child_entry.py depends on.
    fnp = importlib.import_module("flopscope.numpy")
    for name in ("frombuffer", "zeros", "asarray", "float32", "ndarray"):
        assert hasattr(fnp, name), (
            f"flopscope.numpy missing runner-critical name: {name}"
        )


def test_numpy_mirrors_top_level_proxyable_surface():
    # Guarantee: flopscope.numpy must expose every proxyable op that is
    # registered at the top level AND every special-cased top-level function.
    #
    # Why not dir(flops)?  dir() includes implementation-internal names
    # (builtins, struct, get_connection, …) that must NOT leak into the
    # participant fnp namespace.
    #
    # Why not flops.__all__?  numpy.py does ``from flopscope import *``, which
    # imports exactly the names in flops.__all__ into fnp's namespace.  Checking
    # ``hasattr(fnp, n) for n in flops.__all__`` is therefore tautological and
    # cannot catch drift between the top-level surface and flopscope.numpy.
    #
    # Independent source of truth: iter_proxyable() comes straight from the
    # registry (FUNCTION_CATEGORIES), which is completely independent of
    # __all__ and the star-import.  A proxyable op that exists in the registry
    # and is wired up at top level (non-dotted name) but is somehow absent from
    # fnp (e.g. removed from __all__ by mistake, or numpy.py's star-import
    # replaced by a selective import that omits it) will be caught here.
    import flopscope as flops
    from flopscope._registry import iter_proxyable

    fnp = importlib.import_module("flopscope.numpy")

    # Special-cased functions defined directly in __init__.py (not generated
    # from the proxy loop, but still part of the public top-level surface).
    SPECIAL_CASED = {"array", "einsum", "load", "save", "savez", "savez_compressed"}

    # Registry-driven expected set: all proxyable ops with no dot in the name
    # (dotted names like "linalg.solve" belong to submodules, not the top level).
    registry_top = {name for name in iter_proxyable() if "." not in name}

    expected = registry_top | SPECIAL_CASED

    missing_from_flops = {n for n in expected if not hasattr(flops, n)}
    missing_from_fnp = {n for n in expected if not hasattr(fnp, n)}

    assert not missing_from_flops, (
        f"top-level flopscope missing proxyable ops: {sorted(missing_from_flops)}"
    )
    assert not missing_from_fnp, (
        f"flopscope.numpy missing names present at top level: {sorted(missing_from_fnp)}"
    )


def test_top_level_submission_api_present():
    # The origin/main client submission API. (load/save/savez/Module are PR #116
    # features, intentionally NOT part of this rehab.) `configure` is added in
    # this phase; this test guards that the top-level surface stays intact.
    import flopscope as flops

    for name in ("BudgetContext", "SymmetryGroup", "array", "einsum", "configure"):
        assert hasattr(flops, name), f"top-level flopscope missing: {name}"
