"""Client API-contract guard.

Runs in the participant (client) venv — pyzmq present, numpy absent, no server
needed (only attribute/import checks). Protects the submission + evaluator-runner
contract: `import flopscope.numpy as fnp` must work and expose the names the
evaluator's `_child_entry.py` and starter-kit submissions rely on. A divergence
here breaks every submission before participant code runs.
"""

import importlib
import types


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
    # Self-consistency: flopscope.numpy must expose every name declared in
    # flopscope.__all__ (catches numpy.py drifting from the top level).
    # We use __all__ (the explicit public surface) rather than dir(), which
    # also exposes implementation-internal names like builtins/struct/BLACKLISTED
    # that must NOT leak into the participant namespace.
    import flopscope as flops

    fnp = importlib.import_module("flopscope.numpy")
    public_top = set(flops.__all__)
    missing = {n for n in public_top if not hasattr(fnp, n)}
    assert not missing, (
        f"flopscope.numpy missing names present at top level: {sorted(missing)}"
    )


def test_top_level_submission_api_present():
    # The origin/main client submission API. (load/save/savez/Module are PR #116
    # features, intentionally NOT part of this rehab.) `configure` is added in
    # this phase; this test guards that the top-level surface stays intact.
    import flopscope as flops

    for name in ("BudgetContext", "SymmetryGroup", "array", "einsum", "configure"):
        assert hasattr(flops, name), f"top-level flopscope missing: {name}"
