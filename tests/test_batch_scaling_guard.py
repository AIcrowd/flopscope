"""Hard CI guard: every charged op scales correctly with a batch/broadcast
dimension, or is explicitly classified as not applicable.

Governance
----------
Tasks 1-8 of the batch/broadcast/dropped-dimension audit fixed 18 ops whose
billed cost scaled with fewer dimensions than numpy's real work (e.g.
``linalg.solve`` reading its batch size off ``a`` alone while numpy also
broadcasts ``b``'s leading dims). ``tests/batch_scan.py`` mechanically
re-derives this invariant for every charged op by comparing a base call
against a batch-prepended / broadcast / parameter-repeated variant and
checking the bill scales with numpy's real work.

For a NEW op (or a cost-formula change to an existing one): run the scanner
(``scan_all()`` in ``tests/batch_scan.py``, or ``python -m tests.batch_scan``
for a quick histogram). Either it comes back ``OK-SCALES`` -- nothing to do,
``test_no_charged_op_underbills_batch_or_broadcast`` below already covers it
-- or it needs a classified entry in ``tests/data/batch_scan_classification.json``
with a concrete reason (``"OK-SCALES"`` if the scanner's generic probes can't
reach it but you've verified it scales some other way, or ``"NO-BATCH"`` if
the op genuinely has no batch/broadcast dimension -- a scalar-shape creation
op, a set op that flattens its input, an in-place scatter, etc.). An op that
comes back ``UNDER-BILL`` is a real bug: fix the cost formula, don't classify
around it.
"""

import json
from pathlib import Path

import numpy as np

from tests.batch_scan import charged_ops, scan_all

CLASSIFICATION = json.loads(
    (Path(__file__).parent / "data" / "batch_scan_classification.json").read_text()
)


def _version_limited_ops() -> dict[str, str]:
    """Frozen-OK ops that cannot scale-test on the RUNNING numpy because the
    backing function (or the batched form the scanner probes) does not exist
    here. Detected empirically -- never from a version table -- so this
    shrinks to {} on any numpy that has the capability, restoring full
    enforcement there. The wrappers' own old-numpy contract
    (UnsupportedFunctionError) is pinned by test_numpy_version_support.py."""
    limited: dict[str, str] = {}
    for op in ("matvec", "vecmat", "cumulative_sum", "cumulative_prod"):
        if not hasattr(np, op):
            limited[op] = f"np.{op} does not exist on numpy {np.__version__}"
    try:
        np.trim_zeros(np.zeros((2, 2)))
    except ValueError:
        limited["trim_zeros"] = (
            f"trim_zeros is 1-D-only on numpy {np.__version__}; no batch form"
        )
    return limited


def test_no_charged_op_underbills_batch_or_broadcast():
    results = scan_all()
    underbills = {op: r for op, r in results.items() if r["verdict"] == "UNDER-BILL"}
    assert not underbills, f"batch/broadcast under-bills: {sorted(underbills)}"


def test_every_charged_op_is_classified():
    missing = [op for op in charged_ops() if op not in CLASSIFICATION]
    assert not missing, (
        "unclassified charged ops (scale them or add a NO-BATCH reason to "
        f"tests/data/batch_scan_classification.json): {missing}"
    )


def test_no_ok_scales_op_silently_loses_scale_coverage():
    """An op frozen as OK-SCALES must still scale-test as OK-SCALES on every run.
    Catches a broken/removed recipe that would silently turn the op UNTESTED (or
    a downgrade to NO-BATCH-NA) -- either of which removes the op's under-bill
    protection while the other two guard tests stay green."""
    results = scan_all()
    frozen_ok = {op for op, e in CLASSIFICATION.items() if e["class"] == "OK-SCALES"}
    frozen_ok -= set(_version_limited_ops())
    regressed = {
        op: results.get(op, {}).get("verdict", "MISSING")
        for op in frozen_ok
        if results.get(op, {}).get("verdict") != "OK-SCALES"
    }
    assert not regressed, (
        "ops frozen as OK-SCALES no longer scale-test as OK-SCALES (recipe broke "
        "or op downgraded -- under-bill protection silently lost): "
        f"{dict(sorted(regressed.items()))}"
    )
