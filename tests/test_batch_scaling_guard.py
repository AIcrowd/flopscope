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

from tests.batch_scan import charged_ops, scan_all

CLASSIFICATION = json.loads(
    (Path(__file__).parent / "data" / "batch_scan_classification.json").read_text()
)


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
