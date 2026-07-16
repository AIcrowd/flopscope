import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._registry import REGISTRY
from scripts.cost_sheet.canonical_inputs import resolve
from scripts.cost_sheet.measure import capture_cost_site

# Known representatives, one per charged category: a stable, readable smoke
# sample rather than whatever op happens to appear first in the registry for
# a category. Full charged-op coverage (curation is complete) is asserted
# separately by test_every_charged_op_resolves_post_curation.
REPRESENTATIVES = {
    "counted_unary": "abs",
    "counted_binary": "add",
    "counted_reduction": "sum",
    "counted_custom": "matmul",  # seeded in CANONICAL_INPUTS
    "counted_random_method": "random.Generator.standard_normal",
}


def test_known_representative_per_category_resolves_and_runs():
    for cat, op in REPRESENTATIVES.items():
        entry = REGISTRY[op]
        assert entry.get("category") == cat, f"{op} moved out of {cat}"
        spec = resolve(op, entry)
        assert spec is not None, f"{cat}:{op} did not resolve"
        call = spec.make(np.float32, 1)  # inputs built here, outside the context
        with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
            call()
        assert b.flops_used > 0, f"{cat}:{op} ran but charged nothing"


def test_cost_site_attributes_to_the_op_not_input_construction():
    # Inputs must be built at make() time: capture_cost_site records the FIRST
    # deduct, so building arrays inside the returned callable would attribute
    # every op's cost site to asarray.
    asarray_site = capture_cost_site(lambda: fnp.asarray(np.ones(4, dtype=np.float32)))
    assert asarray_site is not None
    for cat, op in REPRESENTATIVES.items():
        spec = resolve(op, REGISTRY[op])
        assert spec is not None
        site = capture_cost_site(spec.make(np.float32, 1))
        assert site is not None, f"{cat}:{op} fired no deduct"
        assert site != asarray_site, f"{cat}:{op} cost site landed on asarray"


def test_unknown_op_resolves_to_none_not_raise():
    # The resolver returns None (never raises) for an op it has no coverage
    # for. Post-curation every real charged op IS covered, so probe with a
    # synthetic registry entry instead of hunting for an uncovered real one.
    assert resolve("not_a_real_op_xyz", {"category": "counted_custom"}) is None


def test_every_charged_op_resolves_post_curation():
    # Curation closed the long tail: every charged registry op resolves.
    charged = {
        "counted_unary",
        "counted_binary",
        "counted_reduction",
        "counted_custom",
        "counted_random_method",
    }
    unresolved = [
        k
        for k, e in REGISTRY.items()
        if e.get("category") in charged and resolve(k, e) is None
    ]
    assert unresolved == [], f"uncovered charged ops: {unresolved}"
