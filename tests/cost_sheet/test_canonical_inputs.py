import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._registry import REGISTRY
from scripts.cost_sheet.canonical_inputs import CANONICAL_INPUTS, resolve
from scripts.cost_sheet.measure import capture_cost_site

# Known representatives, one per charged category. At this stage only the
# explicit seeds and the category defaults resolve (the long tail is filled by
# the later curation task), so sample well-known ops rather than whatever op
# happens to appear first in the registry for a category.
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


def test_uncovered_charged_op_resolves_to_none_not_raise():
    # counted_custom has no category default, so any unseeded custom op is
    # uncovered until the curation task fills the long tail.
    op, entry = next(
        (k, e)
        for k, e in REGISTRY.items()
        if e.get("category") == "counted_custom" and k not in CANONICAL_INPUTS
    )
    assert resolve(op, entry) is None, f"{op} should be uncovered at this stage"
