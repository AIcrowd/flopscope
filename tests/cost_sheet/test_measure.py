import numpy as np

import flopscope.numpy as fnp
from scripts.cost_sheet.measure import capture_cost_site, measure_billed, measure_raw


def _a(dt):
    return fnp.asarray(np.ones(1000, dtype=dt))


def test_measure_raw_isolates_flop_cost():
    # unit weights, real dtype -> raw flop_cost (add of 1000 -> 1000)
    assert measure_raw(lambda: fnp.add(_a(np.float32), _a(np.float32))) == 1000


def test_measure_billed_applies_production_rates():
    # complex128 add -> 1000 * rate 2.0 * complex_factor 2.0 = 4000
    assert measure_billed(lambda: fnp.add(_a(np.complex128), _a(np.complex128))) == 4000


def test_capture_cost_site_points_into_flopscope_source():
    site = capture_cost_site(lambda: fnp.add(_a(np.float32), _a(np.float32)))
    assert site is not None
    rel, line = site
    assert rel.startswith("src/flopscope/") and rel.endswith(".py") and line > 0
