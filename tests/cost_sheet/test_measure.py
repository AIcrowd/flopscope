import numpy as np

import flopscope.numpy as fnp
from scripts.cost_sheet.measure import capture_cost_site, measure_billed, measure_raw


def _a(dt):
    return fnp.asarray(np.ones(1000, dtype=dt))


def test_measure_raw_isolates_flop_cost():
    # unit weights, real dtype -> raw flop_cost (add of 1000 -> 1000)
    assert measure_raw(lambda: fnp.add(_a(np.float32), _a(np.float32))) == 1000


def test_measure_raw_resets_production_weights():
    # Discriminating isolation test: even with production weights already
    # loaded, measure_raw must reset to unit mode. A float64 add is 1000 raw
    # (unit), not 2000 (float64 dtype_rate 2.0) -- so this fails if the
    # reset_weights() in measure_raw is removed.
    from flopscope._weights import load_weights

    load_weights()
    assert measure_raw(lambda: fnp.add(_a(np.float64), _a(np.float64))) == 1000


def test_measure_billed_applies_production_rates():
    # complex128 add -> 1000 * rate 2.0 * complex_factor 2.0 = 4000
    assert measure_billed(lambda: fnp.add(_a(np.complex128), _a(np.complex128))) == 4000


def test_capture_cost_site_points_into_flopscope_source():
    site = capture_cost_site(lambda: fnp.add(_a(np.float32), _a(np.float32)))
    assert site is not None
    rel, line = site
    assert rel.startswith("src/flopscope/") and rel.endswith(".py") and line > 0


def test_measure_op_honors_raw_dtype_for_integer_only_ops():
    # bitwise_and rejects float32, so its RAW pass must run at an integer
    # dtype; the billed columns still honor their own dtypes (float32 raises).
    from scripts.cost_sheet.measure import measure_op

    def make(dt, s=1):
        a = fnp.asarray(np.ones(1000 * s, dtype=dt))
        b = fnp.asarray(np.ones(1000 * s, dtype=dt))
        return lambda: fnp.bitwise_and(a, b)

    m = measure_op(make, scalable=True, raw_dtype="int32")
    assert m["raw_flop_cost"] == 1000 and m["raw_flop_cost_2x"] == 2000
    assert m["billed"]["int16"] == 1000  # int16 rate 1.0
    assert m["billed"]["float32"] == "raises"  # numpy rejects float bitwise
