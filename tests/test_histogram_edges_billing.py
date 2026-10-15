"""histogram and histogram_bin_edges bill the bins + 1 edges they build (#277).

For an integer or string ``bins`` numpy builds the edges with
``linspace(first_edge, last_edge, nbins + 1)``, and ``density=True`` returns
``counts / diff(edges) / counts.sum()``. Both are per-bin work, so with few
samples and many bins they were almost free. The edges must cost what
``fnp.linspace`` costs for them, and the density what that expression costs
written out with ``fnp``. An array ``bins`` is the edges already.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights

K = 1 << 20  # the issue's bin count


def _bill(fn):
    """FLOPs billed for ``fn()`` alone (production weights), plus its result."""
    load_weights()
    with flops.BudgetContext(flop_budget=2**55, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def _records(fn):
    """``(op_name, flop_cost)`` for every op ``fn()`` records."""
    load_weights()
    with flops.BudgetContext(flop_budget=2**55, quiet=True) as bc:
        fn()
    return [(r.op_name, r.flop_cost) for r in bc.op_log]


def _samples(n=2):
    return fnp.asarray(np.random.default_rng(0).random(n))


def _edges(values) -> Any:
    """Explicit edges. The mirrored signature types ``bins`` as int, str or
    Sequence[int]; numpy and the runtime also take an array of edges."""
    return fnp.asarray(np.asarray(values))


def test_issue_bin_edges_bill_the_linspace_they_are():
    a = _samples()
    lin_bill, lin = _bill(lambda: fnp.linspace(0.0, 1.0, K + 1))
    # The per-sample part is what the same call bills with the edges given.
    given, _ = _bill(lambda: fnp.histogram_bin_edges(a, bins=_edges(lin)))
    billed, edges = _bill(lambda: fnp.histogram_bin_edges(a, bins=K, range=(0.0, 1.0)))
    assert billed == given + lin_bill
    np.testing.assert_array_equal(np.asarray(edges), np.asarray(lin))


def test_issue_histogram_bills_its_edges():
    a = _samples()
    lin_bill, lin = _bill(lambda: fnp.linspace(0.0, 1.0, K + 1))
    billed, (_, edges) = _bill(lambda: fnp.histogram(a, bins=K, range=(0.0, 1.0)))
    # n * ceil(log2(bins)) at weight 4 and the int64 counts rate 2: the
    # issue's 320, unchanged, plus the edges.
    assert billed == 2 * math.ceil(math.log2(K)) * 4 * 2 + lin_bill
    np.testing.assert_array_equal(np.asarray(edges), np.asarray(lin))


def test_issue_density_bills_its_per_bin_arithmetic():
    a = _samples()
    plain, (counts, edges) = _bill(lambda: fnp.histogram(a, bins=K, range=(0.0, 1.0)))
    billed, (dens, _) = _bill(
        lambda: fnp.histogram(a, bins=K, range=(0.0, 1.0), density=True)
    )
    spelled_bill, spelled = _bill(lambda: counts / fnp.diff(edges) / fnp.sum(counts))
    assert billed == plain + spelled_bill
    np.testing.assert_array_equal(np.asarray(dens), np.asarray(spelled))


def test_string_bins_bill_the_resolved_edges_as_linspace():
    a = _samples(200)
    edges = np.histogram_bin_edges(np.asarray(a), bins="auto")
    lin_bill, _ = _bill(lambda: fnp.linspace(edges[0], edges[-1], len(edges)))
    for call in (
        lambda: fnp.histogram_bin_edges(a, bins="auto"),
        lambda: fnp.histogram(a, bins="auto"),
    ):
        assert [cost for name, cost in _records(call) if name == "linspace"] == [
            lin_bill
        ]


def test_array_bins_are_the_edges_and_add_nothing():
    a = _samples(100)
    edges = _edges(np.linspace(0.0, 1.0, 9))
    for call in (
        lambda: fnp.histogram_bin_edges(a, bins=edges),
        lambda: fnp.histogram(a, bins=edges),
    ):
        assert len(_records(call)) == 1
    # Only the binning pass, as before: n * ceil(log2(len(bins))) at
    # weight 4 and rate 2.
    assert _bill(lambda: fnp.histogram(a, bins=edges))[0] == 100 * 4 * 4 * 2


def test_array_bins_density_bills_its_per_bin_arithmetic():
    a = _samples(100)
    edges = _edges(np.linspace(0.0, 1.0, 9))
    plain, (counts, _) = _bill(lambda: fnp.histogram(a, bins=edges))
    billed, _ = _bill(lambda: fnp.histogram(a, bins=edges, density=True))
    spelled_bill, _ = _bill(lambda: counts / fnp.diff(edges) / fnp.sum(counts))
    assert billed == plain + spelled_bill


def test_issue_control_per_sample_terms_are_unchanged():
    big = _samples(K)
    lin_bill, _ = _bill(lambda: fnp.linspace(0.0, 1.0, K + 1))
    hbe, _ = _bill(lambda: fnp.histogram_bin_edges(big, bins=K, range=(0.0, 1.0)))
    hist, _ = _bill(lambda: fnp.histogram(big, bins=K, range=(0.0, 1.0)))
    # The issue's per-sample bills, 8,388,608 and 167,772,160, plus the edges.
    assert hbe == 8_388_608 + lin_bill
    assert hist == 167_772_160 + lin_bill
