"""Every counted op with a live numpy callable must be reachable by dispatch.

``FlopscopeArray`` subclasses ``numpy.ndarray``, so NumPy's own
``__array_function__`` protocol is a live fallback for any ``np.<func>(...)``
call whose callable isn't in flopscope's explicit dispatch map. That fallback
only fires when a *plain* ``numpy.ndarray`` rides along among the relevant
args -- an all-``FlopscopeArray`` call fails **closed** (``TypeError``: "no
implementation found"), which is exactly why the existing test suite (which
only ever exercises flopscope arrays) never caught this: a mixed call like
``np.cov(flopscope_array, plain_ndarray)`` silently falls through to NumPy's
default implementation and bills 0 FLOPs for real compute.

The map is a LAZILY-BUILT class attribute keyed by the numpy CALLABLE (not by
name), so the test must force the build via the classmethod first.
"""

import numpy as np
import pytest

from flopscope._ndarray import FlopscopeArray
from flopscope._registry import REGISTRY

DEMONSTRATED = [
    "copyto",
    "putmask",
    "place",
    "select",
    "cov",
    "corrcoef",
    "linalg.tensorsolve",
]


def _numpy_callable(dotted):
    obj = np
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def _covered():
    """Set of numpy callables flopscope will route rather than fail open on."""
    dispatch = FlopscopeArray._get_array_function_dispatch()  # forces lazy build
    passthrough = FlopscopeArray._PASSTHROUGH or set()
    return set(dispatch) | set(passthrough)


@pytest.mark.parametrize("name", DEMONSTRATED)
def test_demonstrated_ops_are_dispatch_covered(name):
    fn = _numpy_callable(name)
    assert fn is not None, f"numpy has no {name}"
    assert fn in _covered(), f"np.{name} falls through to numpy and bills 0"


# Ops whose dispatch coverage we pin by NAME. A count would be a mystery
# number on failure and is numpy-version-dependent besides: this set was
# measured 2026-08-18 in this test's own execution context (forcing the
# lazy dispatch-map build first, since import order can change what's
# registered), against numpy 2.2.6 -- 121 names. A set difference names
# the op that appeared or vanished, which is what a maintainer actually
# needs; a bare count does not.
_EXPECTED_COVERED = frozenset(
    {
        "all",
        "allclose",
        "amax",
        "amin",
        "any",
        "apply_over_axes",
        "argmax",
        "argmin",
        "argpartition",
        "argsort",
        "array_equiv",
        "astype",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "average",
        "bincount",
        "broadcast_to",
        "clip",
        "column_stack",
        "concat",
        "concatenate",
        "copy",
        "copyto",
        "corrcoef",
        "cov",
        "cross",
        "cumprod",
        "cumsum",
        "diagonal",
        "diff",
        "digitize",
        "dot",
        "dsplit",
        "einsum",
        "expand_dims",
        "flip",
        "gradient",
        "histogram",
        "histogram2d",
        "histogram_bin_edges",
        "histogramdd",
        "hsplit",
        "hstack",
        "in1d",
        "inner",
        "intersect1d",
        "isclose",
        "isin",
        "lexsort",
        "linalg.cholesky",
        "linalg.det",
        "linalg.eig",
        "linalg.eigh",
        "linalg.eigvals",
        "linalg.eigvalsh",
        "linalg.inv",
        "linalg.lstsq",
        "linalg.matrix_power",
        "linalg.matrix_rank",
        "linalg.multi_dot",
        "linalg.norm",
        "linalg.pinv",
        "linalg.qr",
        "linalg.slogdet",
        "linalg.solve",
        "linalg.svd",
        "linalg.tensorsolve",
        "matmul",
        "matrix_transpose",
        "max",
        "mean",
        "median",
        "meshgrid",
        "min",
        "moveaxis",
        "nonzero",
        "outer",
        "pad",
        "partition",
        "percentile",
        "permute_dims",
        "piecewise",
        "place",
        "prod",
        "ptp",
        "putmask",
        "quantile",
        "ravel",
        "repeat",
        "reshape",
        "roll",
        "round",
        "searchsorted",
        "select",
        "setdiff1d",
        "setxor1d",
        "sort",
        "split",
        "squeeze",
        "stack",
        "std",
        "sum",
        "swapaxes",
        "tensordot",
        "tile",
        "trace",
        "transpose",
        "tril",
        "triu",
        "union1d",
        "unique",
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        "vander",
        "var",
        "vsplit",
        "vstack",
        "where",
    }
)

# Names whose presence legitimately varies across the supported numpy range,
# listed explicitly (never as a blanket escape hatch) so a real regression
# cannot hide here.
#
# "in1d": bound via _bind("in1d", "in1d") in _get_array_function_dispatch,
# and np.in1d still exists (deprecated) on numpy 2.2.6, so it resolves and
# counts here. flopscope's own registry documents its removal
# (_registry.py's "in1d" entry: ``"max_numpy": "2.4"``, "Removed in numpy
# 2.4; use `isin` instead."). Once np.in1d no longer exists as an attribute,
# `_numpy_callable("in1d")` returns None and "in1d" silently drops out of
# `covered` -- not a lost binding, an upstream removal already on record.
# Confirmed this is the exact (and only) name that moves between 2.2.6 and
# 2.4: simulating its absence (popping np.in1d from the dispatch map and
# deleting the attribute) drops the measured count from 121 to 120, matching
# CI's `test (3.12, 2.4)` failure exactly. ``trapz`` is also
# ``max_numpy: "2.4"`` in the registry but is never bound in the dispatch
# map on any numpy version, so its removal does not move this set.
_VERSION_DEPENDENT = frozenset({"in1d"})


def test_dispatch_coverage_does_not_silently_shrink():
    """Pin the exact SET of covered names, not a count.

    A count is a mystery number on failure; a set difference names the op
    that appeared or vanished, which is exactly what a maintainer needs to
    act on. If you ADD dispatch coverage, this test should fail on the
    ``added`` assertion -- update ``_EXPECTED_COVERED`` and say why in the
    commit message. If ``missing`` is non-empty, a binding was lost (or
    dropped off this numpy version -- check ``_VERSION_DEPENDENT`` first).
    """
    covered = {
        n
        for n in REGISTRY
        if (fn := _numpy_callable(n)) is not None and fn in _covered()
    }
    missing = _EXPECTED_COVERED - covered - _VERSION_DEPENDENT
    added = covered - _EXPECTED_COVERED - _VERSION_DEPENDENT
    assert not missing, f"dispatch bindings LOST: {sorted(missing)}"
    assert not added, f"dispatch bindings ADDED (update the pin): {sorted(added)}"


def test_mixed_operands_do_not_fail_open():
    """The hole only fires when a PLAIN ndarray rides along -- all-flopscope
    args fail closed, which is why existing tests are blind to it."""
    import flopscope as flops
    import flopscope.numpy as fnp

    F = fnp.array(np.random.default_rng(0).random((200, 200)))
    raw = np.random.default_rng(1).random((200, 200))
    with flops.budget(10**14, quiet=True) as b:
        np.cov(F, raw)
        assert b.flops_used > 0, "np.cov(flopscope, raw) billed nothing"
