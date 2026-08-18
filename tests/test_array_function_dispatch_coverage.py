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
    "copyto", "putmask", "place", "select", "cov", "corrcoef",
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
    dispatch = FlopscopeArray._get_array_function_dispatch()   # forces lazy build
    passthrough = FlopscopeArray._PASSTHROUGH or set()
    return set(dispatch) | set(passthrough)


@pytest.mark.parametrize("name", DEMONSTRATED)
def test_demonstrated_ops_are_dispatch_covered(name):
    fn = _numpy_callable(name)
    assert fn is not None, f"numpy has no {name}"
    assert fn in _covered(), f"np.{name} falls through to numpy and bills 0"


def test_dispatch_coverage_does_not_silently_shrink():
    """A floor, so removing an entry is caught even before full coverage."""
    covered = _covered()
    hit = sum(
        1 for n in REGISTRY
        if (fn := _numpy_callable(n)) is not None and fn in covered
    )
    assert hit >= 100, f"dispatch coverage dropped to {hit}; entries were removed"


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
