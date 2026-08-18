"""Out-of-process probes that RAW numpy refuses a 1-tuple ``out=`` for its
random ``Generator`` methods.

These are the ``random.Generator.*`` siblings of ``choose`` in
``_NUMPY_REFUSES_THE_CONTAINER``: they take ``out=`` but NOT through the ufunc
protocol, so numpy wants the array itself and refuses ``out=(array,)``. The
premise is the same one asserted in-process for ``choose`` --- the only reason
these live in their own module is the ACCEPTED spelling.

The accepted spelling here is a raw numpy random *fill* --- e.g.
``np.random.default_rng(0).standard_normal(out=np.zeros(8))`` --- and that fill
segfaults numpy intermittently under pytest-xdist load (a numpy-internal crash,
not a flopscope or refusal-logic defect: the bare call runs clean in isolation
and only faults under the full concurrent suite). Run inside the test process
that segfault kills the xdist worker and drops coverage below the fail-under,
reddening whole matrix cells at once. Run here, in a fresh interpreter spawned
per op, the crash is just a signal-kill the parent test reads as a negative
return code --- so one flaky numpy segfault fails at most a single case instead
of cascading.

This module is import-light on purpose (numpy only, no flopscope): the parent
test invokes it as ``python tests/_numpy_container_out_probes.py <op-name>``.
It is also imported by the test module purely for :data:`PROBES` (the op names
under test) --- the leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

#: ``op name -> call(out)``. A bare ``out=array`` is the accepted spelling;
#: ``out=(array,)`` must be refused by numpy itself. Single source of truth for
#: these calls --- the test module reads the keys, the ``__main__`` below runs
#: the values.
PROBES: dict[str, Callable[[Any], Any]] = {
    "random.Generator.random": lambda out: np.random.default_rng(0).random(out=out),
    "random.Generator.standard_normal": lambda out: np.random.default_rng(
        0
    ).standard_normal(out=out),
    "random.Generator.standard_exponential": lambda out: np.random.default_rng(
        0
    ).standard_exponential(out=out),
    "random.Generator.standard_gamma": lambda out: np.random.default_rng(
        0
    ).standard_gamma(1.0, out=out),
    "random.Generator.permuted": lambda out: np.random.default_rng(0).permuted(
        np.arange(8.0), out=out
    ),
}


def check(name: str) -> None:
    """Assert RAW numpy accepts a bare ``out=`` here and refuses a 1-tuple.

    The bare call is the segfault-prone one; it runs first so a crash surfaces
    as a signal-kill of this process rather than a swallowed exception. If the
    tuple form is NOT refused, numpy has adopted the ufunc protocol here and the
    ``_NOT_DRIVEN`` excuse is stale --- that is a real failure, raised loudly.
    """
    call = PROBES[name]
    call(np.zeros(8))  # the bare array is the accepted spelling
    try:
        call((np.zeros(8),))
    except (TypeError, ValueError):
        return
    raise AssertionError(f"numpy no longer refuses a 1-tuple out= for {name}")


if __name__ == "__main__":
    check(sys.argv[1])
