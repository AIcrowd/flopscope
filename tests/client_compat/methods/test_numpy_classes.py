"""Wrap numpy's ndarray method/operator test classes so they run inside this
conftest's scope (budget context + construction patch active).

Importing numpy's test module at module level happens *before* pytest_configure
fires the construction patch, so numpy.ma and other stdlib-side imports resolve
against native numpy.  The subclasses below are thin shells: they inherit every
test_ method and pytest collects them as items *inside* tests/client_compat/methods/,
which means the _server and _fresh_connection_and_budget autouse fixtures apply and
np.array/zeros/... are routed to RemoteArray during the test body.
"""

from __future__ import annotations

# Import BEFORE the construction patch fires (module import order: this file is
# imported by pytest's collector before pytest_configure / pytest_sessionstart).
from numpy._core.tests.test_multiarray import (
    TestArgmax as _TestArgmax,
)
from numpy._core.tests.test_multiarray import (
    TestArgmin as _TestArgmin,
)
from numpy._core.tests.test_multiarray import (
    TestClip as _TestClip,
)
from numpy._core.tests.test_multiarray import (
    TestConversion as _TestConversion,
)
from numpy._core.tests.test_multiarray import (
    TestMethods as _TestMethods,
)
from numpy._core.tests.test_multiarray import (
    TestStats as _TestStats,
)
from numpy._core.tests.test_multiarray import (
    TestTake as _TestTake,
)


class TestMethodsClient(_TestMethods):
    """numpy.ndarray TestMethods exercised against RemoteArray."""


class TestArgmaxClient(_TestArgmax):
    """numpy.ndarray TestArgmax exercised against RemoteArray."""


class TestArgminClient(_TestArgmin):
    """numpy.ndarray TestArgmin exercised against RemoteArray."""


class TestClipClient(_TestClip):
    """numpy.ndarray TestClip exercised against RemoteArray."""


class TestTakeClient(_TestTake):
    """numpy.ndarray TestTake exercised against RemoteArray."""


class TestStatsClient(_TestStats):
    """numpy.ndarray TestStats exercised against RemoteArray."""


class TestConversionClient(_TestConversion):
    """numpy.ndarray TestConversion exercised against RemoteArray."""
