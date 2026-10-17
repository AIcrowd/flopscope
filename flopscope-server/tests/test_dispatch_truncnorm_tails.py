"""The existing stats dispatch must retain stable tail results and metadata."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session
from scipy import stats


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_tail_stats_dispatch(method):
    values = (
        np.array([0.1, 0.5, 0.9]) if method == "ppf" else np.array([40.0, 40.01, 40.1])
    )
    session = Session(flop_budget=10**9)
    try:
        handler = RequestHandler(session)
        handle = session.store_array(values)
        before = session.budget_remaining
        response = handler.handle(
            {
                "op": "stats.truncnorm." + method,
                "args": [handle, 40.0, 41.0],
                "kwargs": {},
            }
        )
        assert response["status"] == "ok", response
        metadata = response["result"]
        result = np.asarray(session.get_array(metadata["id"]))
        assert result.shape == (3,) and result.dtype == np.float64
        assert session.budget_remaining < before
        np.testing.assert_allclose(
            result,
            getattr(stats.truncnorm, method)(values, 40, 41),
            rtol=5e-13,
            atol=5e-13,
        )
    finally:
        session.close()
