"""Guards the client-parity harness actually stood up (server + client)."""

from __future__ import annotations


def test_client_is_the_proxy_not_native():
    import flopscope

    # The client package lives under flopscope-client/src; native under src/.
    assert "flopscope-client" in flopscope.__file__, (
        f"expected the CLIENT flopscope on sys.path, got {flopscope.__file__}"
    )


def test_server_round_trips_a_simple_op():
    # Relies on the ambient BudgetContext opened by the autouse
    # _fresh_connection_and_budget fixture (the client raises
    # NoBudgetContextError without an active budget). Opening our own here would
    # nest-conflict, and NumPy's suite never opens one — so this mirrors how the
    # real suite runs.
    import flopscope as fnp

    a = fnp.array([1, 2, 3])
    b = fnp.array([4, 5, 6])
    assert fnp.add(a, b).tolist() == [5, 7, 9]


def test_numpy_sum_is_routed_to_client(_patch_active):
    import numpy as np

    # np.sum is a non-ufunc reduction in the client registry, so the patch
    # targets it (ufuncs like np.add are intentionally skipped). np.array stays
    # native (it is in _SKIP), so the input arrives as a real ndarray and the
    # patch's input coercer must convert it (client rejects raw ndarrays).
    # Relies on the ambient BudgetContext from the autouse fixture.
    out = np.sum(np.array([1, 2, 3, 4]))
    assert type(out).__name__ == "RemoteArray", f"got {type(out)!r}; swap inactive"
    assert float(out) == 10.0


def test_numpy_testing_helpers_coerce_remote_array(_patch_active):
    import numpy as np

    # np.cumsum is a non-ufunc reduction -> patched -> returns a RemoteArray.
    # numpy's own assert helper must accept that remote handle (it routes through
    # asanyarray, which _coerce wraps to materialize via .tolist()).
    # Relies on the ambient BudgetContext from the autouse fixture.
    out = np.cumsum(np.array([1.0, 2.0, 3.0]))
    assert type(out).__name__ == "RemoteArray", f"got {type(out)!r}; swap inactive"
    np.testing.assert_allclose(out, [1.0, 3.0, 6.0])
