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
