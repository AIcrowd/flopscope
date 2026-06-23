"""Guards the client-parity harness actually stood up (server + client)."""
from __future__ import annotations


def test_client_is_the_proxy_not_native():
    import flopscope
    # The client package lives under flopscope-client/src; native under src/.
    assert "flopscope-client" in flopscope.__file__, (
        f"expected the CLIENT flopscope on sys.path, got {flopscope.__file__}"
    )


def test_server_round_trips_a_simple_op():
    import flopscope as fnp
    with fnp.BudgetContext(flop_budget=10**15):
        a = fnp.array([1, 2, 3])
        b = fnp.array([4, 5, 6])
        assert fnp.add(a, b).tolist() == [5, 7, 9]
