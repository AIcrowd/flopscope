import pytest


def test_top_level_server_only_clear_error():
    import flopscope as flops

    with pytest.raises(AttributeError, match="server-side"):
        _ = flops.SymmetricTensor


def test_top_level_budget_reset_is_available():
    import flopscope as flops

    assert callable(flops.budget_reset)


def test_flops_submodule_has_getattr():
    import flopscope.flops as flops_mod

    # flops submodule now delegates unknown names through make_module_getattr,
    # so flops.* server-only names (populated in C4) raise the clear error
    # instead of a bare AttributeError.
    assert callable(getattr(flops_mod, "__getattr__", None))


def test_flops_cost_helper_gives_clear_error():
    import flopscope.flops as flops_mod

    with pytest.raises(AttributeError, match="not available in the flopscope client"):
        _ = flops_mod.det_cost
