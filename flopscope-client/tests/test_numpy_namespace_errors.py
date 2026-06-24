import pytest


def test_fnp_blacklisted_gives_clear_error():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError, match="not supported in the flopscope client"):
        _ = fnp.ndindex


def test_fnp_server_only_gives_clear_error():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError, match="server-side"):
        _ = fnp.SymmetricTensor


@pytest.mark.parametrize("name", ["vectorize", "frompyfunc"])
def test_fnp_pyfunc_wrapper_gives_actionable_error(name):
    """np.vectorize/frompyfunc wrap an arbitrary Python callable applied per
    element -- it can't be FLOP-counted or dispatched (it would run uncounted in
    the client). Participants must get an actionable error, not a bare
    'no attribute' (prod sub 310855 hit the cryptic AttributeError)."""
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError, match="not supported in the flopscope client"):
        _ = getattr(fnp, name)


def test_fnp_real_op_still_resolves():
    import flopscope.numpy as fnp

    assert callable(fnp.matmul)  # real proxy, unaffected


def test_fnp_private_name_raises_plain_attributeerror():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError):
        _ = fnp._nonexistent_private  # underscore names must not be re-routed
