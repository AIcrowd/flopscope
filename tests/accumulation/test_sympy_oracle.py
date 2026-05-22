import pytest


def test_sympy_oracle_alpha_for_s2_matches_singleton():
    from flopscope._perm_group import _dimino
    from flopscope._perm_group import _Permutation as Permutation
    from tests.accumulation._sympy_oracle import sympy_brute_force_alpha

    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    alpha = sympy_brute_force_alpha(
        elements=elements, sizes=(4, 4), visible_positions=(0,)
    )
    assert alpha == 16


def test_sympy_oracle_alpha_trivial_group():
    from flopscope._perm_group import _Permutation as Permutation
    from tests.accumulation._sympy_oracle import sympy_brute_force_alpha

    identity = Permutation.identity(2)
    alpha = sympy_brute_force_alpha(
        elements=(identity,), sizes=(3, 4), visible_positions=(0,)
    )
    assert alpha == 12


def test_sympy_oracle_refuses_too_large_inputs():
    from flopscope._perm_group import _Permutation as Permutation
    from tests.accumulation._sympy_oracle import sympy_brute_force_alpha

    with pytest.raises(ValueError, match="too large"):
        sympy_brute_force_alpha(
            elements=(Permutation.identity(5),),
            sizes=(20, 20, 20, 20, 20),
            visible_positions=(0,),
        )
