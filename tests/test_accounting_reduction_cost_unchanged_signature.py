"""flopscope.accounting.reduction_cost: signature unchanged, weight still applied."""

from flopscope.accounting import reduction_cost


def test_signature_accepts_op_name_and_returns_weighted_cost():
    cost = reduction_cost("sum", input_shape=(10,), axis=0, symmetry=None)
    assert cost == 9


def test_returns_numeric_type():
    cost = reduction_cost("sum", input_shape=(10,), axis=0, symmetry=None)
    assert isinstance(cost, (int, float))
