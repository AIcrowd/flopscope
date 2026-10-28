"""SymmetricTensor.reshape and SymmetricTensor.flatten (issue #347).

Square zeros, ones, full and eye, outer(v, v), a Gram einsum and a stack of
symmetric matrices all return a SymmetricTensor, so its methods run in
ordinary code. ``reshape`` must accept what ``ndarray.reshape`` accepts
(separate ints, a tuple or list, ``order=``, ``copy=``), and ``flatten`` must
return a metered array, as ``ravel`` does, so that everything computed from it
is billed. A plain array with random values is the untagged control in every
parametrized test.
"""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray
from flopscope._symmetric import SymmetricTensor

pytestmark = pytest.mark.filterwarnings("ignore::flopscope.errors.SymmetryLossWarning")

NUMPY_HAS_RESHAPE_COPY = tuple(int(p) for p in np.__version__.split(".")[:2]) >= (
    2,
    1,
)

TAGGED = ["zeros", "ones", "full", "eye", "outer", "gram_einsum", "stack"]
NAMES = [*TAGGED, "plain"]
SQUARE_SHAPES = [(2, 8), (8, 2), (-1, 1), (2, 2, 4), (4, 4)]
STACK_SHAPES = [(9, 3), (3, 9), (-1, 3), (3, 3, 3)]
RESHAPE_CASES = [
    (name, shape)
    for name in NAMES
    for shape in (STACK_SHAPES if name == "stack" else SQUARE_SHAPES)
]
SPLIT = {name: (9, 3) if name == "stack" else (2, 8) for name in NAMES}


@pytest.fixture
def ctx():
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as budget:
        yield budget


@pytest.fixture
def arrays(ctx):
    rng = np.random.default_rng(0)
    v = fnp.asarray(rng.standard_normal(4))
    X = fnp.asarray(rng.standard_normal((6, 4)))
    vs = [fnp.asarray(rng.standard_normal(3)) for _ in range(3)]
    return {
        "zeros": fnp.zeros((4, 4)),
        "ones": fnp.ones((4, 4)),
        "full": fnp.full((4, 4), 2.5),
        "eye": fnp.eye(4),
        "outer": fnp.outer(v, v),
        "gram_einsum": fnp.einsum("ij,ik->jk", X, X),
        "stack": fnp.stack([fnp.outer(u, u) for u in vs]),
        "plain": fnp.asarray(rng.standard_normal((4, 4))),
    }


def _billed(ctx, call):
    before = ctx.flops_used
    result = call()
    return result, ctx.flops_used - before


def test_these_constructors_return_a_symmetric_tensor(arrays):
    for name in TAGGED:
        assert type(arrays[name]) is SymmetricTensor, name
    assert type(arrays["plain"]) is FlopscopeArray


@pytest.mark.parametrize("name, shape", RESHAPE_CASES)
def test_reshape_with_separate_ints_matches_numpy_and_the_tuple_form(
    arrays, name, shape
):
    x = arrays[name]
    expected = np.asarray(x).reshape(shape)
    by_ints = x.reshape(*shape)
    by_tuple = x.reshape(shape)
    np.testing.assert_array_equal(np.asarray(by_ints), expected)
    assert isinstance(by_ints, FlopscopeArray)
    assert type(by_ints) is type(by_tuple)
    assert getattr(by_ints, "symmetry", None) == getattr(by_tuple, "symmetry", None)


@pytest.mark.parametrize("name, shape", RESHAPE_CASES)
def test_reshape_with_a_tuple_or_list_matches_numpy(arrays, name, shape):
    x = arrays[name]
    expected = np.asarray(x).reshape(shape)
    for got in (x.reshape(shape), x.reshape(list(shape))):
        assert isinstance(got, FlopscopeArray)
        np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("order", ["C", "F", "A"])
@pytest.mark.parametrize("name", NAMES)
def test_reshape_order_keyword_matches_numpy(arrays, name, order):
    x = arrays[name]
    expected = np.asarray(x).reshape(SPLIT[name], order=order)
    for got in (
        x.reshape(*SPLIT[name], order=order),
        x.reshape(SPLIT[name], order=order),
    ):
        np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.skipif(
    not NUMPY_HAS_RESHAPE_COPY, reason="ndarray.reshape takes copy= from numpy 2.1"
)
@pytest.mark.parametrize("name", NAMES)
def test_reshape_copy_keyword_matches_numpy(arrays, name):
    x = arrays[name]
    got = x.reshape(*SPLIT[name], copy=True)
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(x).reshape(SPLIT[name], copy=True)
    )
    assert not np.shares_memory(np.asarray(got), np.asarray(x))


@pytest.mark.parametrize("name", NAMES)
def test_reshape_bills_numel_for_every_spelling(ctx, arrays, name):
    x = arrays[name]
    calls = (
        lambda: x.reshape(*SPLIT[name]),
        lambda: x.reshape(SPLIT[name]),
        lambda: x.reshape(*SPLIT[name], order="F"),
    )
    assert [_billed(ctx, call)[1] for call in calls] == [x.size] * len(calls)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("name", NAMES)
def test_flatten_returns_a_metered_copy(arrays, name, order):
    x = arrays[name]
    f = x.flatten(order=order)
    assert type(f) is FlopscopeArray
    np.testing.assert_array_equal(np.asarray(f), np.asarray(x).flatten(order=order))
    assert not np.shares_memory(np.asarray(f), np.asarray(x))


@pytest.mark.parametrize("name", NAMES)
def test_arithmetic_on_flatten_is_billed_like_ravel(ctx, arrays, name):
    x = arrays[name]

    def bills(flat):
        ops = (lambda: flat * flat, lambda: flat @ flat, lambda: flat.sum())
        return [_billed(ctx, op)[1] for op in ops]

    flat_bills = bills(x.flatten())
    assert flat_bills == bills(x.ravel())
    assert min(flat_bills) > 0


def test_matmul_on_a_flattened_outer_product_is_billed(ctx):
    w = fnp.asarray(np.random.default_rng(1).standard_normal(64))
    G = fnp.outer(w, w)

    def matmul_bill(flat):
        m = flat.reshape(64, 64)
        return _billed(ctx, lambda: m @ m)[1]

    via_flatten = matmul_bill(G.flatten())
    assert via_flatten == matmul_bill(G.ravel())
    assert via_flatten > 0
