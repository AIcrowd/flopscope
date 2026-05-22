"""Smoke tests for uncovered free (zero-FLOP) operations.

Each test just verifies the function runs and returns the expected type/shape.
These are all trivial pass-throughs to numpy so fnp just need to execute them.
"""

import numpy
import pytest

import flopscope as flops
import flopscope._free_ops as ops
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor
from flopscope.errors import SymmetryLossWarning

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def arr(*shape, dtype=float):
    return numpy.arange(1, numpy.prod(shape) + 1, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Tensor creation wrappers (uncovered)
# ---------------------------------------------------------------------------


def test_full_like():
    a = numpy.zeros((2, 3))
    r = ops.full_like(a, 7.0)
    assert r.shape == (2, 3)
    assert numpy.all(r == 7.0)


def test_empty():
    r = ops.empty((3, 4))
    assert r.shape == (3, 4)


def test_empty_like():
    a = numpy.zeros((2, 5))
    r = ops.empty_like(a)
    assert r.shape == (2, 5)


def test_identity():
    r = ops.identity(4)
    assert r.shape == (4, 4)
    assert numpy.allclose(r, numpy.eye(4))


# ---------------------------------------------------------------------------
# Tensor manipulation wrappers (uncovered)
# ---------------------------------------------------------------------------


def test_swapaxes():
    a = arr(2, 3, 4)
    r = ops.swapaxes(a, 0, 2)
    assert r.shape == (4, 3, 2)


def test_moveaxis():
    a = arr(2, 3, 4)
    r = ops.moveaxis(a, 0, -1)
    assert r.shape == (3, 4, 2)


def test_moveaxis_remaps_symmetry():
    a = flops.as_symmetric(
        numpy.stack([numpy.eye(3), 2 * numpy.eye(3)], axis=-1), symmetry=(0, 1)
    )
    r = ops.moveaxis(a, 2, 0)
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (2, 3, 3)
    assert r.symmetry == flops.SymmetryGroup.symmetric(axes=(1, 2))


def test_expand_dims_remaps_full_symmetry_from_issue_49():
    a = flops.as_symmetric(numpy.ones((3, 3, 3)), symmetry=(0, 1, 2))
    r = ops.expand_dims(a, axis=1)
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (3, 1, 3, 3)
    assert r.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 2, 3))


def test_expand_dims_remaps_partial_symmetry_front_middle_and_end():
    a = flops.as_symmetric(
        numpy.stack([numpy.eye(3), 2 * numpy.eye(3)], axis=-1), symmetry=(0, 1)
    )
    front = ops.expand_dims(a, axis=0)
    middle = ops.expand_dims(a, axis=1)
    end = ops.expand_dims(a, axis=-1)
    assert front.symmetry == flops.SymmetryGroup.symmetric(axes=(1, 2))  # pyright: ignore[reportAttributeAccessIssue]
    assert middle.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 2))  # pyright: ignore[reportAttributeAccessIssue]
    assert end.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))  # pyright: ignore[reportAttributeAccessIssue]


def test_expand_dims_negative_axis_remaps_symmetry():
    a = flops.as_symmetric(numpy.ones((3, 3, 3)), symmetry=(0, 1, 2))
    r = ops.expand_dims(a, axis=-4)
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (1, 3, 3, 3)
    assert r.symmetry == flops.SymmetryGroup.symmetric(axes=(1, 2, 3))


def test_expand_dims_tuple_axis_remaps_symmetry():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))
    r = ops.expand_dims(a, axis=(0, 2))
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (1, 3, 1, 3)
    assert r.symmetry == flops.SymmetryGroup.young(blocks=((0, 2), (1, 3)))


def test_expand_dims_plain_arrays_remain_plain():
    r = ops.expand_dims(numpy.array([1, 2, 3]), axis=0)
    assert not isinstance(r, SymmetricTensor)
    assert r.shape == (1, 3)


def test_expand_dims_plain_arrays_gain_inserted_axis_symmetry():
    r = ops.expand_dims(numpy.array([1, 2, 3]), axis=(1, 2, 3))
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (3, 1, 1, 1)
    assert r.symmetry == flops.SymmetryGroup.symmetric(axes=(1, 2, 3))


def test_expand_dims_invalid_axis_matches_numpy():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))
    with pytest.raises(Exception) as ours:
        ops.expand_dims(a, axis=4)
    with pytest.raises(Exception) as numpy_err:
        numpy.expand_dims(numpy.eye(3), axis=4)
    assert type(ours.value) is type(numpy_err.value)


def test_expand_dims_duplicate_axes_match_numpy():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))
    with pytest.raises(Exception) as ours:
        ops.expand_dims(a, axis=(0, 0))
    with pytest.raises(Exception) as numpy_err:
        numpy.expand_dims(numpy.eye(3), axis=(0, 0))
    assert type(ours.value) is type(numpy_err.value)


def test_newaxis_slicing_matches_expand_dims():
    a = fnp.eye(3)
    by_fn = ops.expand_dims(a, axis=(0, 2))
    by_slice = a[None, :, None, :]
    assert isinstance(by_fn, SymmetricTensor)
    assert isinstance(by_slice, SymmetricTensor)
    assert numpy.array_equal(by_fn, by_slice)
    assert by_slice.symmetry == by_fn.symmetry
    assert by_slice.symmetry == flops.SymmetryGroup.young(blocks=((0, 2), (1, 3)))


def test_vstack():
    r = ops.vstack([numpy.ones((2, 3)), numpy.zeros((2, 3))])
    assert r.shape == (4, 3)


def test_hstack():
    r = ops.hstack([numpy.ones((3, 2)), numpy.zeros((3, 2))])
    assert r.shape == (3, 4)


def test_split():
    parts = ops.split(numpy.arange(6), 3)
    assert len(parts) == 3


def test_hsplit():
    a = numpy.arange(12).reshape(3, 4)
    parts = ops.hsplit(a, 2)
    assert len(parts) == 2


def test_vsplit():
    a = numpy.arange(12).reshape(4, 3)
    parts = ops.vsplit(a, 2)
    assert len(parts) == 2


def test_ravel():
    a = arr(2, 3)
    r = ops.ravel(a)
    assert r.shape == (6,)


def test_ravel_drops_symmetry():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))
    with pytest.warns(SymmetryLossWarning):
        r = ops.ravel(a)
    assert r.shape == (9,)
    assert not isinstance(r, SymmetricTensor)


def test_tile():
    a = numpy.array([1, 2])
    r = ops.tile(a, 3)
    assert list(r) == [1, 2, 1, 2, 1, 2]


def test_repeat():
    a = numpy.array([1, 2, 3])
    r = ops.repeat(a, 2)
    assert r.shape == (6,)


def test_flip():
    a = numpy.array([1, 2, 3])
    r = ops.flip(a)
    assert list(r) == [3, 2, 1]


def test_roll():
    a = numpy.array([1, 2, 3, 4])
    r = ops.roll(a, 1)
    assert list(r) == [4, 1, 2, 3]


def test_pad():
    a = numpy.array([1, 2, 3])
    r = ops.pad(a, 1)
    assert r.shape == (5,)


def test_tril():
    a = numpy.ones((3, 3))
    r = ops.tril(a)
    assert numpy.allclose(r, numpy.tril(a))


def test_diagonal():
    a = numpy.array([[1, 2], [3, 4]])
    r = ops.diagonal(a)
    assert list(r) == [1, 4]


def test_diagonal_view_copy_does_not_crash():
    a = flops.as_symmetric(numpy.eye(4), symmetry=(0, 1))
    copied = fnp.diagonal(a).copy()
    assert copied.shape == (4,)
    assert numpy.allclose(copied, numpy.diagonal(numpy.asarray(a)))


def test_broadcast_to():
    a = numpy.array([1, 2, 3])
    r = ops.broadcast_to(a, (2, 3))
    assert r.shape == (2, 3)


def test_broadcast_to_only_adds_equal_size_blocks():
    a = numpy.array([1, 2])
    r = ops.broadcast_to(a, (2, 3, 2))
    assert r.shape == (2, 3, 2)
    assert not isinstance(r, SymmetricTensor)


def test_broadcast_to_adds_equal_size_created_block():
    a = numpy.array([1, 2])
    r = ops.broadcast_to(a, (4, 4, 2))
    assert isinstance(r, SymmetricTensor)
    assert r.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))


def test_meshgrid():
    x = numpy.array([1, 2])
    y = numpy.array([3, 4, 5])
    Xg, Yg = ops.meshgrid(x, y)
    assert Xg.shape == (3, 2)


def test_transpose_negative_axes_preserves_symmetry():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))
    r = ops.transpose(a, axes=(-1, -2))
    assert isinstance(r, SymmetricTensor)
    assert r.shape == (3, 3)
    assert r.symmetry == a.symmetry


def test_astype():
    a = numpy.array([1.0, 2.0])
    r = ops.astype(a, numpy.int32)
    assert r.dtype == numpy.int32


def test_astype_and_copy_do_not_leak_invalid_view_symmetry():
    a = flops.as_symmetric(numpy.eye(3), symmetry=(0, 1))

    cast = ops.astype(a, a.dtype, copy=False)
    assert cast.shape == a.shape
    assert not isinstance(cast, SymmetricTensor)

    method_cast = a.astype(a.dtype, copy=False)
    assert method_cast.shape == a.shape
    assert not isinstance(method_cast, SymmetricTensor)

    copied = ops.copy(a)
    assert isinstance(copied, SymmetricTensor)
    assert copied.symmetry == a.symmetry

    method_copy = a.copy()
    assert isinstance(method_copy, SymmetricTensor)
    assert method_copy.symmetry == a.symmetry


def test_asarray():
    r = ops.asarray([1, 2, 3])
    assert isinstance(r, numpy.ndarray)


# ---------------------------------------------------------------------------
# Type / info helpers and new free ops (uncovered lines ~438+)
# ---------------------------------------------------------------------------


def test_isnan():
    a = numpy.array([1.0, float("nan"), 3.0])
    r = ops.isnan(a)
    assert list(r) == [False, True, False]


def test_isinf():
    a = numpy.array([1.0, float("inf"), 3.0])
    r = ops.isinf(a)
    assert list(r) == [False, True, False]


def test_isfinite():
    a = numpy.array([1.0, float("inf"), float("nan")])
    r = ops.isfinite(a)
    assert list(r) == [True, False, False]


def test_append():
    a = numpy.array([1, 2])
    r = ops.append(a, 3)
    assert list(r) == [1, 2, 3]


def test_argwhere():
    a = numpy.array([0, 1, 0, 2])
    r = ops.argwhere(a)
    assert r.shape == (2, 1)


def test_array_split():
    parts = ops.array_split(numpy.arange(5), 3)
    assert len(parts) == 3


def test_asarray_chkfinite():
    r = ops.asarray_chkfinite([1.0, 2.0])
    assert isinstance(r, numpy.ndarray)


def test_atleast_1d():
    r = ops.atleast_1d(5)
    assert r.shape == (1,)  # pyright: ignore[reportAttributeAccessIssue]


def test_atleast_2d():
    r = ops.atleast_2d(numpy.array([1, 2]))
    assert r.ndim == 2  # pyright: ignore[reportAttributeAccessIssue]


def test_atleast_3d():
    r = ops.atleast_3d(numpy.array([1, 2]))
    assert r.ndim == 3  # pyright: ignore[reportAttributeAccessIssue]


def test_base_repr():
    r = ops.base_repr(255, base=16)
    assert isinstance(r, str)


def test_binary_repr():
    r = ops.binary_repr(5)
    assert r == "101"


def test_block():
    r = ops.block([[numpy.ones((2, 2)), numpy.zeros((2, 2))]])
    assert r.shape == (2, 4)


def test_broadcast_arrays():
    a = numpy.array([1, 2, 3])
    b = numpy.array([[1], [2]])
    results = ops.broadcast_arrays(a, b)
    assert len(results) == 2


def test_broadcast_shapes():
    r = ops.broadcast_shapes((3, 1), (1, 4))
    assert r == (3, 4)


def test_can_cast():
    assert ops.can_cast(numpy.float32, numpy.float64)


def test_choose():
    choices = [numpy.array([0, 1, 2]), numpy.array([10, 11, 12])]
    r = ops.choose(numpy.array([0, 1, 0]), choices)
    assert list(r) == [0, 11, 2]


def test_column_stack():
    r = ops.column_stack([numpy.array([1, 2]), numpy.array([3, 4])])
    assert r.shape == (2, 2)


def test_common_type():
    r = ops.common_type(numpy.array([1], dtype=numpy.float32), numpy.array([1.0]))
    assert r in (numpy.float64, numpy.complex128, numpy.float32)


def test_compress():
    a = numpy.array([1, 2, 3, 4])
    r = ops.compress([True, False, True, False], a)
    assert list(r) == [1, 3]


def test_concat():
    r = ops.concat([numpy.array([1, 2]), numpy.array([3, 4])])
    assert list(r) == [1, 2, 3, 4]


def test_copyto():
    dst = numpy.zeros(3)
    ops.copyto(dst, numpy.array([1.0, 2.0, 3.0]))
    assert list(dst) == [1.0, 2.0, 3.0]


def test_delete():
    a = numpy.array([1, 2, 3, 4])
    r = ops.delete(a, 1)
    assert list(r) == [1, 3, 4]


def test_diag_indices():
    r = ops.diag_indices(3)
    assert len(r) == 2


def test_diag_indices_from():
    a = numpy.eye(3)
    r = ops.diag_indices_from(a)
    assert len(r) == 2


def test_diagflat():
    r = ops.diagflat([1, 2, 3])
    assert r.shape == (3, 3)


def test_dsplit():
    a = numpy.zeros((2, 2, 4))
    parts = ops.dsplit(a, 2)
    assert len(parts) == 2


def test_dstack():
    a = numpy.ones((2, 2))
    b = numpy.zeros((2, 2))
    r = ops.dstack([a, b])
    assert r.shape == (2, 2, 2)


def test_extract():
    a = numpy.array([1, 2, 3, 4])
    cond = a > 2
    r = ops.extract(cond, a)
    assert list(r) == [3, 4]


def test_fill_diagonal():
    a = numpy.zeros((3, 3))
    ops.fill_diagonal(a, 5)
    assert numpy.all(numpy.diag(a) == 5)


def test_flatnonzero():
    a = numpy.array([0, 1, 0, 2])
    r = ops.flatnonzero(a)
    assert list(r) == [1, 3]


def test_fliplr():
    a = numpy.array([[1, 2], [3, 4]])
    r = ops.fliplr(a)
    assert list(r[0]) == [2, 1]


def test_flipud():
    a = numpy.array([[1, 2], [3, 4]])
    r = ops.flipud(a)
    assert list(r[0]) == [3, 4]


def test_fromfunction():
    r = ops.fromfunction(lambda i, j: i + j, (3, 3))
    assert r.shape == (3, 3)


def test_fromiter():
    r = ops.fromiter(range(5), dtype=float)
    assert list(r) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_fromstring():
    r = ops.fromstring("1 2 3", sep=" ")
    assert list(r) == [1.0, 2.0, 3.0]


def test_indices():
    r = ops.indices((2, 3))
    assert r.shape == (2, 2, 3)


def test_insert():
    a = numpy.array([1, 2, 3])
    r = ops.insert(a, 1, 99)
    assert list(r) == [1, 99, 2, 3]


def test_isdtype():
    assert ops.isdtype(numpy.float64, "real floating")


def test_isfortran():
    a = numpy.asfortranarray(numpy.ones((3, 3)))
    assert ops.isfortran(a)


def test_isin():
    a = numpy.array([1, 2, 3])
    b = numpy.array([2])
    r = ops.isin(a, b)
    assert list(r) == [False, True, False]


def test_isscalar():
    assert ops.isscalar(5.0)
    assert not ops.isscalar(numpy.array([5.0]))


def test_issubdtype():
    assert ops.issubdtype(numpy.int32, numpy.integer)


def test_iterable():
    assert ops.iterable([1, 2, 3])
    assert not ops.iterable(5)


def test_ix_():
    r = ops.ix_([0, 1], [2, 3])
    assert len(r) == 2


def test_mask_indices():
    r = ops.mask_indices(3, numpy.triu)
    assert len(r) == 2


def test_matrix_transpose():
    a = numpy.ones((2, 3))
    r = ops.matrix_transpose(a)
    assert r.shape == (3, 2)


def test_may_share_memory():
    a = numpy.ones(5)
    b = a[1:3]
    assert ops.may_share_memory(a, b)


def test_min_scalar_type():
    r = ops.min_scalar_type(10)
    assert isinstance(r, numpy.dtype)


def test_mintypecode():
    r = ops.mintypecode(["d", "f"])
    assert isinstance(r, str)


def test_ndim():
    assert ops.ndim(numpy.ones((2, 3))) == 2


def test_nonzero():
    a = numpy.array([0, 1, 0, 2])
    r = ops.nonzero(a)
    assert list(r[0]) == [1, 3]


def test_packbits():
    a = numpy.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=numpy.uint8)
    r = ops.packbits(a)
    assert r.shape == (1,)


def test_permute_dims():
    a = numpy.ones((2, 3, 4))
    r = ops.permute_dims(a, (2, 0, 1))
    assert r.shape == (4, 2, 3)


def test_promote_types():
    r = ops.promote_types(numpy.float32, numpy.float64)
    assert r == numpy.float64


def test_put():
    a = numpy.zeros(5)
    ops.put(a, [1, 3], [10.0, 20.0])
    assert a[1] == 10.0


def test_put_along_axis():
    a = numpy.zeros((3, 3))
    indices = numpy.array([[0, 1, 2]])
    ops.put_along_axis(a, indices, 5.0, axis=0)
    assert a[0, 0] == 5.0


def test_putmask():
    a = numpy.array([1.0, 2.0, 3.0])
    ops.putmask(a, a > 1, 0.0)
    assert a[0] == 1.0
    assert a[1] == 0.0


def test_ravel_multi_index():
    r = ops.ravel_multi_index(([0, 1], [1, 2]), (3, 3))
    assert list(r) == [1, 5]


def test_require():
    a = numpy.ones((3, 3))
    r = ops.require(a, dtype=numpy.float64)
    assert r.dtype == numpy.float64


def test_resize():
    r = ops.resize(numpy.arange(4), (2, 4))
    assert r.shape == (2, 4)


def test_result_type():
    r = ops.result_type(numpy.float32, numpy.int64)
    assert isinstance(r, numpy.dtype)


def test_rollaxis():
    a = numpy.ones((2, 3, 4))
    r = ops.rollaxis(a, 2)
    assert r.shape == (4, 2, 3)


def test_rot90():
    a = numpy.array([[1, 2], [3, 4]])
    r = ops.rot90(a)
    assert r.shape == (2, 2)


def test_row_stack():
    r = ops.row_stack([numpy.array([1, 2]), numpy.array([3, 4])])
    assert r.shape == (2, 2)


def test_select():
    x = numpy.array([1, 2, 3, 4])
    conds = [x < 2, x > 3]
    choices = [-1, 1]
    r = ops.select(conds, choices, default=0)
    assert list(r) == [-1, 0, 0, 1]


def test_shape():
    r = ops.shape(numpy.ones((2, 3, 4)))
    assert r == (2, 3, 4)


def test_shares_memory():
    a = numpy.ones(5)
    b = a[1:3]
    assert ops.shares_memory(a, b)


def test_size():
    assert ops.size(numpy.ones((2, 3))) == 6


def test_take():
    a = numpy.array([10, 20, 30, 40])
    r = ops.take(a, [0, 2])
    assert list(r) == [10, 30]


def test_take_along_axis():
    a = numpy.array([[10, 20], [30, 40]])
    idx = numpy.array([[1, 0]])
    r = ops.take_along_axis(a, idx, axis=0)
    assert r.shape == (1, 2)


def test_tri():
    r = ops.tri(3)
    assert r.shape == (3, 3)


def test_tril_indices():
    r = ops.tril_indices(3)
    assert len(r) == 2


def test_tril_indices_from():
    a = numpy.ones((4, 4))
    r = ops.tril_indices_from(a)
    assert len(r) == 2


def test_trim_zeros():
    a = numpy.array([0, 0, 1, 2, 0])
    r = ops.trim_zeros(a)
    assert list(r) == [1, 2]


def test_triu_indices():
    r = ops.triu_indices(3)
    assert len(r) == 2


def test_triu_indices_from():
    a = numpy.ones((4, 4))
    r = ops.triu_indices_from(a)
    assert len(r) == 2


def test_typename():
    r = ops.typename("f")
    assert isinstance(r, str)


def test_unpackbits():
    a = numpy.array([85], dtype=numpy.uint8)
    r = ops.unpackbits(a)
    assert r.shape == (8,)


def test_unravel_index():
    r = ops.unravel_index(5, (3, 3))
    assert r == (1, 2)


def test_unstack():
    a = numpy.array([[1, 2], [3, 4], [5, 6]])
    parts = ops.unstack(a)
    assert len(parts) == 3


def test_frombuffer():
    import struct

    buf = struct.pack("3f", 1.0, 2.0, 3.0)
    r = ops.frombuffer(buf, dtype=numpy.float32)
    assert list(r) == pytest.approx([1.0, 2.0, 3.0])


def test_linspace():
    r = ops.linspace(0, 1, 5)
    assert r.shape == (5,)
    assert numpy.isclose(r[0], 0.0)
    assert numpy.isclose(r[-1], 1.0)
