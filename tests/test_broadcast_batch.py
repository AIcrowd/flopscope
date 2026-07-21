import pytest

from flopscope._batch import _broadcast_batch


def test_single_operand_matches_old_batch_size():
    assert _broadcast_batch((256, 256), core_ranks=(2,)) == 1
    assert _broadcast_batch((40, 256, 256), core_ranks=(2,)) == 40
    assert _broadcast_batch((5, 4, 8, 8), core_ranks=(2,)) == 20


def test_two_operand_broadcast():
    # a=(m,m) core2 loop (); b=(K,m,n) core2 loop (K,) -> broadcast = (K,)
    assert _broadcast_batch((8, 8), (40, 8, 8), core_ranks=(2, 2)) == 40
    # mutual broadcast of loop dims
    assert _broadcast_batch((3, 1, 8, 8), (1, 4, 8, 8), core_ranks=(2, 2)) == 12
    # b is a single 1-D RHS vector (core rank 1): shape (m,), loop ()
    assert _broadcast_batch((8, 8), (8,), core_ranks=(2, 1)) == 1


def test_incompatible_loop_dims_raise():
    with pytest.raises(ValueError):
        _broadcast_batch((3, 8, 8), (4, 8, 8), core_ranks=(2, 2))
