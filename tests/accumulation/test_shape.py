"""Tests for _shape.py — port of shapeLayer.js."""

from flopscope._accumulation._shape import detect_shape
from flopscope._perm_group import _Permutation as Permutation


def test_detect_shape_trivial_when_no_elements():
    assert detect_shape(va=("i",), wa=("j",), elements=()) == "trivial"


def test_detect_shape_trivial_when_single_element():
    identity = Permutation.identity(2)
    assert detect_shape(va=("i",), wa=("j",), elements=(identity,)) == "trivial"


def test_detect_shape_all_visible_when_w_empty():
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    assert detect_shape(va=("i", "j"), wa=(), elements=(identity, swap)) == "allVisible"


def test_detect_shape_all_summed_when_v_empty():
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    assert detect_shape(va=(), wa=("i", "j"), elements=(identity, swap)) == "allSummed"


def test_detect_shape_mixed_when_both_nonempty():
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    assert detect_shape(va=("i",), wa=("j",), elements=(identity, swap)) == "mixed"
