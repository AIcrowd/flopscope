"""Tests for flops.Module state_dict / save / load / from_file."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


class Linear(flops.Module):
    def __init__(self, i, o):
        self.W = fnp.array(np.zeros((o, i)))
        self.b = fnp.array(np.zeros((o,)))


class MLP(flops.Module):
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.layers = [Linear(a, b) for a, b in zip(sizes, sizes[1:], strict=False)]
        self._scratch = fnp.array(np.ones((3,)))  # underscore → excluded

    def config(self):
        return {"sizes": self.sizes}


def test_state_dict_discovers_nested_and_lists():
    m = MLP([4, 3, 2])
    keys = set(m.state_dict())
    assert keys == {"layers.0.W", "layers.0.b", "layers.1.W", "layers.1.b"}


def test_underscore_attrs_excluded():
    assert "_scratch" not in " ".join(MLP([2, 2]).state_dict())


def test_save_and_from_file_roundtrip(tmp_path):
    m = MLP([4, 3, 2])
    m.layers[0].W = fnp.array(np.full((3, 4), 9.0))
    p = tmp_path / "mlp.npz"
    m.save(str(p))
    m2 = MLP.from_file(str(p))
    assert m2.sizes == [4, 3, 2]
    np.testing.assert_array_equal(np.asarray(m2.layers[0].W), np.full((3, 4), 9.0))


def test_load_in_place(tmp_path):
    m = MLP([4, 3, 2])
    m.layers[1].b = fnp.array(np.array([5.0, 6.0]))
    p = tmp_path / "mlp.npz"
    m.save(str(p))
    fresh = MLP([4, 3, 2])
    fresh.load(str(p))
    np.testing.assert_array_equal(np.asarray(fresh.layers[1].b), np.array([5.0, 6.0]))


def test_load_state_dict_strict_mismatch():
    m = MLP([4, 3, 2])
    with pytest.raises(ValueError, match="mismatch"):
        m.load_state_dict({"layers.0.W": fnp.array(np.zeros((3, 4)))}, strict=True)
