from __future__ import annotations


def test_itemsize_matches():
    import numpy as np

    assert np.array([1.0, 2.0, 3.0]).itemsize == 8  # float64


def test_strides_is_tuple():
    import numpy as np

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert isinstance(a.strides, tuple) and len(a.strides) == 2


def test_flags_contiguous_and_writeable():
    import numpy as np

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert a.flags["C_CONTIGUOUS"] is True
    assert a.flags["WRITEABLE"] is False  # client RemoteArray is immutable
