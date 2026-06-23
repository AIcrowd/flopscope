from __future__ import annotations


def test_contains():
    import numpy as np

    a = np.array([1.0, 2.0, 3.0])
    assert (2.0 in a) is True
    assert (9.0 in a) is False


def test_pos():
    import numpy as np

    assert (+np.array([1.0, -2.0])).tolist() == [1.0, -2.0]


def test_complex_conv():
    import numpy as np

    assert complex(np.array([3.0])) == complex(3.0)


def test_copy_and_deepcopy():
    import copy

    import numpy as np

    a = np.array([1.0, 2.0])
    assert copy.copy(a).tolist() == [1.0, 2.0]
    assert copy.deepcopy(a).tolist() == [1.0, 2.0]
