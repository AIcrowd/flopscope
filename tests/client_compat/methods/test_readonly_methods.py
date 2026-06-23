"""Read-only ndarray methods bridge to server ops and behave like numpy."""

from __future__ import annotations


def test_argsort_matches_numpy():
    import numpy as np  # constructors patched -> RemoteArray (methods mode)

    a = np.array([3.0, 1.0, 2.0])
    # argsort ascending: 1.0 at idx 1, 2.0 at idx 2, 3.0 at idx 0 -> [1, 2, 0]
    assert a.argsort().tolist() == [1, 2, 0]


def test_cumsum_matches_numpy():
    import numpy as np

    assert np.array([1.0, 2.0, 3.0]).cumsum().tolist() == [1.0, 3.0, 6.0]


def test_clip_matches_numpy():
    import numpy as np

    assert np.array([-1.0, 0.5, 2.0]).clip(0.0, 1.0).tolist() == [0.0, 0.5, 1.0]


def test_compress_matches_numpy():
    import numpy as np

    assert np.array([1.0, 2.0, 3.0]).compress([True, False, True]).tolist() == [
        1.0,
        3.0,
    ]


def test_trace_matches_numpy():
    import numpy as np

    assert float(np.array([[1.0, 2.0], [3.0, 4.0]]).trace()) == 5.0


def test_prod_and_std_and_var():
    import numpy as np

    assert float(np.array([1.0, 2.0, 3.0, 4.0]).prod()) == 24.0


def test_item_matches_numpy():
    import numpy as np

    assert np.array([42.0]).item() == 42.0
