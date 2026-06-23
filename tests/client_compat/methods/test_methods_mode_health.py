"""Guards the construction-patched methods mode: np.array() returns a client
RemoteArray, an existing method works on it, and numpy asserts coerce it."""

from __future__ import annotations


def test_constructor_returns_remote_array():
    import numpy as np

    a = np.array([1, 2, 3])
    assert type(a).__name__ == "RemoteArray", (
        f"got {type(a)!r}; construction not patched"
    )


def test_existing_method_works_on_remote_array():
    import numpy as np

    a = np.array([1.0, 2.0, 3.0])
    assert float(a.sum()) == 6.0


def test_numpy_assert_coerces_remote_array():
    import numpy as np

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(a + b, [5.0, 7.0, 9.0])
