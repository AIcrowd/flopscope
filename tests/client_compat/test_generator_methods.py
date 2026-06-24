"""RemoteGenerator must proxy permuted + chisquare (numpy Generator parity).

Prod regression (subs 310786, 311261, 311929): "'RemoteGenerator' object has no
attribute 'permuted' / 'chisquare'".
"""

from __future__ import annotations

import flopscope as fnp


def test_generator_chisquare():
    g = fnp.random.default_rng(0)
    out = g.chisquare(3.0, size=5)
    assert type(out).__name__ == "RemoteArray"
    assert out.shape == (5,)
    assert all(v >= 0 for v in out.tolist())  # chi-square is non-negative


def test_generator_permuted():
    g = fnp.random.default_rng(0)
    a = fnp.array([1.0, 2.0, 3.0, 4.0])
    out = g.permuted(a)
    assert type(out).__name__ == "RemoteArray"
    assert sorted(out.tolist()) == [1.0, 2.0, 3.0, 4.0]
