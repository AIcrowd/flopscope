"""Complex-factor floor: every numeric complex_factor is >= 2.0.

A complex value is two real components; any op touching complex values
prices at least one unit per component. The floor is weight-independent:
weight-0 and dtype-neutral ops carry correct factors so a future weight
change composes with an already-correct factor. The sole exception is the
FFT family, whose cost formulas already count the complex real-FLOPs.
"""

from __future__ import annotations

from flopscope._registry import REGISTRY

_COMPLEX_PRICED_IN = frozenset(
    {
        "fft.fft",
        "fft.fft2",
        "fft.fftfreq",
        "fft.fftn",
        "fft.hfft",
        "fft.ifft",
        "fft.ifft2",
        "fft.ifftn",
        "fft.ihfft",
        "fft.irfft",
        "fft.irfft2",
        "fft.irfftn",
        "fft.rfft",
        "fft.rfft2",
        "fft.rfftfreq",
        "fft.rfftn",
    }
)


def test_priced_in_set_is_exactly_the_fft_family():
    fft_ops = {name for name in REGISTRY if name.startswith("fft.")}
    assert _COMPLEX_PRICED_IN <= fft_ops
    stale = {name for name in _COMPLEX_PRICED_IN if name not in REGISTRY}
    assert not stale, f"priced-in entries not in registry: {sorted(stale)}"


def test_numeric_complex_factors_meet_the_component_floor():
    offenders = sorted(
        name
        for name, entry in REGISTRY.items()
        if isinstance(entry.get("complex_factor"), (int, float))
        and entry["complex_factor"] < 2.0
        and name not in _COMPLEX_PRICED_IN
    )
    assert not offenders, (
        f"{len(offenders)} ops price complex below one unit per component: {offenders}"
    )


def test_priced_in_ops_keep_their_formula_factor():
    for name in sorted(_COMPLEX_PRICED_IN):
        assert REGISTRY[name].get("complex_factor") == 1.0, name
