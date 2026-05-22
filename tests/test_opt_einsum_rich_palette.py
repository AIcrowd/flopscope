from __future__ import annotations

from opt_einsum.parser import get_symbol

import flopscope._opt_einsum._hsluv as hsluv
from flopscope._opt_einsum._contract import PathInfo


def _alpha_symbols(count: int) -> list[str]:
    symbols: list[str] = []
    index = 0
    while len(symbols) < count:
        symbol = get_symbol(index)
        if symbol.isalpha():
            symbols.append(symbol)
        index += 1
    return symbols


def _pathinfo_for_expression(
    input_subscripts: str, output_subscript: str = ""
) -> PathInfo:
    return PathInfo(
        path=[],
        steps=[],
        naive_cost=0,
        optimized_cost=0,
        largest_intermediate=0,
        speedup=1.0,
        input_subscripts=input_subscripts,
        output_subscript=output_subscript,
    )


def test_precomputed_palette_matches_historical_generator() -> None:
    assert hsluv.PRECOMPUTED_QUALITATIVE_HSLUV_PALETTE_64 == tuple(
        hsluv.qualitative_hsluv_palette(64, lightness=0.5, saturation=0.9)
    )


def test_rich_label_palette_64_does_not_call_generator(monkeypatch) -> None:
    hsluv.rich_label_palette.cache_clear()

    def _explode(*args, **kwargs):
        raise AssertionError("qualitative_hsluv_palette should not be called for 64")

    monkeypatch.setattr(hsluv, "qualitative_hsluv_palette", _explode)

    assert (
        hsluv.rich_label_palette(64) == hsluv.PRECOMPUTED_QUALITATIVE_HSLUV_PALETTE_64
    )


def test_rich_label_palette_larger_sizes_are_lazy_and_cached(monkeypatch) -> None:
    hsluv.rich_label_palette.cache_clear()
    calls: list[int] = []

    def _fake_palette(
        n_colors: int,
        *,
        hue: float = 0.01,
        saturation: float = 0.9,
        lightness: float = 0.5,
        hue_samples: int | None = None,
    ) -> list[str]:
        calls.append(n_colors)
        assert hue == 0.01
        assert saturation == 0.9
        assert lightness == 0.5
        assert hue_samples is None
        return [f"#{index:06X}" for index in range(n_colors)]

    monkeypatch.setattr(hsluv, "qualitative_hsluv_palette", _fake_palette)

    first = hsluv.rich_label_palette(80)
    second = hsluv.rich_label_palette(80)

    assert calls == [80]
    assert len(first) == 80
    assert first == second


def test_pathinfo_label_styles_are_unique_and_stable_for_normal_expression() -> None:
    info = _pathinfo_for_expression("ij,ik", "jk")

    first = info._label_styles
    second = info._label_styles

    assert set(first) == {"i", "j", "k"}
    assert len(set(first.values())) == 3
    assert first == second


def test_pathinfo_label_styles_support_more_than_64_active_labels() -> None:
    labels = _alpha_symbols(65)
    info = _pathinfo_for_expression("".join(labels))

    styles = info._label_styles

    assert len(styles) == 65
    assert len(set(styles.values())) == 65
