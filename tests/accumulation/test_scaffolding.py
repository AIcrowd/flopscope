"""Scaffolding test — verifies the new subpackage exists and all internal modules import."""


def test_accumulation_package_imports():
    import flopscope._accumulation  # noqa: F401


def test_accumulation_internal_modules_import():
    from flopscope._accumulation import (
        _bipartite,
        _burnside,
        _components,
        _cost,
        _detection,
        _ladder,
        _output_orbit,
        _partition,
        _regimes,
        _shape,
        _wreath,
    )

    # Touch each module so the linter doesn't complain about unused imports.
    for mod in (
        _bipartite,
        _burnside,
        _components,
        _cost,
        _detection,
        _ladder,
        _output_orbit,
        _partition,
        _regimes,
        _shape,
        _wreath,
    ):
        assert mod.__doc__ is not None
