"""Tests for numpy docstring inheritance."""

import numpy as np

import flopscope.numpy as fnp
from flopscope._docstrings import attach_docstring


def test_counted_unary_has_flopscope_cost():
    doc = fnp.exp.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_counted_unary_has_numpy_docstring():
    doc = fnp.exp.__doc__
    # The NumPy docstring content should be present directly (not behind a separator)
    assert "Calculate the exponential" in doc  # pyright: ignore[reportOperatorIssue]


def test_counted_binary_has_flopscope_cost():
    doc = fnp.add.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_free_op_has_zero_cost():
    doc = fnp.zeros.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_flopscope_cost():
    doc = fnp.sum.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_custom_op_has_flopscope_cost():
    doc = fnp.dot.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


# ---------------------------------------------------------------------------
# Returns-section override
# ---------------------------------------------------------------------------
#
# Inheriting NumPy's docstring wholesale is right almost everywhere: a wrapper
# takes the same arguments and returns the same thing, and a FlopscopeArray IS
# an ndarray, so "out : ndarray" stays true. It stops being true when flopscope
# deliberately returns a DIFFERENT type from NumPy -- ``bmat`` is the one such
# op today -- and an inherited Returns section then states the wrong type in
# the published reference. The override replaces just that section, and records
# itself on the wrapper so the docs generator (which reads NumPy's docstring
# directly, not the wrapper's) can apply the same correction.


def test_returns_override_replaces_the_inherited_returns_section():
    def wrapper():
        pass  # pragma: no cover - only its __doc__ is under test

    attach_docstring(
        wrapper,
        np.bmat,
        "counted_custom",
        "numel(output) FLOPs",
        returns=("FlopscopeArray", "A plain 2-D flopscope array."),
    )
    doc = wrapper.__doc__ or ""
    assert "out : FlopscopeArray" in doc
    assert "A plain 2-D flopscope array." in doc
    assert "Returns a matrix object" not in doc, (
        "the override left NumPy's own Returns body in place"
    )
    # Everything else NumPy documented must survive the surgery.
    assert "FLOP Cost" in doc
    assert "Parameters" in doc
    assert "See Also" in doc
    assert "Examples" in doc


def test_returns_override_is_recorded_for_the_docs_generator():
    def wrapper():
        pass  # pragma: no cover - only its attributes are under test

    attach_docstring(
        wrapper,
        np.bmat,
        "counted_custom",
        "numel(output) FLOPs",
        returns=("FlopscopeArray", "A plain 2-D flopscope array."),
    )
    assert getattr(wrapper, "__flopscope_returns__") == (  # noqa: B009
        "FlopscopeArray",
        "A plain 2-D flopscope array.",
    )


def test_docstrings_without_an_override_are_untouched():
    """The override is opt-in; the 256 other call sites must not move."""

    def wrapper():
        pass  # pragma: no cover - only its __doc__ is under test

    attach_docstring(wrapper, np.bmat, "counted_custom", "numel(output) FLOPs")
    assert "Returns a matrix object" in (wrapper.__doc__ or "")
    assert not hasattr(wrapper, "__flopscope_returns__")
