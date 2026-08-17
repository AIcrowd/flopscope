"""Docstring inheritance helper for flopscope wrappers."""

from __future__ import annotations

SECTION_HEADERS = frozenset(
    {
        "Parameters",
        "Returns",
        "Raises",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Yields",
        "Warns",
        "Other Parameters",
        "Attributes",
        "Methods",
    }
)


def _is_section_header(lines: list[str], i: int) -> bool:
    """True when ``lines[i]`` is a NumPy-style section header with its underline."""
    return (
        lines[i].strip() in SECTION_HEADERS
        and i + 1 < len(lines)
        and lines[i + 1].strip().startswith("---")
    )


def _replace_returns_section(np_doc: str, returns: tuple[str, str]) -> str:
    """Swap the body of NumPy's ``Returns`` section for flopscope's own.

    Used only where flopscope deliberately returns a different type from
    NumPy, so that the inherited docstring does not state the wrong one.
    Leaves every other section, and the doc's own indentation, alone. A
    docstring with no ``Returns`` section is returned unchanged -- the caller
    is asserting what the op returns, not that NumPy documented it.
    """
    name, description = returns
    lines = np_doc.split("\n")

    start = next(
        (
            i
            for i in range(len(lines))
            if lines[i].strip() == "Returns" and _is_section_header(lines, i)
        ),
        None,
    )
    if start is None:
        return np_doc

    # The section runs to the next section header, or to the end of the doc.
    end = next(
        (i for i in range(start + 2, len(lines)) if _is_section_header(lines, i)),
        len(lines),
    )
    # Keep any trailing blank lines that separated this section from the next.
    while end > start + 2 and not lines[end - 1].strip():
        end -= 1

    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    replacement = [
        f"{indent}Returns",
        f"{indent}-------",
        f"{indent}out : {name}",
        f"{indent}    {description}",
    ]
    return "\n".join(lines[:start] + replacement + lines[end:])


def attach_docstring(
    wrapper,
    np_func,
    category: str,
    cost_description: str,
    *,
    returns: tuple[str, str] | None = None,
) -> None:
    """Attach NumPy's docstring to a flopscope wrapper with a FLOP Cost section.

    Inserts a dedicated "FLOP Cost" section after the summary line(s),
    before the Parameters section. This keeps cost info prominent and
    separate from NumPy's own Notes.

    ``returns`` is an optional ``(type, description)`` pair that replaces
    NumPy's own ``Returns`` section. Inheriting that section wholesale is
    right almost everywhere -- a wrapper returns the same thing NumPy does,
    and a ``FlopscopeArray`` IS an ``ndarray``, so "out : ndarray" stays true.
    It is wrong where flopscope deliberately returns a DIFFERENT type, which
    would otherwise leave the published reference stating a type the op does
    not return. The pair is also recorded on the wrapper as
    ``__flopscope_returns__`` for ``scripts/generate_api_docs.py``, which
    builds the website from NumPy's docstring directly rather than from this
    one and so cannot see the substitution otherwise.
    """
    np_doc = getattr(np_func, "__doc__", None) or ""

    cost_section = f"FLOP Cost\n---------\n{cost_description}\n"

    if returns is not None:
        wrapper.__flopscope_returns__ = returns
        np_doc = _replace_returns_section(np_doc, returns)

    if not np_doc:
        wrapper.__doc__ = (
            f"Counted wrapper for ``numpy.{np_func.__name__}``.\n\n{cost_section}"
        )
        return

    # Find the first standard section header (Parameters, Returns, etc.)
    # and insert the cost section before it.
    lines = np_doc.split("\n")

    insert_idx = None
    for i in range(len(lines)):
        if _is_section_header(lines, i):
            insert_idx = i
            break

    if insert_idx is not None:
        # Insert cost section before the first standard section
        before = "\n".join(lines[:insert_idx]).rstrip()
        after = "\n".join(lines[insert_idx:])
        wrapper.__doc__ = f"{before}\n\n{cost_section}\n{after}"
    else:
        # No standard sections found — append cost section
        wrapper.__doc__ = f"{np_doc.rstrip()}\n\n{cost_section}"
