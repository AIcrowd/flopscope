"""Module-level ``__getattr__`` factory for helpful error messages.

Produces a ``__getattr__`` function that checks the registry and raises
:class:`AttributeError` with a descriptive message for blacklisted,
registered-but-unimplemented, or completely unknown names.
"""

from __future__ import annotations

from flopscope._registry import BLACKLISTED, get_category

try:
    from flopscope._server_only_data import SERVER_ONLY
except ImportError:  # pragma: no cover - pre-sync safety
    SERVER_ONLY = frozenset()

# numpy abstract scalar types: have no meaning in the numpy-free remote client.
_ABSTRACT_SCALAR_TYPES = frozenset(
    {
        "generic",
        "number",
        "integer",
        "signedinteger",
        "unsignedinteger",
        "inexact",
        "floating",
        "complexfloating",
        "flexible",
        "character",
    }
)


def make_module_getattr(module_prefix: str, module_label: str):
    """Return a ``__getattr__`` suitable for assignment at module scope.

    Parameters
    ----------
    module_prefix:
        Prefix prepended before looking up the registry, e.g. ``"fft."``
        for the ``flopscope.numpy.fft`` submodule.  Pass ``""`` for the top-level
        package.
    module_label:
        Human-readable module name used in error messages, e.g.
        ``"flopscope.numpy.fft"``.
    """

    def __getattr__(name: str):
        # Skip dunder/private names to avoid interfering with import machinery
        if name.startswith("_"):
            raise AttributeError(f"module '{module_label}' has no attribute '{name}'")

        qualified = f"{module_prefix}{name}" if module_prefix else name
        category = get_category(qualified)

        if category == BLACKLISTED:
            raise AttributeError(
                f"'{module_label}.{name}' is not supported in the flopscope "
                f"client. It is intentionally excluded (I/O, string formatting, "
                f"iterators, global state, dtype introspection, or other "
                f"operations that aren't meaningful in a remote-compute "
                f"environment)."
            )
        elif category is not None:
            raise AttributeError(
                f"'{module_label}.{name}' is registered but not yet "
                f"implemented as a client-side proxy. "
                f"Category: {category}."
            )
        else:
            if qualified in SERVER_ONLY:
                raise AttributeError(
                    f"'{module_label}.{name}' is a flopscope server-side / "
                    f"analysis API and is not available in the flopscope client. "
                    f"It is only usable when running flopscope in-process (the "
                    f"starter kit), not on the remote grader."
                )
            if name in _ABSTRACT_SCALAR_TYPES:
                raise AttributeError(
                    f"'{module_label}.{name}' is a numpy abstract scalar type, "
                    f"which isn't available in the numpy-free flopscope client. "
                    f"Inspect 'array.dtype' (a string) instead of using "
                    f"isinstance against scalar types."
                )
            raise AttributeError(f"module '{module_label}' has no attribute '{name}'")

    return __getattr__
