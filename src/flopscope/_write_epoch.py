"""Buffer write epochs -- the mechanism that voids a symmetry tag when its data changes.

A symmetry tag is a billing claim about buffer contents. It is validated once,
by :func:`flopscope.as_symmetric` or inferred from shape on a constant fill,
and from then on the cost model trusts it and never re-reads the data. Any
write into the buffer therefore invalidates the claim -- including a write made
through an *untagged* alias, because ``as_symmetric`` returns a view and
``asarray``/``ravel`` hand back untagged aliases of a tagged buffer.

The aliases of a buffer cannot be enumerated from the buffer, so tags are not
cleared eagerly on write. Instead each tag records the buffer's write count at
the moment it was stamped, and reads void the tag when the counts diverge.
Every alias shares one counter because they share one ``.base`` chain root,
which is what lets a write through an untagged alias void a tag it cannot see.

Counters exist only for buffers that have actually been written and are dropped
when the buffer dies, so the table stays proportional to live written buffers
rather than to all arrays.
"""

from __future__ import annotations

import weakref

import numpy as _np

_ExtentKey = tuple[int, int]

_EPOCHS: dict[_ExtentKey, int] = {}
_ROOTS: dict[_ExtentKey, weakref.ref] = {}


def buffer_root(arr):
    """Return the array that owns ``arr``'s memory, following the view chain.

    The walk continues through non-array links and keeps the last ndarray it
    saw. Some views interpose a non-array object -- ``as_strided`` inserts a
    private shim -- and stopping there would treat the view as its own root, so
    a write through it would not reach the tags on the real buffer. Keeping the
    last ndarray also means the root is always weak-referenceable, even when the
    chain bottoms out in an exporter such as ``bytes``.
    """
    root = arr
    base = getattr(arr, "base", None)
    while base is not None:
        if isinstance(base, _np.ndarray):
            root = base
        base = getattr(base, "base", None)
    return root


def _extent_key(root) -> _ExtentKey:
    """Identify the memory a root array covers, not the array object itself.

    Arrays built independently over one exporter -- two ``frombuffer`` calls on
    the same ``bytearray``, say -- each end their view chain at a different
    ndarray, so object identity would give them separate counters and a write
    through one would not reach tags on the other. Their address and extent are
    shared, which is the property that actually matters here.
    """
    try:
        return (root.__array_interface__["data"][0], root.nbytes)
    except (AttributeError, TypeError, KeyError):
        return (id(root), -1)


def epoch_of(arr) -> int:
    """Number of recorded writes to ``arr``'s buffer. 0 if never written."""
    return _EPOCHS.get(_extent_key(buffer_root(arr)), 0)


def note_write(target) -> None:
    """Record that ``target``'s buffer was written, voiding tags that observe it.

    Accepts the shapes an ``out=`` argument can take, including the tuple form
    used by multi-output ufuncs, and ignores non-arrays.
    """
    if isinstance(target, (tuple, list)):
        for item in target:
            note_write(item)
        return
    if not isinstance(target, _np.ndarray):
        return
    root = buffer_root(target)
    key = _extent_key(root)
    _EPOCHS[key] = _EPOCHS.get(key, 0) + 1
    if key not in _ROOTS:
        # Drop the count once the buffer we first saw at this extent goes away,
        # so the table tracks live buffers. Losing a count can only make a
        # stamped epoch stop matching, which voids a tag rather than reviving
        # one -- the safe direction if the address is later reused.
        def _drop(_dead, key=key):
            _EPOCHS.pop(key, None)
            _ROOTS.pop(key, None)

        try:
            _ROOTS[key] = weakref.ref(root, _drop)
        except TypeError:  # pragma: no cover - ndarrays are weak-referenceable
            pass
