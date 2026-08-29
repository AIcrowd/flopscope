"""Canonical-copy: make an accepted symmetry claim exactly true before it is tagged.

``as_symmetric`` validates a symmetry claim with ``np.allclose``, which accepts
"close enough" data. The cost model then treats every position in a symmetry
orbit as a redundant degree of freedom and never re-reads the buffer. Those two
facts disagree: a caller can scale an asymmetric tensor down until its orbit
differences fall under ``atol``, collect the tag, and scale back up through an
ordinary pointwise op -- recovering independent values in positions the cost
model has already priced as redundant.

This module closes that gap at the trust boundary. Once tolerant validation
accepts a buffer, one representative per orbit is copied over the whole orbit.
The hidden values are destroyed before the tag is minted, which is what makes
the downstream symmetry discount honest without re-checking anything.

The scope is worth stating exactly, because it is narrower than "a tag always
means exact invariance". What this module guarantees is a property of the
*ingress* points -- ``as_symmetric`` and the public ``SymmetricTensor``
constructor, the two places a caller hands over a buffer the library has never
inspected. Symmetry that propagates algebraically through later operations is
trusted on the mathematics, not re-established here, and float arithmetic makes
that a genuinely weaker claim: the Reynolds projection sums each orbit in a
fixed element order, so its own output is typically invariant only to about an
ulp, and a symmetric matmul is the same. Those tags are sound for accounting --
the residue is rounding, not information a caller placed there -- but code that
needs a buffer to be invariant *to the bit* must either come through an ingress
point or ask :func:`is_exactly_invariant`.

The orbit map depends only on ``(shape, axes, generator action)`` -- never on
buffer contents -- so it is built once per distinct action and cached. Building
it walks the *generators*, not the group elements: enumerating ``|G|`` would
make ``as_symmetric`` cost as much as the Reynolds projection it deliberately
is not.
"""

from __future__ import annotations

import functools

import numpy as np

from flopscope._perm_group import SymmetryGroup


def _resolved_axes(group: SymmetryGroup) -> tuple[int, ...]:
    """Tensor axes the group acts on, applying the same fallback as validation."""
    axes = group.axes
    return tuple(axes) if axes is not None else tuple(range(group.degree))


def _generator_fingerprint(group: SymmetryGroup) -> tuple:
    """Hashable identity of the group's ACTION, without enumerating the group.

    ``SymmetryGroup.__hash__`` canonicalizes through ``elements()``, which runs
    Dimino and can blow the enumeration budget -- exactly the cost this module
    exists to avoid. The generator literals pin the action just as tightly for
    caching purposes; two spellings of one group merely get two identical
    cache entries.
    """
    return (
        _resolved_axes(group),
        group.degree,
        tuple(tuple(gen.array_form) for gen in group.generators if not gen.is_identity),
    )


def _generator_images(shape: tuple[int, ...], axes, degree, gen_forms):
    """Flat-index image of each generator, one vectorized pass per generator."""
    ndim = len(shape)
    flat = np.arange(int(np.prod(shape)), dtype=np.intp).reshape(shape)
    images = []
    for form in gen_forms:
        perm = list(range(ndim))
        for i in range(degree):
            perm[axes[i]] = axes[form[i]]
        images.append(np.transpose(flat, perm).ravel())
    return images


def _build_canonical_map(shape: tuple[int, ...], fingerprint: tuple) -> np.ndarray:
    """``map[i]`` = smallest C-order flat index in ``i``'s orbit.

    Min-label propagation over the generator action: each round pushes every
    position's label down to the smallest label reachable in one generator
    step, then pointer-jumps so labels reach orbit minima in log-many rounds.
    Cost is ``O(N * r)`` per round with no Python-level loop over elements,
    versus ``O(N * |G|)`` for an element enumeration.
    """
    axes, degree, gen_forms = fingerprint
    n = int(np.prod(shape))
    labels = np.arange(n, dtype=np.intp)
    if not gen_forms:
        labels.flags.writeable = False
        return labels

    images = _generator_images(shape, axes, degree, gen_forms)
    while True:
        previous = labels
        for image in images:
            labels = np.minimum(labels, labels[image])
        labels = labels[labels]  # pointer jumping
        if np.array_equal(labels, previous):
            break

    # Cached and shared across calls: never let a caller mutate it.
    labels.flags.writeable = False
    return labels


#: Ceiling on the memory the orbit-map cache may hold, in bytes.
#: An entry is one index per tensor element, so unlike flopscope's other LRUs
#: -- whose entries are small cost records -- entries here scale with the
#: tensors they describe: 8 MB for a 1024x1024 map, 34 MB at 2048x2048. A
#: plain entry count would therefore bound the number of maps while leaving
#: the footprint unbounded, and every distinct shape a caller asks about mints
#: a new one. 256 MB keeps the working set of a realistic estimator resident
#: while refusing to grow without limit.
_CANONICAL_MAP_CACHE_BYTES = 256 * 1024 * 1024


class _OrbitMapCache:
    """LRU over orbit maps, bounded by total bytes rather than entry count.

    Mirrors enough of ``functools.lru_cache``'s surface (``cache_info``,
    ``cache_clear``) to be used and inspected the same way.
    """

    __slots__ = ("_entries", "_max_bytes", "_bytes", "_hits", "_misses")

    def __init__(self, max_bytes: int) -> None:
        self._entries: dict[tuple, np.ndarray] = {}
        self._max_bytes = max_bytes
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, shape: tuple[int, ...], fingerprint: tuple) -> np.ndarray:
        key = (shape, fingerprint)
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached  # refresh recency
            self._hits += 1
            return cached

        self._misses += 1
        mapping = _build_canonical_map(shape, fingerprint)
        # A single map larger than the whole budget is served but not kept,
        # so one outsized request cannot evict everything and still not fit.
        if mapping.nbytes <= self._max_bytes:
            self._entries[key] = mapping
            self._bytes += mapping.nbytes
            # dicts iterate in insertion order and `get` reinserts on a hit,
            # so the first key is the least recently used.
            while self._bytes > self._max_bytes:
                evicted = self._entries.pop(next(iter(self._entries)))
                self._bytes -= evicted.nbytes
        return mapping

    def cache_info(self):
        return functools._CacheInfo(  # type: ignore[attr-defined]
            self._hits, self._misses, self._max_bytes, len(self._entries)
        )

    def cache_clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    @property
    def nbytes(self) -> int:
        return self._bytes


#: Process-wide orbit-map cache. Named for the lookup it performs so that
#: ``cache_info()``/``cache_clear()`` read the same as flopscope's other LRUs.
_canonical_map_cached = _OrbitMapCache(_CANONICAL_MAP_CACHE_BYTES)


def canonical_map(shape: tuple[int, ...], group: SymmetryGroup) -> np.ndarray:
    """Cached orbit map for ``(shape, group action)``.

    Returns a view rather than the cached array itself. NumPy lets a caller
    re-enable the writeable flag on an array that owns its data, but not on
    one whose base is read-only, and a map mutated in place would silently
    mis-canonicalize every later call for the same shape and group.
    """
    return _canonical_map_cached.get(tuple(shape), _generator_fingerprint(group)).view()


def is_exactly_invariant(array: np.ndarray, group: SymmetryGroup) -> bool:
    """Whether every orbit already holds one repeated value, to the bit.

    Checking generators is enough: they generate the group, so a buffer fixed
    by each generator is fixed by every element. This is the tolerance-free
    twin of the ``allclose`` check validation runs -- equality, not closeness,
    is precisely the property the tag is read as asserting.

    Answering this with ``==`` alone would be too generous by exactly one
    value: ``-0.0 == 0.0`` is true, yet the two differ in a bit that
    ``copysign`` reads straight back out. A sign bit sitting in a position the
    cost model prices as redundant is information like any other, so zeros
    that disagree in sign count as a difference here and send the buffer down
    the copying path.
    """
    array = np.asarray(array)
    axes = _resolved_axes(group)
    ndim = array.ndim
    signed = array.dtype.kind in "fc"
    for gen in group.generators:
        if gen.is_identity:
            continue
        perm = list(range(ndim))
        for i in range(group.degree):
            perm[axes[i]] = axes[gen.array_form[i]]
        transposed = array.transpose(perm)
        if not np.array_equal(array, transposed):
            return False
        if signed:
            if not np.array_equal(
                np.signbit(array.real), np.signbit(transposed.real)
            ) or (
                array.dtype.kind == "c"
                and not np.array_equal(
                    np.signbit(array.imag), np.signbit(transposed.imag)
                )
            ):
                return False
    return True


def canonicalize(array: np.ndarray, group: SymmetryGroup) -> np.ndarray:
    """Return data whose orbits are exactly constant, copying only if needed.

    Data that is already exactly invariant is returned untouched, so the
    common case -- a genuinely symmetric buffer -- keeps ``as_symmetric``'s
    zero-copy view semantics, including its use as an ``out=`` destination
    with a caller-chosen memory layout. Only a buffer that merely passed the
    tolerant check gets rewritten, which is exactly the case where the tag
    would otherwise certify more than the data supports.
    """
    array = np.asarray(array)
    if array.size == 0 or is_exactly_invariant(array, group):
        return array
    return canonical_copy(array, group)


def canonical_copy(array: np.ndarray, group: SymmetryGroup) -> np.ndarray:
    """Return a fresh array whose orbits each hold one representative value.

    The representative is the orbit's lexicographically smallest tensor index.
    Advanced indexing gathers rather than computes, so the dtype survives
    exactly -- unlike the Reynolds projection, which must upcast to average --
    and the result is a new buffer, so a caller's array is never mutated and
    the tagged data can no longer be reached through the caller's alias.
    """
    array = np.asarray(array)
    if array.size == 0:
        return array.copy()
    mapping = canonical_map(array.shape, group)
    return array.reshape(-1)[mapping].reshape(array.shape)


def clear_canonical_map_cache() -> None:
    """Drop cached orbit maps (used by cache-management hooks and tests)."""
    _canonical_map_cached.cache_clear()
