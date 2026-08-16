"""Ops whose numpy result is a structured (namedtuple) container, and ops whose
result is a plain tuple — expressed once, backend-agnostically.

Every case is a callable taking a numpy-like namespace, so the identical call
runs against in-process flopscope and against the client. That is what lets a
test compare the two WITHOUT writing numpy's field names down anywhere: the
in-process result supplies the expectation, and a numpy release that renames a
field (or adds a structured result) moves both sides together instead of
turning a hard-coded literal stale.

Imports nothing from flopscope or numpy, so both the in-process subprocess and
the client-side test session can load it without dragging a backend in.
"""

from __future__ import annotations

from typing import Any


def _matrix(fnp: Any) -> Any:
    # Symmetric and well-conditioned, so eigh/qr/svd/slogdet are all stable.
    return fnp.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 5.0],
        ]
    )


def _vector(fnp: Any) -> Any:
    return fnp.array([1.0, 2.0, 2.0, 3.0])


#: Ops numpy answers with a namedtuple. The container type must survive the
#: wire, so `.U` / `.Q` / `.eigenvalues` resolve remotely as they do locally.
STRUCTURED_CASES: dict[str, Any] = {
    "linalg.svd": lambda fnp: fnp.linalg.svd(_matrix(fnp)),
    "linalg.eig": lambda fnp: fnp.linalg.eig(_matrix(fnp)),
    "linalg.eigh": lambda fnp: fnp.linalg.eigh(_matrix(fnp)),
    "linalg.qr": lambda fnp: fnp.linalg.qr(_matrix(fnp)),
    "linalg.slogdet": lambda fnp: fnp.linalg.slogdet(_matrix(fnp)),
    "unique_all": lambda fnp: fnp.unique_all(_vector(fnp)),
    "unique_counts": lambda fnp: fnp.unique_counts(_vector(fnp)),
    "unique_inverse": lambda fnp: fnp.unique_inverse(_vector(fnp)),
}

#: Ops that return an ordinary tuple. These are what the multi-result form was
#: built for; they must stay ordinary tuples, with no container smuggled in.
PLAIN_CASES: dict[str, Any] = {
    "nonzero": lambda fnp: fnp.nonzero(_vector(fnp)),
    "modf": lambda fnp: fnp.modf(_vector(fnp)),
    "frexp": lambda fnp: fnp.frexp(_vector(fnp)),
}


def container_fields(result: Any) -> list[str] | None:
    """Return *result*'s namedtuple field names, or ``None`` if it has none."""
    fields = getattr(type(result), "_fields", None)
    if fields is None:
        return None
    return list(fields)


def collect_fields(fnp: Any, cases: dict[str, Any]) -> dict[str, list[str] | None]:
    """Run every case in *cases* against *fnp* and record its container fields."""
    return {name: container_fields(case(fnp)) for name, case in cases.items()}
