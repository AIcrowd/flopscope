"""`flops.Module` — a state-dict base class for participant models.

The file ever stores only named numeric arrays + an inert JSON `config()` blob;
the class itself comes from participant code. No pickle, ever. Mirrors the in-process
Module (src/flopscope/_module.py) — keep the two in sync.
"""

from __future__ import annotations

from typing import Any, TypeVar

from flopscope._io import load as _load
from flopscope._io import savez as _savez
from flopscope._remote_array import RemoteArray

_M = TypeVar("_M", bound="Module")

_ARRAY_TYPES = (RemoteArray,)


def _is_array(val: Any) -> bool:
    return isinstance(val, _ARRAY_TYPES)


def _collect(key: str, val: Any) -> dict[str, Any]:
    if _is_array(val):
        return {key: val}
    if isinstance(val, Module):
        return val.state_dict(prefix=f"{key}.")
    if isinstance(val, (list, tuple)):
        out: dict[str, Any] = {}
        for i, item in enumerate(val):
            out.update(_collect(f"{key}.{i}", item))
        return out
    if isinstance(val, dict):
        out = {}
        for k, item in val.items():
            out.update(_collect(f"{key}.{k}", item))
        return out
    return {}


def _rebuild(key: str, val: Any, sd: dict[str, Any]) -> Any:
    if _is_array(val):
        return sd.get(key, val)
    if isinstance(val, Module):
        sub = {k[len(key) + 1 :]: v for k, v in sd.items() if k.startswith(key + ".")}
        val.load_state_dict(sub, strict=False)
        return val
    if isinstance(val, list):
        return [_rebuild(f"{key}.{i}", item, sd) for i, item in enumerate(val)]
    if isinstance(val, tuple):
        return tuple(_rebuild(f"{key}.{i}", item, sd) for i, item in enumerate(val))
    if isinstance(val, dict):
        return {k: _rebuild(f"{key}.{k}", item, sd) for k, item in val.items()}
    return val


class Module:
    """Base class: auto-discovers array state from public attributes."""

    def state_dict(self, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            out.update(_collect(f"{prefix}{name}", val))
        return out

    def load_state_dict(self, sd: dict[str, Any], *, strict: bool = True) -> None:
        own = set(self.state_dict())
        if strict:
            provided = set(sd) - {"__meta__"}
            missing, unexpected = own - provided, provided - own
            if missing or unexpected:
                raise ValueError(
                    f"state_dict mismatch. missing={sorted(missing)} "
                    f"unexpected={sorted(unexpected)}"
                )
        for name, val in list(vars(self).items()):
            if name.startswith("_"):
                continue
            setattr(self, name, _rebuild(name, val, sd))

    def config(self) -> dict:  # type: ignore[type-arg]
        return {}

    def save(self, path: str, *, meta: dict | None = None) -> None:  # type: ignore[type-arg]
        full_meta = {**self.config(), **(meta or {})}
        _savez(path, __meta__=full_meta, **self.state_dict())

    def load(self, path: str) -> Module:
        d = _load(path)
        d.pop("__meta__", None)
        self.load_state_dict(d)
        return self

    @classmethod
    def from_file(cls: type[_M], path: str, **overrides: Any) -> _M:
        d = _load(path)
        meta = d.pop("__meta__", {})
        obj = cls(**{**meta, **overrides})  # type: ignore[call-arg]
        obj.load_state_dict(d)
        return obj
