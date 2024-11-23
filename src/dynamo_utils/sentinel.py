"""Implementation of PEP 661: Sentinel values.

See: https://www.python.org/dev/peps/pep-0661

Usage:
    >>> import sentinel
    >>>
    >>> MISSING = sentinel.Sentinel("MISSING")
"""

from __future__ import annotations

import inspect
import sys as _sys
from threading import Lock as _Lock
from typing import TYPE_CHECKING, Final, Self, cast

if TYPE_CHECKING:
    from types import UnionType


class Sentinel:
    """A unique sentinel value that maintains identity across copies and pickle operations."""

    _module_name: str
    _name: str
    _repr: str
    _truthiness: bool

    def __new__(
        cls,
        name: str,
        /,
        truthiness: bool | None = None,
        repr: str | None = None,  # noqa: A002
        module_name: str | None = None,
    ) -> Self:
        name = str(name)
        repr = f"<{name.split('.')[-1]}>" if repr is None else str(repr)  # noqa: A001
        truthiness = False if truthiness is None else truthiness
        module_name = _get_module_name() if module_name is None else str(module_name)

        registry_key = _sys.intern(f"{cls.__module__}-{cls.__qualname__}-{module_name}-{name}")

        if (existing := _registry.get(registry_key, None)) is not None:
            return cast(Self, existing)
        instance_type = type(
            name,
            (cls,),
            {
                "_name": name,
                "_repr": repr,
                "_module_name": module_name,
                "_truthiness": truthiness,
            },
        )

        instance: Self = super().__new__(instance_type)
        instance.__class__ = instance_type

        with _lock:
            return cast(Self, _registry.setdefault(registry_key, instance))

    def __repr__(self) -> str:
        return self._repr

    def __reduce__(self) -> tuple[type[Sentinel], tuple[str, bool, str, str]]:
        """Support for pickle operations."""
        return (Sentinel, (self._name, self._truthiness, self._repr, self._module_name))

    def __bool__(self) -> bool:
        """Falsy by default."""
        return self._truthiness

    def __or__(self, other: type) -> UnionType:
        return other | type(self)

    def __ror__(self, other: type) -> UnionType:
        return self.__or__(other)

    def __instancecheck__(self, instance: object) -> bool:
        """Support for isinstance(x, sentinel)."""
        return instance is self

    def __subclasscheck__(self, subclass: type) -> bool:
        """Support for issubclass(x, sentinel)."""
        return subclass is type(self)


# Module-level registry and lock
_registry: Final[dict[str, Sentinel]] = {}
_lock: Final = _Lock()


def _get_module_name() -> str:
    frame = inspect.currentframe()
    if frame is None:
        msg = "Could not determine caller frame"
        raise RuntimeError(msg)

    caller_frame = frame.f_back
    if caller_frame is None:
        msg = "Could not determine caller module"
        raise RuntimeError(msg)

    try:
        return str(caller_frame.f_globals["__name__"])
    finally:
        del frame
        del caller_frame
