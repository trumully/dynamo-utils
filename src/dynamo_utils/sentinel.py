"""Implementation of PEP 661: Sentinel values.

See: https://www.python.org/dev/peps/pep-0661

Usage:
    >>> import sentinel
    >>>
    >>> # Assign to constant (recommended)
    >>> MISSING = sentinel.sentinel("MISSING")
    >>>
    >>> # Or inherit from Sentinel
    >>> class MySentinel(sentinel.Sentinel):
    ...     pass
"""

from __future__ import annotations

import inspect
import threading
from typing import Final


class Sentinel:
    """A unique sentinel value that maintains identity across copies and pickle operations."""

    def __init__(self, module_name: str, instance_name: str, *, repr: str | None = None) -> None:  # noqa: A002
        """Initialize a sentinel value.

        Args:
            module_name: The module where this sentinel is defined
            instance_name: The name of this sentinel instance
            repr: Optional custom string representation
        """
        self._module_name = module_name
        self._instance_name = instance_name
        self._repr = repr or f"<{instance_name}>"
        self._qualified_name = f"{module_name}.{instance_name}"

    def __new__(cls, module_name: str, instance_name: str, *, repr: str | None = None) -> Sentinel:  # noqa: A002, ARG003,PYI034
        qualified_name = f"{module_name}.{instance_name}"

        with _registry_lock:
            if (existing := _registry.get(qualified_name)) is not None:
                return existing

            instance = super().__new__(cls)
            _registry[qualified_name] = instance
            return instance

    def __repr__(self) -> str:
        return self._repr

    def __reduce__(self) -> tuple[type[Sentinel], tuple[str, str], dict[str, str | None]]:
        """Support for pickle operations."""
        return (self.__class__, (self._module_name, self._instance_name), {"repr": self._repr})

    def __bool__(self) -> bool:
        """Falsy by default."""
        return False


# Module-level registry and lock
_registry: Final[dict[str, Sentinel]] = {}
_registry_lock: Final = threading.Lock()


def sentinel(name: str, *, repr: str | None = None) -> Sentinel:  # noqa: A002
    """Create a new sentinel value.

    Args:
        name: Name of the sentinel
        repr: Optional custom string representation

    Returns:
        A unique sentinel value

    Raises:
        RuntimeError: If the caller frame cannot be determined
    """
    frame = inspect.currentframe()
    if frame is None:
        msg = "Could not determine caller frame"
        raise RuntimeError(msg)

    caller_frame = frame.f_back
    if caller_frame is None:
        msg = "Could not determine caller module"
        raise RuntimeError(msg)

    try:
        module_name = caller_frame.f_globals["__name__"]
    finally:
        # Explicitly delete references to frames to help garbage collection
        del frame
        del caller_frame

    return Sentinel(module_name, name, repr=repr)
