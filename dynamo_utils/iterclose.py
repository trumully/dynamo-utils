"""Implementation of PEP 533: Deterministic cleanup for iterators.

See: https://peps.python.org/pep-0533
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class IterCloseProtocol[T](Protocol):
    """Protocol for iterators that support cleanup."""

    def __iter__(self) -> Iterator[T]: ...
    def __next__(self) -> T: ...
    def __iterclose__(self) -> None: ...


class AsyncIterCloseProtocol[T](Protocol):
    """Protocol for async iterators that support cleanup."""

    def __aiter__(self) -> AsyncIterator[T]: ...
    async def __anext__(self) -> T: ...
    async def __aiterclose__(self) -> None: ...


@contextlib.contextmanager
def preserve[T](iterator: Iterator[T]) -> Iterator[Iterator[T]]:
    """Preserve an iterator's state by preventing __iterclose__ from being called."""

    class PreservedIterator:
        def __init__(self, it: Iterator[T]) -> None:
            self._iterator = it

        def __iter__(self) -> Iterator[T]:
            return self

        def __next__(self) -> T:
            return next(self._iterator)

        def __iterclose__(self) -> None:
            pass  # Explicitly do nothing to preserve the iterator

    yield PreservedIterator(iterator)


@contextlib.asynccontextmanager
async def apreserve[T](iterator: AsyncIterator[T]) -> AsyncIterator[AsyncIterator[T]]:
    """Preserve an async iterator's state by preventing __aiterclose__ from being called."""

    class PreservedAsyncIterator:
        def __init__(self, it: AsyncIterator[T]) -> None:
            self._iterator = it

        def __aiter__(self) -> AsyncIterator[T]:
            return self

        async def __anext__(self) -> T:
            return await self._iterator.__anext__()

        async def __aiterclose__(self) -> None:
            pass  # Explicitly do nothing to preserve the iterator

    yield PreservedAsyncIterator(iterator)


@runtime_checkable
class IterCloseable(Protocol):
    def __iterclose__(self) -> None: ...


@runtime_checkable
class AsyncIterCloseable(Protocol):
    async def __aiterclose__(self) -> None: ...


def iterclose(iterator: Iterator[Any]) -> None:
    """Explicitly close an iterator if it supports __iterclose__."""
    if isinstance(iterator, IterCloseable):
        iterator.__iterclose__()


async def aiterclose(iterator: AsyncIterator[Any]) -> None:
    """Explicitly close an async iterator if it supports __aiterclose__."""
    if isinstance(iterator, AsyncIterCloseable):
        await iterator.__aiterclose__()


async def process_async_iterable[T](iterable: AsyncIterator[T]) -> list[T]:
    """Safely process an async iterable with proper cleanup.

    Args:
        iterable: The async iterable to process

    Returns:
        A sequence containing all items from the iterable
    """
    iterator = aiter(iterable)
    try:
        return [item async for item in iterator]
    finally:
        await aiterclose(iterator)
