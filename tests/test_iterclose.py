from collections.abc import Sequence

import pytest
from dynamo_utils.iterclose import (
    aiterclose,
    apreserve,
    iterclose,
    preserve,
    process_async_iterable,
)


def test_preserve() -> None:
    class CustomIterator:
        def __init__(self, items: Sequence[int]) -> None:
            self.items = items
            self.index = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.items):
                raise StopIteration
            value = self.items[self.index]
            self.index += 1
            return value

        def __iterclose__(self):
            self.closed = True

    iterator = CustomIterator([1, 2, 3])
    with preserve(iterator) as preserved:
        assert list(preserved) == [1, 2, 3]
    assert not iterator.closed


@pytest.mark.asyncio
async def test_apreserve() -> None:
    class CustomAsyncIterator:
        def __init__(self, items: Sequence[int]) -> None:
            self.items = items
            self.index = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            value = self.items[self.index]
            self.index += 1
            return value

        async def __aiterclose__(self):
            self.closed = True

    iterator = CustomAsyncIterator([1, 2, 3])
    async with apreserve(iterator) as preserved:
        result = [item async for item in preserved]
        assert result == [1, 2, 3]
    assert not iterator.closed


def test_iterclose() -> None:
    class ClosableIterator:
        def __init__(self):
            self.close_called = False

        def __iterclose__(self):
            self.close_called = True

    iterator = ClosableIterator()
    iterclose(iterator)  # type: ignore[reportArgumentType]
    assert iterator.close_called


@pytest.mark.asyncio
async def test_aiterclose() -> None:
    class AsyncClosableIterator:
        def __init__(self):
            self.close_called = False

        async def __aiterclose__(self):
            self.close_called = True

    iterator = AsyncClosableIterator()
    await aiterclose(iterator)  # type: ignore[reportArgumentType]
    assert iterator.close_called


@pytest.mark.asyncio
async def test_process_async_iterable() -> None:
    class CustomAsyncIterable:
        def __init__(self, items: Sequence[int]) -> None:
            self.items = items
            self.index = 0
            self.close_called = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            value = self.items[self.index]
            self.index += 1
            return value

        async def __aiterclose__(self):
            self.close_called = True

    iterable = CustomAsyncIterable([1, 2, 3])
    result = await process_async_iterable(iterable)
    assert result == [1, 2, 3]
    assert iterable.close_called
