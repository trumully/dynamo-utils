"""A utility for batching similar operations into a single execution.

This module provides a Waterfall class that collects items over time and processes them
in batches when either a time limit or quantity threshold is reached.

Example:
    >>> async def process_batch(items: list[str]) -> None:
    ...     print(f"Processing batch of {len(items)} items")
    ...     await some_api_call(items)
    ...
    >>> # Create waterfall with 5 second max wait, 100 items max per batch
    >>> waterfall = Waterfall(max_wait=5.0, max_quantity=100, async_callback=process_batch)
    >>> waterfall.start()
    >>>
    >>> # Add items to be processed
    >>> await waterfall.put("item1")
    >>> await waterfall.put("item2")
    >>> # Items will be collected and processed in batches
    >>>
    >>> # Stop waterfall and wait for pending items
    >>> await waterfall.stop(wait=True)
"""

import asyncio
import time
from collections.abc import MutableSequence, Sequence
from typing import Any, Literal, overload

from .sentinel import Sentinel
from .typedefs import Coro, CoroFn

__all__ = ("Waterfall",)


MISSING = Sentinel("MISSING")


class Waterfall[T]:
    """A utility class for batching similar operations into a single execution."""

    def __init__(
        self,
        max_wait: float,
        max_quantity: int,
        async_callback: CoroFn[[Sequence[T]], Any],
        *,
        max_wait_finalize: int = 3,
    ) -> None:
        self.queue: asyncio.Queue[T] = asyncio.Queue()
        self.max_wait: float = max_wait
        self.max_quantity: int = max_quantity
        self.max_wait_finalize: int = max_wait_finalize
        self.callback: CoroFn[[Sequence[T]], Any] = async_callback
        self.task: asyncio.Task[None] | MISSING = MISSING  # type: ignore[valid-type]

    def start(self) -> None:
        if self.task is not MISSING:
            msg = "Waterfall is already running."
            raise RuntimeError(msg) from None
        self.task = asyncio.create_task(self._loop(), name="waterfall.loop")

    @overload
    def stop(self, *, wait: Literal[True]) -> Coro[None]: ...
    @overload
    def stop(self, *, wait: Literal[False]) -> None: ...
    def stop(self, *, wait: bool = False) -> Coro[None] | None:
        self.queue.shutdown()
        return self.queue.join() if wait else None

    def put(self, item: T) -> None:
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueShutDown as e:
            msg = "Can't put something in a shut down Waterfall."
            raise RuntimeError(msg) from e

    async def _process_batch(self, queue_items: MutableSequence[T], tasks: set[asyncio.Task[Any]]) -> None:
        t = asyncio.create_task(self.callback(queue_items))
        tasks.add(t)
        t.add_done_callback(tasks.remove)
        for _ in range(len(queue_items)):
            self.queue.task_done()

    async def _loop(self) -> None:
        _tasks: set[asyncio.Task[Any]] = set()
        while True:
            queue_items: MutableSequence[T] = []
            iter_start = time.monotonic()

            while (time.monotonic() - iter_start) < self.max_wait:
                try:
                    remaining_time = self.max_wait - (time.monotonic() - iter_start)
                    n = await asyncio.wait_for(self.queue.get(), remaining_time)
                except asyncio.QueueShutDown:
                    # Queue is shut down, process remaining items and exit
                    if queue_items:
                        await self._process_batch(queue_items, _tasks)
                    return
                except TimeoutError:
                    break
                else:
                    queue_items.append(n)
                    if len(queue_items) >= self.max_quantity:
                        break

            if queue_items:
                await self._process_batch(queue_items, _tasks)
