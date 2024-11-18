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

from .typedefs import Coro, CoroFn

__all__ = ("Waterfall",)


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
        self.task: asyncio.Task[None] | None = None
        self._alive: bool = False

    def start(self) -> None:
        if self.task is not None:
            msg = "Waterfall is already running."
            raise RuntimeError(msg) from None
        self._alive = True
        self.task = asyncio.create_task(self._loop(), name="waterfall.loop")

    @overload
    def stop(self, wait: Literal[True]) -> Coro[None]: ...
    @overload
    def stop(self, wait: Literal[False]) -> None: ...
    @overload
    def stop(self, wait: bool = False) -> Coro[None] | None: ...  # noqa: FBT001, FBT002
    def stop(self, wait: bool = False) -> Coro[None] | None:  # noqa: FBT001, FBT002
        self._alive = False
        return self.queue.join() if wait else None

    def put(self, item: T) -> None:
        if not self._alive:
            msg = "Can't put something in a non-running Waterfall."
            raise RuntimeError(msg) from None
        self.queue.put_nowait(item)

    async def _loop(self) -> None:
        try:
            _tasks: set[asyncio.Task[Any]] = set()
            while self._alive:
                queue_items: MutableSequence[T] = []
                iter_start = time.monotonic()

                while (this_max_wait := (time.monotonic() - iter_start)) < self.max_wait:
                    try:
                        n = await asyncio.wait_for(self.queue.get(), this_max_wait)
                    except TimeoutError:
                        continue
                    else:
                        queue_items.append(n)
                    if len(queue_items) >= self.max_quantity:
                        break

                    if not queue_items:
                        continue

                num_items = len(queue_items)

                t = asyncio.create_task(self.callback(queue_items))
                _tasks.add(t)
                t.add_done_callback(_tasks.remove)

                for _ in range(num_items):
                    self.queue.task_done()

        finally:
            f = asyncio.create_task(self._finalize(), name="waterfall.finalizer")
            await asyncio.wait_for(f, timeout=self.max_wait_finalize)

    async def _finalize(self) -> None:
        # WARNING: Do not allow an async context switch before the gather below

        self._alive = False
        remaining_items: MutableSequence[T] = []

        while not self.queue.empty():
            try:
                ev = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                # we should never hit this, asyncio queues know their size reliably when used appropriately.
                break

            remaining_items.append(ev)

        if not remaining_items:
            return

        num_remaining = len(remaining_items)

        pending_futures: list[asyncio.Task[Any]] = []

        for chunk in (remaining_items[p : p + self.max_quantity] for p in range(0, num_remaining, self.max_quantity)):
            fut = asyncio.create_task(self.callback(chunk))
            pending_futures.append(fut)

        gathered = asyncio.gather(*pending_futures)

        try:
            await asyncio.wait_for(gathered, timeout=self.max_wait_finalize)
        except TimeoutError:
            for task in pending_futures:
                task.cancel()

        for _ in range(num_remaining):
            self.queue.task_done()
