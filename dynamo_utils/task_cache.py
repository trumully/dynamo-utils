"""Task cache with support for TTL and LRU.

Usage:
    >>> from dynamo_utils.task_cache import task_cache, lru_task_cache
    >>>
    >>> # Unbound and is essentially a memoization decorator
    >>> @task_cache
    >>> async def cached_func(x: int) -> int:
    ...     return x * 2
    >>>
    >>> # Bound by maxsize with LRU and TTL eviction
    >>> @lru_task_cache(maxsize=2, ttl=0.1)
    >>> async def cached_func(x: int) -> int:
    ...     return x * 2
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, cast, overload

from dynamo_utils.sentinel import Sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dynamo_utils.typedefs import Coro, TaskCoroFunc


__all__ = ("LRU", "lru_task_cache", "task_cache", "tracked_lru_task_cache", "tracked_task_cache")


class HashedSequence(list[Hashable]):
    __slots__ = ("hashvalue",)

    def __init__(
        self,
        tup: tuple[Hashable, ...],
        hash: Callable[[object], int] = hash,  # noqa: A002
    ) -> None:
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self) -> int:  # type: ignore[reportIncompatibleMethodOverride]
        return self.hashvalue

    def __eq__(self, other: object) -> bool:
        return self[:] == other[:] if isinstance(other, type(self)) else False

    @classmethod
    def from_call(
        cls: type[HashedSequence],
        args: tuple[Hashable, ...],
        kwargs: Mapping[str, Hashable],
        fast_types: tuple[type, type] = (int, str),
        _sentinel: Hashable = object,
    ) -> HashedSequence | int | str:
        key: tuple[Any, ...] = args if not kwargs else (*args, _sentinel, *kwargs.items())
        first: (int | str) | Any = key[0]
        return first if len(key) == 1 and type(first) in fast_types else cls(key)


MISSING = Sentinel("MISSING")


class LRU[K, V]:
    """A Least Recently Used cache."""

    def __init__(self, maxsize: int, /) -> None:
        self.cache: dict[K, V] = {}
        self.maxsize = maxsize

    @overload
    def get(self, key: K, /) -> V: ...
    @overload
    def get[T](self, key: K, default: T, /) -> V | T: ...
    def get(self, key: K, default: Any = MISSING, /) -> Any:
        if key not in self.cache:
            if default is MISSING:
                raise KeyError(key)
            return default

        return self[key]

    def __getitem__(self, key: K, /) -> V:
        return self.cache.setdefault(key, self.cache.pop(key))

    def __setitem__(self, key: K, value: V, /) -> None:
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.pop(next(iter(self.cache)))

    def remove(self, key: K, /) -> None:
        self.cache.pop(key, None)

    def clear(self) -> None:
        self.cache.clear()


class CacheStats:
    __slots__ = ("hits", "misses")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        return f"CacheStats(hits={self.hits}, misses={self.misses})"


def _lru_evict(
    ttl: float,
    cache: LRU[HashedSequence | int | str, Any],
    key: HashedSequence | int | str,
    _task: object,
) -> None:
    asyncio.get_running_loop().call_later(ttl, cache.remove, key)


_WRAP_ASSIGN = ("__module__", "__name__", "__qualname__", "__doc__")


class TaskFunc[**P, R]:
    __slots__ = ("_task",)

    def __init__(self, task: TaskCoroFunc[P, R]) -> None:
        self._task = task

    def __call__(self, *args: Any, **kwargs: Any) -> Coro[R] | asyncio.Task[R]:
        return self._task(*args, **kwargs)

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

    def __repr__(self) -> str:
        return f"<cached task {getattr(self._task, '__qualname__', '?')}>"


class TrackedTaskFunc[**P, R](TaskFunc[P, R]):
    def cache_stats(self) -> CacheStats: ...

    def __repr__(self) -> str:
        return f"<cached task {getattr(self._task, '__qualname__', '?')} {self.cache_stats()!r}>"


class BadMaxsizeArgument(RuntimeError):
    """Raised when maxsize is less than or equal to 0."""


@overload
def task_cache[**P, R](coro: TaskCoroFunc[P, R], /) -> TaskFunc[P, R]: ...
@overload
def task_cache[**P, R](ttl: float | None = None) -> Callable[[TaskCoroFunc[P, R]], TaskFunc[P, R]]: ...
def task_cache[**P, R](
    ttl: float | TaskCoroFunc[P, R] | None = None,
) -> Callable[[TaskCoroFunc[P, R]], TaskFunc[P, R]] | TaskFunc[P, R]:
    if isinstance(ttl, float):
        ttl = None if ttl <= 0 else ttl

    def wrapper(coro: TaskCoroFunc[P, R]) -> TaskFunc[P, R]:
        internal_cache: dict[HashedSequence | int | str, asyncio.Task[R]] = {}

        @wraps(coro, assigned=_WRAP_ASSIGN)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
            key = HashedSequence.from_call(args, kwargs)
            try:
                return internal_cache[key]
            except KeyError:
                internal_cache[key] = task = asyncio.ensure_future(coro(*args, **kwargs))
                if ttl is not None and not callable(ttl):
                    call_after_ttl = partial(
                        asyncio.get_running_loop().call_later,
                        ttl,
                        internal_cache.pop,
                        key,
                    )
                    task.add_done_callback(call_after_ttl)  # type: ignore[reportArgumentType]
                return task

        def cache_discard(*args: P.args, **kwargs: P.kwargs) -> None:
            key = HashedSequence.from_call(args, kwargs)
            internal_cache.pop(key, None)

        _wrapped = cast(TaskFunc[P, R], wrapped)
        _wrapped.cache_discard = cache_discard
        return _wrapped

    return wrapper(ttl) if callable(ttl) else wrapper


@overload
def lru_task_cache[**P, R](coro: TaskCoroFunc[P, R], /) -> TaskFunc[P, R]: ...
@overload
def lru_task_cache[**P, R](
    ttl: float | None = None,
    maxsize: int = 1024,
) -> Callable[[TaskCoroFunc[P, R]], TaskFunc[P, R]]: ...
def lru_task_cache[**P, R](
    ttl: float | TaskCoroFunc[P, R] | None = None,
    maxsize: int = 1024,
) -> Callable[[TaskCoroFunc[P, R]], TaskFunc[P, R]] | TaskFunc[P, R]:
    if isinstance(ttl, float):
        ttl = None if ttl <= 0 else ttl

    if maxsize <= 0:
        raise BadMaxsizeArgument

    def wrapper(coro: TaskCoroFunc[P, R]) -> TaskFunc[P, R]:
        internal_cache: LRU[HashedSequence | int | str, asyncio.Task[R]] = LRU(maxsize)

        @wraps(coro, assigned=_WRAP_ASSIGN)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
            key = HashedSequence.from_call(args, kwargs)
            try:
                return internal_cache[key]
            except KeyError:
                internal_cache[key] = task = asyncio.ensure_future(coro(*args, **kwargs))
                if ttl is not None and not callable(ttl):
                    task.add_done_callback(partial(_lru_evict, ttl, internal_cache, key))
                return task

        def cache_discard(*args: P.args, **kwargs: P.kwargs) -> None:
            key = HashedSequence.from_call(args, kwargs)
            internal_cache.remove(key)

        _wrapped = cast(TaskFunc[P, R], wrapped)
        _wrapped.cache_discard = cache_discard
        return _wrapped

    return wrapper(ttl) if callable(ttl) else wrapper


@overload
def tracked_task_cache[**P, R](coro: TaskCoroFunc[P, R], /) -> TrackedTaskFunc[P, R]: ...
@overload
def tracked_task_cache[**P, R](ttl: float | None = None) -> Callable[[TaskCoroFunc[P, R]], TrackedTaskFunc[P, R]]: ...
def tracked_task_cache[**P, R](
    ttl: float | TaskCoroFunc[P, R] | None = None,
) -> Callable[[TaskCoroFunc[P, R]], TrackedTaskFunc[P, R]] | TrackedTaskFunc[P, R]:
    if isinstance(ttl, float):
        ttl = None if ttl <= 0 else ttl

    def wrapper(coro: TaskCoroFunc[P, R]) -> TrackedTaskFunc[P, R]:
        internal_cache: dict[HashedSequence | int | str, asyncio.Task[R]] = {}
        stats = CacheStats()

        def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
            key = HashedSequence.from_call(args, kwargs)
            try:
                task = internal_cache[key]
            except KeyError:
                stats.misses += 1
                internal_cache[key] = task = asyncio.ensure_future(coro(*args, **kwargs))
                if ttl is not None and not callable(ttl):
                    call_after_ttl = partial(
                        asyncio.get_running_loop().call_later,
                        ttl,
                        internal_cache.pop,
                        key,
                    )
                    task.add_done_callback(call_after_ttl)  # type: ignore[reportArgumentType]
            else:
                stats.hits += 1
            return task

        def cache_discard(*args: P.args, **kwargs: P.kwargs) -> None:
            key = HashedSequence.from_call(args, kwargs)
            stats.misses = 0
            stats.hits = 0
            internal_cache.pop(key, None)

        def cache_stats() -> CacheStats:
            return stats

        _wrapped = cast(TrackedTaskFunc[P, R], wrapped)
        _wrapped.cache_discard = cache_discard
        _wrapped.cache_stats = cache_stats
        return _wrapped

    return wrapper(ttl) if callable(ttl) else wrapper


@overload
def tracked_lru_task_cache[**P, R](coro: TaskCoroFunc[P, R], /) -> TrackedTaskFunc[P, R]: ...
@overload
def tracked_lru_task_cache[**P, R](
    ttl: float | None = None,
    maxsize: int = 1024,
) -> Callable[[TaskCoroFunc[P, R]], TrackedTaskFunc[P, R]]: ...
def tracked_lru_task_cache[**P, R](
    ttl: float | TaskCoroFunc[P, R] | None = None,
    maxsize: int = 1024,
) -> Callable[[TaskCoroFunc[P, R]], TrackedTaskFunc[P, R]] | TrackedTaskFunc[P, R]:
    if isinstance(ttl, float):
        ttl = None if ttl <= 0 else ttl

    if maxsize <= 0:
        raise BadMaxsizeArgument

    def wrapper(coro: TaskCoroFunc[P, R]) -> TrackedTaskFunc[P, R]:
        internal_cache: LRU[HashedSequence | int | str, asyncio.Task[R]] = LRU(maxsize)
        stats = CacheStats()

        @wraps(coro, assigned=_WRAP_ASSIGN)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
            key = HashedSequence.from_call(args, kwargs)
            try:
                task = internal_cache[key]
            except KeyError:
                stats.misses += 1
                internal_cache[key] = task = asyncio.ensure_future(coro(*args, **kwargs))
                if ttl is not None and not callable(ttl):
                    task.add_done_callback(partial(_lru_evict, ttl, internal_cache, key))
            else:
                stats.hits += 1
            return task

        def cache_discard(*args: P.args, **kwargs: P.kwargs) -> None:
            key = HashedSequence.from_call(args, kwargs)
            stats.misses = 0
            stats.hits = 0
            internal_cache.remove(key)

        def cache_stats() -> CacheStats:
            return stats

        _wrapped = cast(TrackedTaskFunc[P, R], wrapped)
        _wrapped.cache_discard = cache_discard
        _wrapped.cache_stats = cache_stats
        return _wrapped

    return wrapper(ttl) if callable(ttl) else wrapper
