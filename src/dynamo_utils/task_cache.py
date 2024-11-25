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
# mypy: disable-error-code="valid-type"

from __future__ import annotations

import asyncio
from collections.abc import Hashable
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Any, Protocol, overload

from .sentinel import Sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dynamo_utils.typedefs import TaskCoroFn


__all__ = ("LRU", "lru_task_cache", "task_cache")


MISSING = Sentinel("MISSING")


class HashedSequence(list[Hashable]):
    __slots__ = ("hashvalue",)

    def __init__(self, tup: tuple[Hashable, ...], hash: Callable[[object], int] = hash) -> None:  # noqa: A002
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self) -> int:  # type: ignore[override]
        return self.hashvalue

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self[:] == other[:]  # type: ignore[index]

    @classmethod
    def from_call(
        cls: type[HashedSequence],
        args: tuple[Hashable, ...],
        kwargs: Mapping[str, Hashable],
        fast_types: tuple[type, type] = (int, str),
        _sentinel: Hashable = MISSING,
    ) -> HashedSequence | int | str:
        key: tuple[Any, ...] = args if not kwargs else (*args, _sentinel, *kwargs.items())
        first: int | str = key[0]
        return first if len(key) == 1 and type(first) in fast_types else cls(key)


class CacheableTask[**P, T](Protocol):
    """A cached coroutine function."""

    __slots__ = ()

    @property
    def __wrapped__(self) -> TaskCoroFn[P, T]: ...

    def __get__[S: object](
        self, instance: S, owner: type[S] | MISSING = MISSING
    ) -> CacheableTask[P, T] | BoundCacheableTask[S, P, T]:
        return self if instance is MISSING else BoundCacheableTask(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]: ...
    def cache_clear(self) -> None: ...
    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class CachedTask[**P, T](CacheableTask[P, T]):
    """A cached coroutine function."""

    __slots__ = ("__cache", "__dict__", "__ttl", "__weakref__", "__wrapped__")

    __wrapped__: TaskCoroFn[P, T]

    def __init__(self, call: TaskCoroFn[P, T], /, ttl: float | MISSING = MISSING) -> None:
        self.__cache: dict[HashedSequence | int | str, asyncio.Task[T]] = {}
        self.__wrapped__ = call
        self.__ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.__cache[key]
        except KeyError:
            self.__cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            if self.__ttl is not MISSING:
                call_after_ttl = partial(asyncio.get_event_loop().call_later, self.__ttl, self.__cache.pop, key)
                task.add_done_callback(call_after_ttl)
        return task

    def cache_clear(self) -> None:
        self.__cache.clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.__cache.pop(key, None)


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

        self.cache[key] = self.cache.pop(key)
        return self.cache[key]

    def __getitem__(self, key: K, /) -> V:
        self.cache[key] = self.cache.pop(key)
        return self.cache[key]

    def __setitem__(self, key: K, value: V, /) -> None:
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.pop(next(iter(self.cache)))

    def remove(self, key: K, /) -> None:
        self.cache.pop(key, MISSING)

    def clear(self) -> None:
        self.cache.clear()


def _lru_evict(
    ttl: float,
    cache: LRU[HashedSequence | int | str, asyncio.Task[Any]],
    key: HashedSequence | int | str,
    _task: MISSING = MISSING,
) -> None:
    asyncio.get_event_loop().call_later(ttl, cache.remove, key)


class LRUCachedTask[**P, T](CacheableTask[P, T]):
    """A cached coroutine function with a Least Recently Used cache with support for TTL."""

    __slots__ = ("__cache", "__dict__", "__ttl", "__weakref__", "__wrapped__")

    __wrapped__: TaskCoroFn[P, T]

    def __init__(
        self,
        call: TaskCoroFn[P, T],
        /,
        maxsize: int,
        ttl: float | MISSING = MISSING,
    ) -> None:
        self.__cache: LRU[HashedSequence | int | str, asyncio.Task[T]] = LRU(maxsize)
        self.__wrapped__ = call
        self.__ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.__cache[key]
        except KeyError:
            self.__cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            if self.__ttl is not MISSING:
                call_after_ttl = partial(_lru_evict, self.__ttl, self.__cache, key)
                task.add_done_callback(call_after_ttl)
        return task

    def cache_clear(self) -> None:
        self.__cache.clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.__cache.remove(key)


class BoundCacheableTask[S, **P, T]:
    """A bound cached coroutine function."""

    __slots__ = ("__self__", "__weakref__", "_task")

    def __init__(self, task: CacheableTask[P, T], __self__: object):
        self._task = task
        self.__self__ = __self__

    @property
    def __wrapped__(self) -> TaskCoroFn[P, T]:
        return self._task.__wrapped__

    @property
    def __func__(self) -> CacheableTask[P, T]:
        return self._task

    @property
    def __annotations__(self) -> dict[str, Any]:
        return self._task.__annotations__

    @property
    def __doc__(self) -> str:
        return self._task.__doc__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        return self._task(*(self.__self__, *args), **kwargs)  # type: ignore[arg-type]

    def __get__[S2: object](
        self: BoundCacheableTask[S, P, T],
        instance: S2,
        owner: type | MISSING = MISSING,
    ) -> BoundCacheableTask[S2, P, T]:
        return BoundCacheableTask(self._task, instance)

    def cache_clear(self) -> None:
        self._task.cache_clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._task.cache_discard(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<bound cached task {getattr(self.__wrapped__, '__qualname__', '?')} of {self.__self__!r}>"


@overload
def task_cache[**P, T](*, ttl: float | MISSING = MISSING) -> Callable[[TaskCoroFn[P, T]], CacheableTask[P, T]]: ...
@overload
def task_cache[**P, T](coro: TaskCoroFn[P, T], /) -> CacheableTask[P, T]: ...
def task_cache[**P, T](
    ttl: TaskCoroFn[P, T] | float | MISSING = MISSING,
) -> Callable[[TaskCoroFn[P, T]], CacheableTask[P, T]] | CacheableTask[P, T]:
    """Decorator to wrap a coroutine function in a cache with support for TTL."""
    if isinstance(ttl, float):
        ttl = max(0, ttl)
    elif callable(ttl):
        fast_wrapper = CachedTask(ttl)
        update_wrapper(fast_wrapper, ttl)
        return fast_wrapper

    def decorator(coro: TaskCoroFn[P, T]) -> CacheableTask[P, T]:
        wrapper = CachedTask(coro, ttl)
        update_wrapper(wrapper, coro)
        return wrapper

    return decorator


@overload
def lru_task_cache[**P, T](
    *, ttl: float | MISSING = MISSING, maxsize: int = 1024
) -> Callable[[TaskCoroFn[P, T]], CacheableTask[P, T]]: ...
@overload
def lru_task_cache[**P, T](coro: TaskCoroFn[P, T], /) -> CacheableTask[P, T]: ...
def lru_task_cache[**P, T](
    ttl: TaskCoroFn[P, T] | float | MISSING = MISSING, maxsize: int = 1024
) -> CacheableTask[P, T] | Callable[[TaskCoroFn[P, T]], CacheableTask[P, T]]:
    """Decorator to wrap a coroutine function in a Least Recently Used cache with support for TTL."""
    if isinstance(ttl, float):
        ttl = max(0, ttl)
    elif callable(ttl):
        fast_wrapper = LRUCachedTask(ttl, maxsize)
        update_wrapper(fast_wrapper, ttl)
        return fast_wrapper

    def decorator(coro: TaskCoroFn[P, T]) -> CacheableTask[P, T]:
        wrapper = LRUCachedTask(coro, maxsize, ttl)
        update_wrapper(wrapper, coro)
        return wrapper

    return decorator
