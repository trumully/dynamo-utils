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
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Any, Concatenate, Protocol, cast, overload

from dynamo_utils.sentinel import Sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dynamo_utils.typedefs import TaskCoroFn


__all__ = ("LRU", "lru_task_cache", "task_cache")


type CacheKey = HashedSequence | int | str


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


class Cacheable[**P, R](Protocol):
    __slots__: tuple[str, ...] = ()

    __wrapped__: TaskCoroFn[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]: ...
    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class TaskCache[**P, R](Cacheable[P, R]):
    """Base class for task caching."""

    __slots__ = ("__dict__", "__weakref__", "__wrapped__", "cache")

    __wrapped__: TaskCoroFn[P, R]

    def __init__(self, fn: TaskCoroFn[P, R]):
        self.__wrapped__ = fn
        self.cache: dict[CacheKey, asyncio.Task[R]] = {}

    def __get__[S](
        self,
        instance: S | None = None,
        owner: type[S] | None = None,
    ) -> TaskCache[P, R] | BoundTaskCache[S, P, R]:
        if instance is None:
            return self
        return BoundTaskCache[S, P, R](cast(Cacheable[Concatenate[S, P], R], self), instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            return self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            return task

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.cache.pop(key)


class BoundTaskCache[S, **P, R]:
    __slots__ = ("__self__", "__weakref__", "task")

    def __init__(self, task: Cacheable[Concatenate[S, P], R], instance: S):
        self.task = task
        self.__self__ = instance

    @property
    def __wrapped__(self) -> TaskCoroFn[Concatenate[S, P], R]:
        return self.task.__wrapped__

    @property
    def __func__(self) -> Cacheable[Concatenate[S, P], R]:
        return self.task

    def __get__[S2](
        self: BoundTaskCache[S, P, R],
        instance: S2,
        owner: type[S2] | None = None,
    ) -> BoundTaskCache[S2, P, R]:
        task = cast(Cacheable[Concatenate[S2, P], R], self.task)
        return BoundTaskCache[S2, P, R](task, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
        return self.task(self.__self__, *args, **kwargs)

    def __repr__(self) -> str:
        name = getattr(self.__wrapped__, "__qualname__", "?")
        return f"<bound task cache {name} of {self.__self__!r}>"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)

    @property
    def __doc__(self) -> str | None:  # type: ignore[reportIncompatibleMethodOverride]
        return self.task.__doc__

    @property
    def __annotations__(self) -> dict[str, Any]:  # type: ignore[reportIncompatibleMethodOverride]
        return self.task.__annotations__


class TTLTaskCache[**P, R](Cacheable[P, R]):
    """Task cache with TTL support."""

    __slots__ = ("__dict__", "__weakref__", "__wrapped__", "cache", "ttl")

    def __init__(self, fn: TaskCoroFn[P, R], ttl: float):
        self.ttl = ttl
        self.__wrapped__ = fn
        self.cache: dict[CacheKey, asyncio.Task[R]] = {}

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            return self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            call_after_ttl = partial(
                asyncio.get_running_loop().call_later,
                self.ttl,
                self.cache.pop,
                key,
            )
            task.add_done_callback(call_after_ttl)  # type: ignore[reportArgumentType]
            return task

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.cache.pop(key)


def _lru_evict(
    ttl: float,
    cache: LRU[HashedSequence | int | str, asyncio.Task[Any]],
    key: HashedSequence | int | str,
    _task: object,
) -> None:
    asyncio.get_event_loop().call_later(ttl, cache.remove, key)


class LRUTaskCache[**P, R](Cacheable[P, R]):
    """LRU task cache with optional TTL."""

    __slots__ = ("__dict__", "__weakref__", "__wrapped__", "cache", "ttl")

    def __init__(self, fn: TaskCoroFn[P, R], maxsize: int, ttl: float | None = None):
        self.cache: LRU[CacheKey, asyncio.Task[R]] = LRU(maxsize)
        self.ttl = ttl
        self.__wrapped__ = fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[R]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            return self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            if self.ttl is not None:
                call_after_ttl = partial(_lru_evict, self.ttl, self.cache, key)
                task.add_done_callback(call_after_ttl)
            return task

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.cache.remove(key)


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


@overload
def task_cache[**P, R](ttl: TaskCoroFn[P, R], /) -> Cacheable[P, R]: ...
@overload
def task_cache[**P, R](*, ttl: float) -> Callable[[TaskCoroFn[P, R]], Cacheable[P, R]]: ...
def task_cache[**P, R](
    task_coro_fn: TaskCoroFn[P, R] | None = None,
    *,
    ttl: float | None = None,
) -> Cacheable[P, R] | Callable[[TaskCoroFn[P, R]], Cacheable[P, R]]:
    if task_coro_fn is not None:
        if ttl is not None:
            msg = "Cannot specify ttl when using @task_cache without parentheses"
            raise TypeError(msg)
        wrapper = TaskCache(task_coro_fn)
        update_wrapper(wrapper, task_coro_fn)
        return wrapper

    def decorator(fn: TaskCoroFn[P, R]) -> Cacheable[P, R]:
        wrapper = TTLTaskCache(fn, ttl) if ttl is not None else TaskCache(fn)
        update_wrapper(wrapper, fn)
        return wrapper

    return decorator


@overload
def lru_task_cache[**P, R](task_coro_fn: TaskCoroFn[P, R], /) -> LRUTaskCache[P, R]: ...
@overload
def lru_task_cache[**P, R](
    *,
    maxsize: int = 1024,
    ttl: float | None = None,
) -> Callable[[TaskCoroFn[P, R]], LRUTaskCache[P, R]]: ...
def lru_task_cache[**P, R](
    task_coro_fn: TaskCoroFn[P, R] | None = None,
    *,
    maxsize: int = 1024,
    ttl: float | None = None,
) -> Callable[[TaskCoroFn[P, R]], LRUTaskCache[P, R]] | LRUTaskCache[P, R]:
    if task_coro_fn is not None:
        if ttl is not None or maxsize != 1024:  # noqa: PLR2004
            msg = "Cannot specify ttl or maxsize when using @lru_task_cache without parentheses"
            raise TypeError(msg)
        wrapper = LRUTaskCache(task_coro_fn, maxsize)
        update_wrapper(wrapper, task_coro_fn)
        return wrapper

    def decorator(fn: TaskCoroFn[P, R]) -> LRUTaskCache[P, R]:
        wrapper = LRUTaskCache(fn, maxsize, ttl)
        update_wrapper(wrapper, fn)
        return wrapper

    return decorator
