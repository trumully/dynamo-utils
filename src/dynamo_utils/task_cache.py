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
from typing import TYPE_CHECKING, Any, overload

from .sentinel import Sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dynamo_utils.typedefs import TaskCoroFn


__all__ = ("LRU", "lru_task_cache", "task_cache")


MISSING = Sentinel("MISSING")

type CacheKey = HashedSequence | int | str


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


class TaskCache[**P, T]:
    """Base class for task caching."""

    __slots__: tuple[str, ...] = ("__weakref__", "__dict__", "__wrapped__", "cache")

    __wrapped__: TaskCoroFn[P, T]
    cache: dict[CacheKey, asyncio.Task[T]] | LRU[CacheKey, asyncio.Task[T]]

    def __init__(self, fn: TaskCoroFn[P, T]):
        self.__wrapped__ = fn
        self.cache = {}

    def __get__(self, instance: object, owner: type | None = None) -> TaskCache[P, T] | BoundTaskCache[object, P, T]:
        return self if instance is None else BoundTaskCache(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
        return task

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.cache.pop(key)


class BoundTaskCache[S, **P, T]:
    __slots__ = ("__weakref__", "task", "__self__")

    def __init__(self, task: TaskCache[P, T], instance: object):
        self.task = task
        self.__self__ = instance

    @property
    def __wrapped__(self) -> Any:
        return self.task.__wrapped__

    @property
    def __func__(self) -> TaskCache[..., T]:
        return self.task

    def __get__[S2](
        self: BoundTaskCache[S, P, T],
        instance: S2,
        owner: type | None = None,
    ) -> BoundTaskCache[S2, P, T]:
        return BoundTaskCache[S2, P, T](self.task, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        return self.task(self.__self__, *args, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        name = getattr(self.__wrapped__, "__qualname__", "?")
        return f"<bound task cache {name} of {self.__self__!r}>"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        return self.task.__doc__

    @property
    def __annotations__(self) -> dict[str, Any]:  # type: ignore[override]
        return self.task.__annotations__


class TTLTaskCache[**P, T](TaskCache[P, T]):
    """Task cache with TTL support."""

    cache: dict[CacheKey, asyncio.Task[T]]

    def __init__(self, fn: TaskCoroFn[P, T], ttl: float):
        super().__init__(fn)
        self.ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            call_after_ttl = partial(asyncio.get_event_loop().call_later, self.ttl, self.cache.pop, key)
            task.add_done_callback(call_after_ttl)
        return task


def _lru_evict(
    ttl: float,
    cache: LRU[HashedSequence | int | str, asyncio.Task[Any]],
    key: HashedSequence | int | str,
    _task: MISSING = MISSING,
) -> None:
    asyncio.get_event_loop().call_later(ttl, cache.remove, key)


class LRUTaskCache[**P, T](TaskCache[P, T]):
    """LRU task cache with optional TTL."""

    cache: LRU[CacheKey, asyncio.Task[T]]

    def __init__(self, fn: TaskCoroFn[P, T], maxsize: int, ttl: float | MISSING = MISSING):
        super().__init__(fn)
        self.cache: LRU[CacheKey, asyncio.Task[T]] = LRU(maxsize)
        self.ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.cache[key]
        except KeyError:
            self.cache[key] = task = asyncio.ensure_future(self.__wrapped__(*args, **kwargs))
            if self.ttl is not MISSING:
                call_after_ttl = partial(_lru_evict, self.ttl, self.cache, key)
                task.add_done_callback(call_after_ttl)
        return task


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


@overload
def task_cache[**P, T](task_coro_fn: TaskCoroFn[P, T], /) -> TaskCache[P, T]: ...
@overload
def task_cache[**P, T](*, ttl: float) -> Callable[[TaskCoroFn[P, T]], TTLTaskCache[P, T]]: ...
def task_cache[**P, T](
    task_coro_fn: TaskCoroFn[P, T] | MISSING = MISSING,
    *,
    ttl: float | MISSING = MISSING,
) -> Callable[[TaskCoroFn[P, T]], TaskCache[P, T]] | TaskCache[P, T]:
    if task_coro_fn is not MISSING:
        if ttl is not MISSING:
            msg = "Cannot specify ttl when using @task_cache without parentheses"
            raise TypeError(msg)
        wrapper = TaskCache(task_coro_fn)
        update_wrapper(wrapper, task_coro_fn)
        return wrapper

    def decorator(fn: TaskCoroFn[P, T]) -> TaskCache[P, T]:
        wrapper = TTLTaskCache(fn, ttl) if ttl is not None else TaskCache(fn)
        update_wrapper(wrapper, fn)
        return wrapper

    return decorator


@overload
def lru_task_cache[**P, T](task_coro_fn: TaskCoroFn[P, T], /) -> LRUTaskCache[P, T]: ...
@overload
def lru_task_cache[**P, T](
    *, maxsize: int = 1024, ttl: float | MISSING = MISSING
) -> Callable[[TaskCoroFn[P, T]], LRUTaskCache[P, T]]: ...
def lru_task_cache[**P, T](
    task_coro_fn: TaskCoroFn[P, T] | MISSING = MISSING,
    *,
    maxsize: int = 1024,
    ttl: float | MISSING = MISSING,
) -> Callable[[TaskCoroFn[P, T]], LRUTaskCache[P, T]] | LRUTaskCache[P, T]:
    if task_coro_fn is not MISSING:
        if ttl is not MISSING or maxsize != 1024:  # noqa: PLR2004
            msg = "Cannot specify ttl or maxsize when using @lru_task_cache without parentheses"
            raise TypeError(msg)
        wrapper = LRUTaskCache(task_coro_fn, maxsize)
        update_wrapper(wrapper, task_coro_fn)
        return wrapper

    def decorator(fn: TaskCoroFn[P, T]) -> LRUTaskCache[P, T]:
        wrapper = LRUTaskCache(fn, maxsize, ttl)
        update_wrapper(wrapper, fn)
        return wrapper

    return decorator
