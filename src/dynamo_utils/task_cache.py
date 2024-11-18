from __future__ import annotations

import asyncio
from collections.abc import Hashable
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Any, Protocol, overload

from .sentinel import Sentinel, sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .typedefs import CoroFn


__all__ = ("task_cache", "lru_task_cache")


KWARGS_SENTINEL = sentinel("KWARGS_SENTINEL")


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
        _sentinel: Hashable = KWARGS_SENTINEL,
    ) -> HashedSequence | int | str:
        key: tuple[Any, ...] = args if not kwargs else (*args, _sentinel, *kwargs.items())
        first: int | str = key[0]
        return first if len(key) == 1 and type(first) in fast_types else cls(key)


class Missing(Sentinel):
    pass


MISSING = Missing(__name__, "MISSING")


class CacheableTask[**P, T](Protocol):
    """A cached coroutine function."""

    __slots__ = ()

    @property
    def __wrapped__(self) -> CoroFn[P, T]: ...

    def __get__[S: object](
        self, instance: S, owner: type[S] | Missing = MISSING
    ) -> CacheableTask[P, T] | BoundCachedTask[S, P, T]:
        return self if instance is MISSING else BoundCachedTask(self, instance)

    __call__: CoroFn[P, T]

    def cache_clear(self) -> None: ...
    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class CachedTask[**P, T](CacheableTask[P, T]):
    """A cached coroutine function."""

    __slots__ = ("__cache", "__ttl", "__dict__", "__wrapped__", "__weakref__")

    __wrapped__: CoroFn[P, T]

    def __init__(self, call: CoroFn[P, T], /, ttl: float | Missing = MISSING) -> None:
        self.__cache: dict[HashedSequence | int | str, asyncio.Task[T]] = {}
        self.__wrapped__ = call
        self.__ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:  # type: ignore[override]
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.__cache[key]
        except KeyError:
            self.__cache[key] = task = asyncio.create_task(self.__wrapped__(*args, **kwargs))
            if self.__ttl is not MISSING:
                call_after_ttl = partial(asyncio.get_event_loop().call_later, self.__ttl, self.__cache.pop, key)  # type: ignore[arg-type]
                task.add_done_callback(call_after_ttl)
        return task

    def cache_clear(self) -> None:
        self.__cache.clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.__cache.pop(key, None)


class LRU[K, V]:
    """A Least Recently Used cache."""

    __slots__ = ("__maxsize", "__cache")

    def __init__(self, maxsize: int, /) -> None:
        self.__cache: dict[K, V] = {}
        self.__maxsize = maxsize

    def get[T](self, key: K, default: T | Missing = MISSING, /) -> V | T:
        try:
            self.__cache[key] = self.__cache.pop(key)
            return self.__cache[key]
        except KeyError as exc:
            # Raise if default is not set explicitly.
            if default is MISSING:
                raise exc from None
            return default  # type: ignore[return-value]

    def __getitem__(self, key: K, /) -> V:
        self.__cache[key] = self.__cache.pop(key)
        return self.__cache[key]

    def __setitem__(self, key: K, value: V, /) -> None:
        self.__cache[key] = value
        if len(self.__cache) > self.__maxsize:
            self.__cache.pop(next(iter(self.__cache)))

    def remove(self, key: K, /) -> None:
        self.__cache.pop(key)

    def clear(self) -> None:
        self.__cache.clear()


def _lru_evict(
    ttl: float,
    cache: LRU[HashedSequence | int | str, asyncio.Task[Any]],
    key: HashedSequence | int | str,
    _task: object = MISSING,
) -> None:
    asyncio.get_event_loop().call_later(ttl, cache.remove, key)


class LRUCachedTask[**P, T](CacheableTask[P, T]):
    """A cached coroutine function with a Least Recently Used cache with support for TTL."""

    __slots__ = ("__cache", "__ttl", "__dict__", "__wrapped__", "__weakref__")

    __wrapped__: CoroFn[P, T]

    def __init__(self, call: CoroFn[P, T], /, maxsize: int, ttl: float | Missing = MISSING) -> None:
        self.__cache: LRU[HashedSequence | int | str, asyncio.Task[T]] = LRU(maxsize)
        self.__wrapped__ = call
        self.__ttl = ttl

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:  # type: ignore[override]
        key = HashedSequence.from_call(args, kwargs)
        try:
            task = self.__cache[key]
        except KeyError:
            self.__cache[key] = task = asyncio.create_task(self.__wrapped__(*args, **kwargs))
            if self.__ttl is not MISSING:
                call_after_ttl = partial(_lru_evict, self.__ttl, self.__cache, key)  # type: ignore[arg-type]
                task.add_done_callback(call_after_ttl)
        return task

    def cache_clear(self) -> None:
        self.__cache.clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = HashedSequence.from_call(args, kwargs)
        self.__cache.remove(key)


class BoundCachedTask[S, **P, T]:
    """A bound cached coroutine function."""

    __slots__ = ("__self__", "__weakref__", "_task")

    def __init__(self, task: CacheableTask[P, T], __self__: S):
        self._task = task
        self.__self__ = __self__

        self.__setattr__("__annotations__", task.__annotations__)
        self.__setattr__("__doc__", task.__doc__)

    @property
    def __wrapped__(self) -> CoroFn[P, T]:
        return self._task.__wrapped__

    @property
    def __func__(self) -> CacheableTask[P, T]:
        return self._task

    def __get__[S2: object](self, instance: S2, owner: type[S2] | None = None) -> BoundCachedTask[S2, P, T]:
        return BoundCachedTask(self._task, instance)

    def cache_clear(self) -> None:
        self._task.cache_clear()

    def cache_discard(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._task.cache_discard(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<bound cached task {getattr(self.__wrapped__, '__qualname__', '?')} of {self.__self__!r}>"


@overload
def task_cache[**P, T](*, ttl: float | Missing = MISSING) -> Callable[[CoroFn[P, T]], CacheableTask[P, T]]: ...
@overload
def task_cache[**P, T](coro: CoroFn[P, T], /) -> CacheableTask[P, T]: ...
def task_cache[**P, T](
    ttl: CoroFn[P, T] | float | Missing = MISSING,
) -> Callable[[CoroFn[P, T]], CacheableTask[P, T]] | CacheableTask[P, T]:
    """Decorator to wrap a coroutine function in a cache with support for TTL."""
    if isinstance(ttl, float):
        ttl = max(0, ttl)
    elif callable(ttl):
        fast_wrapper = CachedTask(ttl)
        update_wrapper(fast_wrapper, ttl)
        return fast_wrapper
    elif ttl is MISSING:
        msg = "First argument must be a float or a coroutine function"
        raise TypeError(msg)

    def decorator(coro: CoroFn[P, T]) -> CacheableTask[P, T]:
        wrapper = CachedTask(coro, ttl)
        update_wrapper(wrapper, coro)
        return wrapper

    return decorator


@overload
def lru_task_cache[**P, T](
    *, ttl: float | Missing = MISSING, maxsize: int = 1024
) -> Callable[[CoroFn[P, T]], CacheableTask[P, T]]: ...
@overload
def lru_task_cache[**P, T](coro: CoroFn[P, T], /) -> CacheableTask[P, T]: ...
def lru_task_cache[**P, T](
    ttl: CoroFn[P, T] | float | Missing = MISSING, maxsize: int = 1024
) -> CacheableTask[P, T] | Callable[[CoroFn[P, T]], CacheableTask[P, T]]:
    """Decorator to wrap a coroutine function in a Least Recently Used cache with support for TTL."""
    if isinstance(ttl, float):
        ttl = max(0, ttl)
    elif callable(ttl):
        fast_wrapper = LRUCachedTask(ttl, maxsize)
        update_wrapper(fast_wrapper, ttl)
        return fast_wrapper

    def decorator(coro: CoroFn[P, T]) -> CacheableTask[P, T]:
        wrapper = LRUCachedTask(coro, maxsize, ttl)
        update_wrapper(wrapper, coro)
        return wrapper

    return decorator
