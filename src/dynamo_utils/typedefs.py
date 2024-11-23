from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

type Coro[T] = Coroutine[Any, Any, T]
type CoroFn[**P, T] = Callable[P, Coro[T]]
type TaskFn[**P, T] = CoroFn[P, T] | Callable[P, asyncio.Task[T]]
type TaskCoroFn[**P, T] = CoroFn[P, T] | TaskFn[P, T]
