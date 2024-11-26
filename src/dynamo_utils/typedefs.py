from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

type Coro[R] = Coroutine[Any, Any, R]
type CoroFn[**P, R] = Callable[P, Coro[R]]
type TaskFn[**P, R] = CoroFn[P, R] | Callable[P, asyncio.Task[R]]
type TaskCoroFn[**P, R] = CoroFn[P, R] | TaskFn[P, R]
