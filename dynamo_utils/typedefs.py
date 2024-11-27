from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

type Coro[R] = Coroutine[Any, Any, R]
type CoroFunc[**P, R] = Callable[P, Coro[R]]
type TaskFunc[**P, R] = CoroFunc[P, R] | Callable[P, asyncio.Task[R]]
type TaskCoroFunc[**P, R] = CoroFunc[P, R] | TaskFunc[P, R]
