from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

type Coro[T] = Coroutine[Any, Any, T]
type CoroFn[**P, T] = Callable[P, Coro[T]]
