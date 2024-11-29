from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

type Coro[R] = Coroutine[Any, Any, R]
type CoroFunc[**P, R] = Callable[P, Coro[R]]
