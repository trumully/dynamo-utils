import asyncio

import pytest
from dynamo_utils.task_cache import LRU, lru_task_cache, task_cache


def test_lru_get():
    cache: LRU[str, int] = LRU(2)
    cache["a"] = 1
    cache["b"] = 2

    assert cache.get("a") == 1
    assert cache.get("b") == 2

    # Raises KeyError if `default` is not set
    with pytest.raises(KeyError):
        cache.get("c")
    assert cache.get("c", None) is None


@pytest.mark.asyncio
async def test_basic_task_cache():
    call_count = 0

    @task_cache
    async def cached_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call should execute
    result1 = await cached_func(5)
    assert result1 == 10
    assert call_count == 1

    # Second call with same args should use cache
    result2 = await cached_func(5)
    assert result2 == 10
    assert call_count == 1

    # Different args should execute new call
    result3 = await cached_func(7)
    assert result3 == 14
    assert call_count == 2


@pytest.mark.asyncio
async def test_task_cache_ttl():
    call_count = 0

    @task_cache(ttl=0.1)
    async def cached_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call
    result1 = await cached_func(5)
    assert result1 == 10
    assert call_count == 1

    # Immediate second call should use cache
    result2 = await cached_func(5)
    assert result2 == 10
    assert call_count == 1

    # Wait for TTL to expire
    await asyncio.sleep(0.2)

    # Call after TTL should execute again
    result3 = await cached_func(5)
    assert result3 == 10
    assert call_count == 2


@pytest.mark.asyncio
async def test_lru_task_cache():
    call_count = 0

    @lru_task_cache(maxsize=2)
    async def cached_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # Fill cache
    await cached_func(1)
    await cached_func(2)
    assert call_count == 2

    # This should evict the first result
    await cached_func(3)
    assert call_count == 3

    # This should cause a new call since 1 was evicted
    await cached_func(1)
    assert call_count == 4
