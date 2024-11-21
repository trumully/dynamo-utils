import asyncio
from collections.abc import Sequence

import pytest
from dynamo_utils.waterfall import Waterfall


@pytest.mark.asyncio
async def test_waterfall_batch_processing():
    processed_items: list[Sequence[str]] = []

    async def process_batch(items: Sequence[str]) -> None:
        processed_items.append(items)
        await asyncio.sleep(1e-3)  # Simulate work

    waterfall: Waterfall[str] = Waterfall(max_wait=0.5, max_quantity=3, async_callback=process_batch)
    waterfall.start()

    # Add items
    waterfall.put("item1")
    waterfall.put("item2")
    waterfall.put("item3")
    waterfall.put("item4")

    await asyncio.sleep(0.6)
    await waterfall.stop(wait=True)
    await asyncio.sleep(0.1)

    assert len(processed_items) == 2
    assert processed_items[0] == ["item1", "item2", "item3"]
    assert processed_items[1] == ["item4"]


@pytest.mark.asyncio
async def test_waterfall_max_wait():
    processed_items: list[Sequence[str]] = []

    async def process_batch(items: Sequence[str]) -> None:
        processed_items.append(items)

    waterfall: Waterfall[str] = Waterfall(max_wait=0.2, max_quantity=5, async_callback=process_batch)
    waterfall.start()

    waterfall.put("item1")
    await asyncio.sleep(0.3)  # Wait longer than max_wait

    assert len(processed_items) == 1
    assert processed_items[0] == ["item1"]

    await waterfall.stop(wait=True)


@pytest.mark.asyncio
async def test_waterfall_put_in_shut_down():
    processed_items: list[Sequence[str]] = []

    async def process_batch(items: Sequence[str]) -> None:
        processed_items.append(items)

    waterfall: Waterfall[str] = Waterfall(max_wait=0.2, max_quantity=5, async_callback=process_batch)
    waterfall.start()

    waterfall.put("item1")
    await waterfall.stop(wait=True)

    with pytest.raises(RuntimeError):
        waterfall.put("item2")
    assert len(processed_items) == 1
    assert processed_items[0] == ["item1"]
