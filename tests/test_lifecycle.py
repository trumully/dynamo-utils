import asyncio
from typing import Any

import pytest
from dynamo_utils.lifecycle import LifecycleHooks


class DummyHooks:
    def __init__(self) -> None:
        self.setup_called = False
        self.main_called = False
        self.async_cleanup_called = False
        self.sync_cleanup_called = False
        self.should_hang = False

    def sync_setup(self, context: Any) -> None:
        self.setup_called = True

    async def async_main(self, context: Any) -> None:
        self.main_called = True
        if self.should_hang:
            await asyncio.Event().wait()

    async def async_cleanup(self, context: Any) -> None:
        self.async_cleanup_called = True

    def sync_cleanup(self, context: Any) -> None:
        self.sync_cleanup_called = True


class DummyLifecycle:
    """A simplified lifecycle for testing without threads."""

    def __init__(self, hooks: LifecycleHooks[Any]) -> None:
        self.hooks = hooks
        self._running = False
        self._cleanup_called = False

    async def run(self) -> None:
        """Run the lifecycle synchronously for testing."""
        self.hooks.sync_setup({})
        self._running = True

        main_task = asyncio.create_task(self.hooks.async_main({}), name="dummy.main")

        try:
            await main_task
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self.hooks.async_cleanup({})
            self.hooks.sync_cleanup({})

    def stop(self) -> None:
        """Stop the lifecycle."""
        if self._running:
            for task in asyncio.all_tasks():
                if task.get_name() == "dummy.main":
                    task.cancel()


@pytest.mark.asyncio
async def test_basic_lifecycle() -> None:
    """Test basic lifecycle flow with normal startup and shutdown."""
    hooks = DummyHooks()
    lifecycle = DummyLifecycle(hooks)

    # Start lifecycle in background task
    task = asyncio.create_task(lifecycle.run())
    await asyncio.sleep(0.1)

    # Stop lifecycle
    lifecycle.stop()
    await task

    assert hooks.setup_called
    assert hooks.main_called
    assert hooks.async_cleanup_called
    assert hooks.sync_cleanup_called


@pytest.mark.asyncio
async def test_hanging_tasks() -> None:
    """Test cleanup of hanging tasks during shutdown."""
    hooks = DummyHooks()
    hooks.should_hang = True
    lifecycle = DummyLifecycle(hooks)

    task = asyncio.create_task(lifecycle.run())
    await asyncio.sleep(0.1)
    lifecycle.stop()
    await task

    assert hooks.async_cleanup_called
    assert hooks.sync_cleanup_called
