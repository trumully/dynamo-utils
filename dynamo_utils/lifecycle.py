"""Implementation of graceful lifecycle management for async applications."""

from __future__ import annotations

import asyncio
import logging
import select
import signal
import socket
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any, Protocol, final

log = logging.getLogger(__name__)

# Type definitions
type SignalCallback = Callable[[signal.Signals], Any]
type StartStopCall = Callable[[], Any]
type _Handler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None

# Available signals
_POSSIBLE_SIGNALS = ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP")
_ACTUAL_SIGNALS = tuple(
    sig for name, sig in signal.Signals.__members__.items() if name in _POSSIBLE_SIGNALS
)


class LifecycleHooks[Context](Protocol):
    """Protocol defining the required hooks for lifecycle management."""

    def sync_setup(self, context: Context) -> Any: ...
    async def async_main(self, context: Context) -> None: ...
    async def async_cleanup(self, context: Context) -> None: ...
    def sync_cleanup(self, context: Context) -> Any: ...


@final
class SignalService:
    """Manages graceful signal handling where the main thread handles signals.

    This service should be paired with event loops running in separate threads.
    """

    def __init__(
        self,
        *,
        startup: list[StartStopCall],
        signal_handlers: list[SignalCallback],
        joins: list[StartStopCall],
    ) -> None:
        """Initialize the signal service.

        Args:
            startup: List of callables to run during startup
            signal_handlers: List of signal handling callbacks
            joins: List of callables to run during shutdown
        """
        self._startup = startup
        self._handlers = signal_handlers
        self._joins = joins

    def add_async_lifecycle(self, lifecycle: AsyncLifecycle[Any], /) -> None:
        """Add an async lifecycle to the service.

        Args:
            lifecycle: The lifecycle instance to add
        """
        start, handler, join = lifecycle.get_service_args()
        self._startup.append(start)
        self._handlers.append(handler)
        self._joins.append(join)

    def run(self) -> None:
        """Run the service and handle signals."""
        server_sock, client_sock = socket.socketpair()
        for sock in (server_sock, client_sock):
            sock.setblocking(False)

        signal.set_wakeup_fd(client_sock.fileno())
        original_handlers: list[_Handler] = []

        try:
            # Setup signal handlers
            for sig in _ACTUAL_SIGNALS:
                original_handlers.append(signal.getsignal(sig))
                signal.signal(sig, lambda s, f: None)
                if sys.platform != "win32":
                    signal.siginterrupt(sig, False)

            # Run startup tasks
            for task in self._startup:
                task()

            # Wait for signal
            select.select([server_sock], [], [])
            data = server_sock.recv(4096)[0]

            # Handle signal
            for handler in self._handlers:
                handler(signal.Signals(data))

            # Run cleanup
            for join in self._joins:
                join()

        finally:
            # Restore original signal handlers
            for sig, original in zip(_ACTUAL_SIGNALS, original_handlers, strict=True):
                signal.signal(sig, original)

            server_sock.close()
            client_sock.close()


@final
class AsyncLifecycle[Context]:
    """Manages the lifecycle of an async application component."""

    def __init__(
        self,
        context: Context,
        loop: asyncio.AbstractEventLoop,
        signal_queue: asyncio.Queue[signal.Signals],
        hooks: LifecycleHooks[Context],
        *,
        timeout: float = 0.1,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            context: The application context
            loop: The event loop to use
            signal_queue: Queue for signal handling
            hooks: Lifecycle hook implementations
            timeout: Timeout for cleanup operations
        """
        self.context = context
        self.loop = loop
        self.signal_queue = signal_queue
        self.hooks = hooks
        self.timeout = timeout
        self.thread: threading.Thread | None = None

    def get_service_args(self) -> tuple[StartStopCall, SignalCallback, StartStopCall]:
        """Get the service arguments for integration with SignalService."""

        def runner() -> None:
            """Main runner function executed in the thread."""
            loop = self.loop
            loop.set_task_factory(asyncio.eager_task_factory)
            asyncio.set_event_loop(loop)

            self.hooks.sync_setup(self.context)

            async def handle_signal() -> None:
                await self.signal_queue.get()
                log.info("Received shutdown signal, initiating shutdown.")
                loop.call_soon(loop.stop)

            async def wrapped_main() -> None:
                main_task = asyncio.create_task(
                    self.hooks.async_main(self.context),
                    name="lifecycle.main",
                )
                signal_task = asyncio.create_task(
                    handle_signal(),
                    name="lifecycle.signal_handler",
                )
                await asyncio.gather(main_task, signal_task)

            def stop_when_done(_fut: asyncio.Future[None]) -> None:
                loop.stop()

            future = asyncio.ensure_future(wrapped_main(), loop=loop)

            try:
                future.add_done_callback(stop_when_done)
                loop.run_forever()
            finally:
                future.remove_done_callback(stop_when_done)

            loop.run_until_complete(self.hooks.async_cleanup(self.context))
            self._finalize_tasks()
            self.hooks.sync_cleanup(self.context)

        def start() -> None:
            """Start the lifecycle thread."""
            if self.thread is not None:
                msg = "Lifecycle is not re-entrant"
                raise RuntimeError(msg)
            self.thread = threading.Thread(target=runner, name="lifecycle-runner")
            self.thread.start()

        def handle_signal(sig: signal.Signals) -> None:
            """Handle incoming signals."""
            self.loop.call_soon_threadsafe(self.signal_queue.put_nowait, sig)

        def join() -> None:
            """Join the lifecycle thread."""
            if thread := self.thread:
                thread.join()

        return start, handle_signal, join

    def _finalize_tasks(self) -> None:
        """Finalize remaining tasks during shutdown."""
        tasks = {
            t
            for t in asyncio.all_tasks(self.loop)
            if not t.done() and t is not asyncio.current_task(self.loop)
        }
        if not tasks:
            return

        async def cleanup() -> None:
            _done, pending = await asyncio.wait(tasks, timeout=self.timeout)
            if not pending:
                log.debug("All tasks completed successfully")
                return

            for task in pending:
                task.cancel()

            _done, pending = await asyncio.wait(pending, timeout=self.timeout)
            for task in pending:
                log.warning(
                    "Task %r wrapping %r did not exit properly",
                    task.get_name(),
                    task.get_coro(),
                )

        self.loop.run_until_complete(cleanup())
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.run_until_complete(self.loop.shutdown_default_executor())

        for task in tasks:
            if not task.cancelled() and (exc := task.exception()):
                self.loop.call_exception_handler({
                    "message": "Unhandled exception during shutdown",
                    "exception": exc,
                    "task": task,
                })

        asyncio.set_event_loop(None)
        self.loop.close()
