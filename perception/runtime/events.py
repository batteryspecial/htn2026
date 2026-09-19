"""Fan-out from the pipeline threads to the WebSocket clients.

Two buses with deliberately different failure modes:

1. Status events are edge-triggered and meant for humans. A dropped one leaves
   a gap in the status board, so they are buffered and replayed to a client
   that connects late.
2. Target states are level-triggered and meant for the controller. A stale one
   is worse than a missing one, so a slow client gets the newest and loses the
   backlog rather than falling behind.

Producers are plain threads, consumers live on the asyncio loop, so every
hand-off goes through `call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any

log = logging.getLogger("perception.events")


class Bus:
    """Thread-safe publish, async subscribe."""

    def __init__(self, *, latest_only: bool = False, history: int = 0, maxsize: int = 256) -> None:
        self._latest_only = latest_only
        self._maxsize = 1 if latest_only else maxsize
        self._history: deque[Any] = deque(maxlen=history)
        self._queues: set[asyncio.Queue] = set()
        self._callbacks: list[Callable[[Any], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self.dropped = 0

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at startup, from the asyncio thread."""
        self._loop = loop

    # 1. Producing -----------------------------------------------------
    def publish(self, item: Any) -> None:
        """Safe from any thread. Never blocks, never raises."""
        if self._history.maxlen:
            self._history.append(item)
        for cb in self._callbacks:
            cb(item)
        loop = self._loop
        if loop is None or not self._queues:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, item)
        except RuntimeError:
            pass  # loop is shutting down

    def _deliver(self, item: Any) -> None:
        for q in self._queues:
            if q.full():
                # Latest-only drops the stale entry; a buffered bus drops the
                # oldest. Either way the producer is never blocked by a client.
                try:
                    q.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(item)

    def subscribe_callback(self, cb: Callable[[Any], None]) -> None:
        """Synchronous tap, used by tests and by in-process wiring."""
        self._callbacks.append(cb)

    # 2. Consuming -----------------------------------------------------
    async def stream(self, replay: bool = True) -> AsyncIterator[Any]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        if replay:
            for item in list(self._history):
                if not q.full():
                    q.put_nowait(item)
        self._queues.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.discard(q)

    def history(self) -> list[Any]:
        return list(self._history)

    @property
    def subscribers(self) -> int:
        return len(self._queues)


def make_buses() -> tuple[Bus, Bus]:
    """Events keep a short backlog so a status board that connects mid-run has
    context. Target states never queue."""
    return Bus(history=50), Bus(latest_only=True)
