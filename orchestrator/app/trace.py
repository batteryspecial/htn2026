"""The agent's work, published as it happens.

The frontend's AGENT TRACE panel is the whole reason this exists: an agent
loop that shows its tool calls is debuggable on stage, and one that does not
is a black box that either works or does not.

An in-process pub/sub, lossy by design. A browser that falls behind gets the
newest entries, not a backlog of stale ones — the same rule the pipeline's
state channel follows.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

log = logging.getLogger("orchestrator.trace")

Kind = Literal[
    "user",         # what the operator said
    "thought",      # the agent's own reasoning, when the model exposes it
    "tool_call",    # about to call a tool, with its arguments
    "tool_result",  # what came back
    "say",          # spoken to the operator
    "reply",        # the turn's final answer
    "event",        # the pipeline woke us
    "error",
    "timer",        # switch timing, for the frontend's clock
]

_ids = itertools.count(1)


@dataclass
class TraceEntry:
    kind: Kind
    label: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    turn: str = ""
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"t{next(_ids)}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceBus:
    """Fan-out to every connected browser, plus a small replay buffer."""

    def __init__(self, replay: int = 40) -> None:
        self._subscribers: set[asyncio.Queue[TraceEntry]] = set()
        self._recent: list[TraceEntry] = []
        self._replay = replay

    def publish(self, entry: TraceEntry) -> None:
        self._recent.append(entry)
        del self._recent[:-self._replay]

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                # A slow client loses the oldest entry rather than stalling
                # the agent. The trace is a view, not a ledger.
                try:
                    queue.get_nowait()
                    queue.put_nowait(entry)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def emit(self, kind: Kind, label: str, detail: str = "",
             turn: str = "", **data: Any) -> TraceEntry:
        entry = TraceEntry(kind=kind, label=label, detail=detail,
                           turn=turn, data=data)
        self.publish(entry)
        return entry

    def subscribe(self, replay: bool = True) -> asyncio.Queue[TraceEntry]:
        queue: asyncio.Queue[TraceEntry] = asyncio.Queue(maxsize=200)
        if replay:
            for entry in self._recent[-self._replay:]:
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    break
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[TraceEntry]) -> None:
        self._subscribers.discard(queue)

    @property
    def listeners(self) -> int:
        return len(self._subscribers)


#: One bus per process. The API and the agent both reach for it.
BUS = TraceBus()
