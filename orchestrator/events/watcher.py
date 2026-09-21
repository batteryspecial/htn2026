"""Wakes the agent when something happens in front of the camera.

Perception's events are a **WebSocket** (`/ws/events`), and this is the only
place in the orchestrator that speaks one outbound. Tool calls are HTTP and
live in `tools/perception.py`; the two never mix.

Two filters stand between an event and an LLM call:

1. `notify` — the pipeline already decides which events are worth waking an
   agent for. A `count_changed` on a highlight is for the screen, not for the
   model.
2. A cooldown per behaviour. A person standing near the duck fires `near` at
   frame rate; without this the agent would be called thirty times a second
   and the demo would stall on rate limits. One alert that arrives is worth
   more than a hundred that get dropped.

Events are also forwarded to the trace regardless of whether they woke the
agent, so the operator sees everything even when the agent stays quiet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

import websockets

from agent.runner import AgentRunner
from app.trace import BUS
from config import CFG
from tools.perception import Perception

log = logging.getLogger("orchestrator.events")

#: Events that are never worth a turn, whatever `notify` says. These fire
#: continuously by nature and carry no decision for the agent.
NEVER_WAKE = {"count_changed", "crossed", "step", "camera_ok"}

# The camera commonly reports a missing frame while its capture thread starts.
# Give recovery a chance to arrive before paying for a turn about a transient.
CAMERA_LOST_GRACE_S = 1.0


def describe(event: dict[str, Any]) -> str:
    """The event as a sentence, which is what the agent is woken with."""
    kind = event.get("type", "something")
    detail = (event.get("detail") or "").strip()
    behavior = event.get("behavior_id")

    head = detail or f"a {kind} event fired"
    where = f" (behaviour {behavior})" if behavior else ""
    return (f"The pipeline reported: {head}{where}. "
            f"Event type: {kind}.")


class EventWatcher:
    """Consumes /ws/events and turns some of them into agent turns."""

    def __init__(self, runner: AgentRunner, perception: Perception) -> None:
        self.runner = runner
        self.perception = perception
        self._task: asyncio.Task | None = None
        self._last_woken: dict[str, float] = {}
        self._stop = asyncio.Event()
        self._pending: dict[str, asyncio.Task] = {}
        self._accept_after = time.time()
        self._seen: dict[str, None] = {}
        self._camera_lost_at = 0.0

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._accept_after = time.time()
            self._task = asyncio.create_task(self._run(), name="event-watcher")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.cancel_pending()

    async def cancel_pending(self) -> None:
        self._accept_after = time.time()
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()

    @property
    def url(self) -> str:
        base = self.perception.base
        return base.replace("http", "ws", 1) + "/ws/events"

    async def _run(self) -> None:
        """Reconnect forever. The pipeline restarting is normal, not an error."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, open_timeout=5) as socket:
                    log.info("watching %s", self.url)
                    BUS.emit("thought", "connected to the pipeline's event stream")
                    backoff = 1.0
                    async for raw in socket:
                        await self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                    # noqa: BLE001
                log.debug("event stream down (%s), retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.6, 10.0)

    async def _handle(self, raw: str | bytes) -> None:
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(event, dict) or not event.get("type"):
            return
        key = f"{event.get('ts')}:{event.get('id')}"
        if key in self._seen:
            return
        self._seen[key] = None
        if len(self._seen) > 1000:
            self._seen.pop(next(iter(self._seen)))

        # The operator sees every event, whether or not the agent reacts.
        BUS.emit("event", event.get("type", "?"),
                 event.get("detail", ""), **{"event": event})

        if event["type"] == "camera_ok":
            # A recovery invalidates both a queued loss alert and one whose
            # model call has started. Cancellation propagates through the
            # runner's submitted task, including while it waits for its lock.
            if event.get("ts", 0) >= self._camera_lost_at:
                pending = self._pending.pop("system:camera_lost", None)
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                self._last_woken.pop("system:camera_lost", None)
            return

        if not self._should_wake(event):
            return

        # Not awaited: a turn takes seconds and the socket must keep draining,
        # or the backlog becomes the thing that breaks.
        key = event.get("behavior_id") or f"system:{event['type']}"
        if event["type"] == "camera_lost":
            self._camera_lost_at = event.get("ts", 0)
        task = asyncio.create_task(self._wake(event))
        self._pending[key] = task
        task.add_done_callback(lambda done: self._pending.pop(key, None)
                               if self._pending.get(key) is done else None)

    def _should_wake(self, event: dict[str, Any]) -> bool:
        kind = event.get("type", "")
        if kind in NEVER_WAKE:
            return False
        if not event.get("notify", True):
            return False

        key = event.get("behavior_id") or f"system:{kind}"
        # Replayed events are useful for the log, never a reason to call the
        # model again. Bound waiting turns so a slow model cannot build a backlog.
        if event.get("ts", 0) < self._accept_after or key in self._pending:
            return False
        if len(self._pending) >= 8:
            return False
        now = time.monotonic()
        last = self._last_woken.get(key, 0.0)
        if now - last < CFG.event_cooldown_s:
            log.debug("suppressed %s on %s (cooldown)", kind, key)
            return False

        self._last_woken[key] = now
        return True

    async def _wake(self, event: dict[str, Any]) -> None:
        try:
            if event["type"] == "camera_lost":
                await asyncio.sleep(CAMERA_LOST_GRACE_S)
                # Recovery can also happen while the websocket reconnects.
                # Consult the current camera state rather than trust an old
                # event when its matching camera_ok was missed.
                health = await self.perception.health()
                if (health.get("ok") and health.get("camera_ok") is True
                        and health.get("status") == "ok"):
                    return
            snapshot = None
            if event.get("snapshot_url"):
                snapshot = await self.perception.event_snapshot(event["snapshot_url"])
            if event.get("ts", 0) < self._accept_after:
                return
            await self.runner.event_turn(describe(event), snapshot)
        except Exception:                               # noqa: BLE001
            log.exception("event turn failed")
