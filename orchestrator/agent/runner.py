"""Owns a turn: build the messages, run the graph, publish the result.

Turns are serialised. A user turn and an event turn arriving together would
otherwise both read "what is running", both decide to replace it, and race
each other to an empty pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Awaitable

from langchain_core.messages import HumanMessage

from agent.graph import build_graph, opening_messages
from app.trace import BUS
from config import CFG
from memory.store import PhraseMemory
from tools.perception import Perception

log = logging.getLogger("orchestrator.runner")

_turns = itertools.count(1)


@dataclass
class Attachment:
    """An image the operator uploaded with their message."""

    filename: str
    content_type: str
    data: bytes

    def data_url(self) -> str:
        kind = self.content_type or "image/jpeg"
        return f"data:{kind};base64,{base64.b64encode(self.data).decode()}"


@dataclass
class TurnResult:
    turn: str
    reply: str
    seconds: float
    ok: bool = True
    outcome: str = "reply"
    behaviors: list[dict] = field(default_factory=list)


class AgentRunner:
    def __init__(self, perception: Perception, memory: PhraseMemory) -> None:
        self.perception = perception
        self.memory = memory
        self.graph = build_graph(perception, memory)
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._history: list[list] = []
        self._stopping = False
        self._cancelled: dict[str, None] = {}
        self._turn_receipts: dict[str, dict[str, dict]] = {}

    async def _submit(self, work: Awaitable[TurnResult]) -> TurnResult:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        if self._stopping:
            task.cancel()
        try:
            return await task
        finally:
            self._tasks.discard(task)

    async def stop(self, turn: str | None = None) -> dict:
        """Cancel active/queued turns before clearing the pipeline."""
        self._stopping = True
        if turn:
            self._cancelled[turn] = None
            if len(self._cancelled) > 200:
                self._cancelled.pop(next(iter(self._cancelled)))
        try:
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._history.clear()
            result = await self.perception.clear_behaviors()
            await self.perception.set_hud("")
            BUS.emit("thought", "Stopped", "Cancelled turns and cleared behaviors")
            return result
        finally:
            self._stopping = False

    async def user_turn(self, text: str,
                        attachments: list[Attachment] | None = None,
                        turn: str | None = None) -> TurnResult:
        return await self._submit(self._user_turn(text, attachments, turn))

    async def _user_turn(self, text: str, attachments: list[Attachment] | None,
                         turn: str | None) -> TurnResult:
        """What the operator typed or said."""
        # The id is allocated here rather than in `_run` so that registering a
        # photo — which happens before the graph starts — is traced under the
        # turn it belongs to, and after the operator's own line rather than
        # above it.
        turn = turn or f"turn-{next(_turns)}"
        if turn in self._cancelled:
            raise asyncio.CancelledError
        BUS.emit("user", text, turn=turn)

        started = time.time()
        deadline = asyncio.get_running_loop().time() + CFG.turn_timeout_s

        try:
            async with asyncio.timeout_at(deadline):
                images: list[str] = []
                prefix = ""

                if attachments:
                    note = await self._register(attachments, turn)
                    if note:
                        prefix = f"{note}\n\n"
                    images = [a.data_url() for a in attachments]

                # Capture only when this turn is ready to enter the serialized
                # runner; a frame taken before a long queue wait is stale.
                async with self._lock:
                    snapshot = await self.perception.snapshot()
                    live_image = None
                    if snapshot:
                        live_image = (
                            "data:image/jpeg;base64,"
                            + base64.b64encode(snapshot).decode())

                    return await self._run_locked(
                        turn, f"{prefix}{text}".strip(), "user", images,
                        live_image, started)
        except TimeoutError:
            return await self._timed_out(turn, started)

    async def event_turn(self, summary: str, snapshot: bytes | None) -> TurnResult:
        return await self._submit(self._event_turn(summary, snapshot))

    async def _event_turn(self, summary: str, snapshot: bytes | None) -> TurnResult:
        """The pipeline woke us. A crop of what happened is usually attached."""
        turn = f"turn-{next(_turns)}"
        BUS.emit("event", summary, turn=turn)

        live_image = None
        if snapshot:
            live_image = (
                f"data:image/jpeg;base64,{base64.b64encode(snapshot).decode()}")
        started = time.time()
        deadline = asyncio.get_running_loop().time() + CFG.turn_timeout_s
        try:
            async with asyncio.timeout_at(deadline):
                async with self._lock:
                    return await self._run_locked(
                        turn, summary, "event", [], live_image, started)
        except TimeoutError:
            return await self._timed_out(turn, started)

    async def _register(self, attachments: list[Attachment], turn: str) -> str:
        """Upload each image to perception and report the ids to the agent."""
        lines: list[str] = []
        for item in attachments:
            result = await self.perception.add_reference_from_image(
                item.data, item.filename)
            if result["ok"]:
                ref_id = result.get("ref_id")
                BUS.emit("tool_result", "add_reference",
                         f"{ref_id} from {item.filename}", turn=turn, ref_id=ref_id)
                lines.append(
                    f'The operator attached "{item.filename}". It is registered as '
                    f'reference {ref_id}. To act on that specific thing, put '
                    f'ref_id "{ref_id}" in the selector with pick "ref".')
            else:
                BUS.emit("error", "add_reference", result["error"], turn=turn)
                lines.append(
                    f'The operator attached "{item.filename}", but registering it '
                    f'failed: {result["error"]}. You can still see the image.')
        return "\n".join(lines)

    async def _run(self, turn: str, instruction: str, origin: str,
                   images: list[str], live_image: str | None = None) -> TurnResult:
        """Compatibility entry used by tests; production paths lock earlier."""
        started = time.time()
        async with self._lock:
            return await self._run_locked(
                turn, instruction, origin, images, live_image, started)

    async def _run_locked(self, turn: str, instruction: str, origin: str,
                          images: list[str], live_image: str | None,
                          started: float) -> TurnResult:
        receipts: dict[str, dict] = {}
        self._turn_receipts[turn] = receipts
        try:
            messages = opening_messages(
                instruction, origin, images, live_image_url=live_image)
            messages[1:1] = [message for group in self._history for message in group]
            visual_inputs = [*images, *([live_image] if live_image else [])]
            final = await self.graph.ainvoke({
                    "messages": messages,
                    "turn": turn,
                    "origin": origin,
                    "instruction": instruction,
                    "image_urls": visual_inputs,
                    "steps": 0,
                    "started": started,
                    "installations": receipts,
                    "require_installation": origin == "user",
                }, {"recursion_limit": CFG.max_steps * 3 + 10})
            reply = final.get("reply") or "Done."
            ok = final.get("ok", True)
            outcome = final.get("outcome", "reply" if ok else "error")
            behaviors = final.get("behaviors", [])
            if origin == "user" and ok:
                generated = final["messages"][len(messages):]
                self._history.append([
                    HumanMessage(content=instruction),
                    *[m for m in generated if m.type != "system"],
                ])
                self._history = self._history[-3:]
        except Exception as exc:                    # noqa: BLE001
            log.exception("turn failed")
            ok = False
            outcome = "error"
            behaviors = [
                {"id": behavior_id, "spec": spec}
                for behavior_id, spec in receipts.items()
            ]
            reply = f"That failed: {exc}"
            BUS.emit("error", "turn failed", str(exc)[:300], turn=turn)

        result = TurnResult(
            turn=turn, reply=reply, seconds=time.time() - started, ok=ok,
            outcome=outcome, behaviors=behaviors)
        self._turn_receipts.pop(turn, None)
        return result

    async def _timed_out(self, turn: str, started: float) -> TurnResult:
        # Reconcile with authoritative pipeline state. A tool may have
        # completed immediately before cancellation.
        listing = await self.perception.list_behaviors()
        running = listing.get("behaviors", []) if listing.get("ok") else []
        receipts = self._turn_receipts.pop(turn, {})
        BUS.emit("error", "turn timeout",
                 f"deadline {CFG.turn_timeout_s:.0f}s; {len(running)} running",
                 turn=turn)
        return TurnResult(
            turn=turn,
            reply=("The agent turn timed out. Existing camera behaviors keep "
                   "running; their live state is shown below."),
            seconds=time.time() - started, ok=False, outcome="error",
            behaviors=[{"id": behavior_id, "spec": spec}
                       for behavior_id, spec in receipts.items()],
        )
