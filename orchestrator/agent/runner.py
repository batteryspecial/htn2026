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


class AgentRunner:
    def __init__(self, perception: Perception, memory: PhraseMemory) -> None:
        self.perception = perception
        self.memory = memory
        self.graph = build_graph(perception, memory)
        self._lock = asyncio.Lock()

    async def user_turn(self, text: str,
                        attachments: list[Attachment] | None = None) -> TurnResult:
        """What the operator typed or said."""
        # The id is allocated here rather than in `_run` so that registering a
        # photo — which happens before the graph starts — is traced under the
        # turn it belongs to, and after the operator's own line rather than
        # above it.
        turn = f"turn-{next(_turns)}"
        BUS.emit("user", text, turn=turn)

        images: list[str] = []
        prefix = ""

        if attachments:
            # Register the photo before the agent thinks, so it is given a
            # ref_id rather than having to discover it needs one. "Track that
            # human" with a photo is two steps, and this is the first.
            note = await self._register(attachments, turn)
            if note:
                prefix = f"{note}\n\n"
            images = [a.data_url() for a in attachments]

        return await self._run(turn, f"{prefix}{text}".strip(), "user", images)

    async def event_turn(self, summary: str, snapshot: bytes | None) -> TurnResult:
        """The pipeline woke us. A crop of what happened is usually attached."""
        turn = f"turn-{next(_turns)}"
        BUS.emit("event", summary, turn=turn)

        images = []
        if snapshot:
            images.append(
                f"data:image/jpeg;base64,{base64.b64encode(snapshot).decode()}")
        return await self._run(turn, summary, "event", images)

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
                   images: list[str]) -> TurnResult:
        started = time.time()

        async with self._lock:
            try:
                final = await self.graph.ainvoke({
                    "messages": opening_messages(instruction, origin, images),
                    "turn": turn,
                    "origin": origin,
                    "instruction": instruction,
                    "image_urls": images,
                    "steps": 0,
                    "started": started,
                }, {"recursion_limit": CFG.max_steps * 2 + 8})
                reply = final.get("reply") or "Done."
            except Exception as exc:                    # noqa: BLE001
                log.exception("turn failed")
                reply = f"That failed: {exc}"
                BUS.emit("error", "turn failed", str(exc)[:300], turn=turn)

        return TurnResult(turn=turn, reply=reply, seconds=time.time() - started)
