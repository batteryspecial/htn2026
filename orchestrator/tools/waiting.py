"""Wait for a standing behavior while perception continues running each frame."""
from __future__ import annotations

import asyncio
import time

from tools.perception import Perception


async def wait_for_state(perception: Perception, behavior_id: str,
                         state: str, timeout_s: float) -> dict:
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        result = await perception.list_behaviors()
        if not result["ok"]:
            return result
        behavior = next((b for b in result.get("behaviors", [])
                         if b.get("id") == behavior_id), None)
        if behavior:
            actual = behavior.get("state")
            if actual == state:
                return {"ok": True, "behavior": behavior}
            if actual in {"FAILED", "PAUSED"}:
                return {"ok": False, "error": f"{behavior_id} is {actual}: {behavior.get('detail', '')}"}
        elif time.monotonic() - started > 2:
            return {"ok": False, "error": f"{behavior_id} is no longer running"}
        await asyncio.sleep(0.25)
    return {"ok": False, "error": f"{behavior_id} did not reach {state} within {timeout_s:g}s"}
