"""HTTP client for perception (:8001).

Every tool the agent has bottoms out in a method here. Two rules hold
throughout:

1. **Never raise at the agent.** A tool that throws ends the turn; a tool that
   returns `{"ok": false, "error": "..."}` lets the agent read the reason and
   try something else. Perception is careful to make its 422s specific
   ("label 'duck' is not in the yoloe vocabulary") precisely so they can be
   acted on, and throwing that away would waste it.
2. **HTTP is HTTP.** This file speaks request/response only. The event stream
   is a WebSocket and lives in `events/watcher.py`; the two never mix.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import CFG

log = logging.getLogger("orchestrator.perception")

#: What a tool returns. `ok` false always carries a human-readable `error`.
Result = dict[str, Any]


def _ok(**fields: Any) -> Result:
    return {"ok": True, **fields}


def _err(message: str, **fields: Any) -> Result:
    return {"ok": False, "error": message, **fields}


def _reason(response: httpx.Response) -> str:
    """The most specific message the response carries."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        for key in ("detail", "message", "reason"):
            value = body.get(key)
            if isinstance(value, str):
                return value
            if value:
                return str(value)[:300]
    return str(body)[:300]


class Perception:
    """One client per process. Holds a connection pool, so it is reused."""

    def __init__(self, base: str | None = None, timeout: float | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base = (base or CFG.perception_base).rstrip("/")
        # `transport` is how the tests put the pipeline in-process. Everything
        # else — URLs, status handling, the {"ok": ...} contract — stays on the
        # production path, so what passes here is what ships.
        self._client = httpx.AsyncClient(
            base_url=self.base,
            timeout=timeout or CFG.tool_timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, path: str, **kw: Any) -> Result:
        try:
            response = await self._client.request(method, path, **kw)
        except httpx.TimeoutException:
            return _err(f"perception did not answer {path} within "
                        f"{CFG.tool_timeout_s:.0f}s")
        except httpx.HTTPError as exc:
            return _err(f"cannot reach perception at {self.base}: {exc}")

        if response.status_code >= 400:
            return _err(_reason(response), status=response.status_code)

        if not response.content:
            return _ok()
        try:
            payload = response.json()
        except ValueError:
            return _ok(raw=response.text[:300])
        return _ok(**payload) if isinstance(payload, dict) else _ok(data=payload)

    # 1. Behaviours ------------------------------------------------------

    async def start_behavior(self, spec: dict[str, Any]) -> Result:
        """POST /behaviors. A 422 names exactly what was wrong with the spec."""
        return await self._call("POST", "/behaviors", json=spec)

    async def behavior_status(self, behavior_id: str) -> Result:
        return await self._call("GET", f"/behaviors/{behavior_id}/status")

    async def operation_status(self, operation_id: str) -> Result:
        return await self._call("GET", f"/operations/{operation_id}")

    async def list_behaviors(self) -> Result:
        return await self._call("GET", "/behaviors")

    async def stop_behavior(self, behavior_id: str) -> Result:
        return await self._call("DELETE", f"/behaviors/{behavior_id}")

    async def clear_behaviors(self) -> Result:
        """Everything stops. An instruction that replaces the objective."""
        return await self._call("DELETE", "/behaviors")

    # 2. Asking the pipeline ---------------------------------------------

    async def probe(self, phrases: list[str]) -> Result:
        """Does this wording find anything, in this room, right now?

        The single most valuable call available. Phrasing does not transfer
        between scenes, so a phrase that worked yesterday is a guess; this
        turns it into a measurement. Runs on a spare detector, so probing
        never disturbs what is being tracked.
        """
        return await self._call("POST", "/probe", json={"phrases": phrases[:6]})

    async def count(self, selector: dict[str, Any], window_s: float = 1.0) -> Result:
        """Median over a window, so one bad frame cannot change the answer."""
        return await self._call("POST", "/query/count",
                                json={"selector": selector, "window_s": window_s})

    async def look(self, selector: dict[str, Any] | None = None) -> Result:
        return await self._call("POST", "/query/look", json={"selector": selector})

    async def describe(self, question: str, candidates: list[str] | None = None,
                       sweep: bool = True) -> Result:
        """Everything needed to answer a question about the scene.

        Perception has no language model and should not grow one. This returns
        the objects it can see and a prompt carrying that grounding; the agent
        supplies the model.
        """
        return await self._call("POST", "/describe", json={
            "question": question,
            "candidates": candidates or [],
            "sweep": sweep,
        })

    # 3. References ------------------------------------------------------

    async def add_reference_from_frame(self, source: str = "largest_person",
                                       label: str | None = None) -> Result:
        """Latch onto what is on screen now — how "follow him" works with
        nothing to upload."""
        return await self._call("POST", "/references",
                                json={"from": source, "label": label})

    async def add_reference_from_image(self, image: bytes, filename: str,
                                       label: str | None = None) -> Result:
        """An uploaded photo. Demos 2 and 12 start here."""
        files = {"file": (filename, image, "application/octet-stream")}
        data = {"label": label} if label else None
        return await self._call("POST", "/references", files=files, data=data)

    async def list_references(self) -> Result:
        return await self._call("GET", "/references")

    # 4. Pipeline state --------------------------------------------------

    async def health(self) -> Result:
        return await self._call("GET", "/health")

    async def state(self) -> Result:
        return await self._call("GET", "/state")

    async def models(self) -> Result:
        return await self._call("GET", "/models")

    async def set_model(self, name: str, role: str | None = None) -> Result:
        """Behaviours the new model cannot serve are paused with a reason and
        resume when a capable one returns. A paused behaviour is not a
        failure; do not restart it."""
        return await self._call("POST", "/model", json={"name": name, "role": role})

    async def set_hud(self, text: str) -> Result:
        return await self._call("POST", "/hud", json={"text": text})

    # 5. Media -------------------------------------------------------------

    async def snapshot(self) -> bytes | None:
        """The raw, un-annotated frame, for the vision model. Overlays would
        be read as part of the scene."""
        try:
            response = await self._client.get("/snapshot")
        except httpx.HTTPError as exc:
            log.warning("snapshot failed: %s", exc)
            return None
        return response.content if response.status_code == 200 else None

    async def event_snapshot(self, url: str) -> bytes | None:
        """The crop an event fired on, so the agent can look rather than ask."""
        try:
            response = await self._client.get(url)
        except httpx.HTTPError as exc:
            log.warning("event snapshot failed: %s", exc)
            return None
        return response.content if response.status_code == 200 else None

    def absolute(self, path: str) -> str:
        """A pipeline-relative URL the browser can fetch directly."""
        return path if path.startswith("http") else f"{self.base}{path}"
