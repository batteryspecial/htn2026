"""HTTP and WebSocket surface. Port 8001.

Shaped by who is on the other end:

- The **agent** starts and stops behaviours and asks questions. It works in
  seconds and must never be held up, so every change returns immediately and
  is applied in the background.
- The **frontend** watches the annotated feed and a status channel.
- Anything that acts on the world reads target state, which is level-triggered
  and lossy in its favour: the newest position, never a queue of stale ones.

Nothing here mutates pipeline state. Requests become operations the inference
loop picks up, which keeps that loop the only writer.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from behaviors.kinds import BUILT, KINDS
from config import CFG, resolve_device
from contracts import BehaviorRequest, StatusEvent
from detectors.base import UnknownClassError, validate_vocab
from detectors.registry import Registry
from runtime.builder import Builder
from runtime.capture import Capture
from runtime.events import Bus
from runtime.loop import InferenceLoop
from runtime.state import Machine
from server.debug_page import DEBUG_PAGE
from server.stream import Streamer

log = logging.getLogger("perception.api")


@dataclass
class Service:
    """Everything the endpoints need, wired together once in main.py."""

    registry: Registry
    capture: Capture
    builder: Builder
    loop: InferenceLoop
    machine: Machine
    events: Bus
    states: Bus

    def start(self) -> None:
        self.capture.start()
        self.builder.start()
        self.loop.start()

    def stop(self) -> None:
        self.loop.stop()
        self.builder.stop()
        self.capture.stop()


def create_app(svc: Service, *, run_threads: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        svc.events.attach(loop)
        svc.states.attach(loop)
        if run_threads:
            svc.start()
        log.info("perception up on %s:%s", CFG.HOST, CFG.PORT)
        try:
            yield
        finally:
            if run_threads:
                svc.stop()

    app = FastAPI(title="perception", lifespan=lifespan)
    streamer = Streamer(svc.loop)
    app.state.svc = svc
    app.state.streamer = streamer

    # 1. Behaviours ------------------------------------------------------
    @app.post("/behaviors", status_code=202)
    async def add_behavior(req: BehaviorRequest):
        """Start a standing instruction. Applied between two frames.

        Rejects up front only what can be known without touching a model: an
        unimplemented kind, an unknown model, or a class a fixed-vocabulary
        model cannot produce. Anything else is answered later as a `failed`
        event, because making the agent wait on a model load defeats the point.
        """
        b = req.behavior
        if b.kind not in BUILT:
            return _reject(svc, req.instruction_id, b.behavior_id,
                           f"behaviour kind {b.kind!r} is not implemented yet; "
                           f"available: {sorted(BUILT)}")
        if b.behavior_id in svc.loop.behaviors:
            return _reject(svc, req.instruction_id, b.behavior_id,
                           f"behaviour {b.behavior_id!r} already running")
        bad = _vocab_error(svc, b.subject.prompts())
        if bad:
            return _reject(svc, req.instruction_id, b.behavior_id, bad)

        svc.builder.add_behavior(req.instruction_id, b)
        return {"accepted": True, "behavior_id": b.behavior_id,
                "instruction_id": req.instruction_id}

    @app.get("/behaviors")
    async def list_behaviors():
        """What is running, and what could be. The agent's menu."""
        summary = svc.loop.last_summary
        return {
            "behaviors": [s.model_dump() for s in (summary.behaviors if summary else [])],
            "kinds": {"available": sorted(BUILT),
                      "planned": sorted(set(KINDS) - BUILT)},
        }

    @app.delete("/behaviors/{behavior_id}", status_code=202)
    async def remove_behavior(behavior_id: str):
        if behavior_id not in svc.loop.behaviors:
            return JSONResponse(status_code=404,
                                content={"detail": f"no behaviour {behavior_id!r}"})
        svc.builder.remove_behavior(f"del-{behavior_id}", behavior_id)
        return {"removed": behavior_id}

    @app.delete("/behaviors", status_code=202)
    async def clear_behaviors():
        """Stop everything. The panic button."""
        svc.builder.clear()
        return {"cleared": True}

    # 2. Model -----------------------------------------------------------
    @app.post("/model", status_code=202)
    async def set_model(body: dict):
        """Swap the detector. Resets tracking, because ids from a different
        model mean nothing."""
        name = body.get("model")
        try:
            entry = svc.registry.entry(name)
        except KeyError:
            return _reject(svc, body.get("instruction_id", "model"), None,
                           f"unknown model {name!r}; available: {svc.registry.names()}")
        if not entry.available:
            return _reject(svc, body.get("instruction_id", "model"), None,
                           f"model {name!r} unavailable: {entry.unavailable_reason}")
        svc.builder.set_model(body.get("instruction_id", "model"), name)
        return {"accepted": True, "model": name}

    @app.get("/models")
    async def models():
        """What this machine can actually do. The agent's device manifest."""
        return {"models": svc.registry.manifest(), "device": resolve_device()}

    # 3. Introspection ---------------------------------------------------
    @app.get("/health")
    async def health():
        s = svc.loop.last_summary
        return {
            "phase": svc.machine.phase.value,
            "fps": round(svc.loop.timings.fps, 1),
            "model": svc.loop._model_name,
            "behaviors": len(svc.loop.behaviors),
            "tracks": len(s.tracks) if s else 0,
            "attributes": bool(svc.loop.shared.bank.encoder),
            "last_frame_age_ms": (round(svc.capture.age() * 1000, 1)
                                  if svc.capture.age() != float("inf") else None),
            "source_open": svc.capture.opened,
            "timings_ms": {k: round(v, 2) for k, v in svc.loop.timings.ewma.items()},
            "frames": svc.loop.timings.frames,
            "device": resolve_device(),
        }

    @app.get("/state")
    async def state():
        """The latest frame summary, for a client that would rather poll."""
        s = svc.loop.last_summary
        return s.model_dump(mode="json") if s else {}

    # 4. Live channels ----------------------------------------------------
    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        """One FrameSummary per frame: tracks and behaviour states.

        Level-triggered and lossy: a client that falls behind is given the
        newest frame rather than a backlog of stale ones.
        """
        await _pump(ws, svc.states, replay=False)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        """Things that happened. Edge-triggered, replayed to a late client."""
        await _pump(ws, svc.events, replay=True)

    # 5. Video ------------------------------------------------------------
    @app.get("/video")
    async def video():
        return StreamingResponse(
            streamer.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/frame.jpg")
    async def frame_jpg():
        return StreamingResponse(iter([streamer.jpeg()]), media_type="image/jpeg")

    @app.get("/", response_class=HTMLResponse)
    async def debug():
        """Dev page: the feed, live state, and a box to start behaviours.

        Not the operator UI. This exists so the pipeline can be driven and
        watched with nothing else running.
        """
        return DEBUG_PAGE

    return app


# --- helpers -------------------------------------------------------------
def _vocab_error(svc: Service, prompts: list[str]) -> str | None:
    """Fixed-vocabulary models are checked before anything is queued, so the
    agent is told at once rather than by an event a second later."""
    world = svc.loop.world
    detector = world.detector if world else None
    if detector is None or detector.classes is None:
        return None
    try:
        validate_vocab(prompts, detector.classes, detector.name)
    except UnknownClassError as exc:
        return str(exc)
    return None


def _reject(svc: Service, instruction_id: str, behavior_id: str | None,
            reason: str) -> JSONResponse:
    """422 plus a `failed` event, so a rejection reaches the UI even when the
    caller swallowed the HTTP error."""
    log.warning("rejected %s: %s", instruction_id, reason)
    svc.events.publish(StatusEvent(
        stage="failed", instruction_id=instruction_id, spec_id=behavior_id,
        detail=reason, ts=time.time(),
    ))
    return JSONResponse(status_code=422, content={"detail": reason})


async def _pump(ws: WebSocket, bus: Bus, replay: bool) -> None:
    await ws.accept()
    stream = bus.stream(replay=replay)
    try:
        async for item in stream:
            await ws.send_text(item.model_dump_json())
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        await stream.aclose()
