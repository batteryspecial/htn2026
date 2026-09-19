"""HTTP and WebSocket surface. Port 8001.

Shaped by who is on the other end. The **agent** starts and stops behaviours
and asks questions; it thinks in seconds and must never be blocked, so every
change returns immediately and is applied in the background. The **frontend**
watches the enriched feed and a state channel.

Nothing here mutates pipeline state. Requests become operations the frame loop
picks up, which keeps that loop the only writer.

Rejections are specific on purpose: the agent reads the reason and retries, so
"label 'duck' is not in the rfdetr vocabulary" is worth more than "invalid".
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from behaviors.kinds import BUILT, KINDS
from config import CFG, resolve_device
from contracts import (
    BehaviorCreated,
    BehaviorSpec,
    CountQuery,
    CountResult,
    Health as HealthView,
    HudText,
    LookQuery,
    LookResult,
    ModelChoice,
)
from detectors.registry import Registry
from runtime.capture import Capture
from runtime.events import Bus
from runtime.health import Health
from runtime.loop import InferenceLoop
from runtime.workers import Builder
from server.debug_page import DEBUG_PAGE

log = logging.getLogger("perception.api")


@dataclass
class Service:
    """Everything the endpoints need, wired together once in main.py."""

    registry: Registry
    capture: Capture
    builder: Builder
    loop: InferenceLoop
    health: Health
    events: Bus
    states: Bus
    snapshots: dict[str, bytes] = field(default_factory=dict)

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
    from server.stream import Streamer

    streamer = Streamer(svc.loop)
    app.state.svc = svc
    app.state.streamer = streamer

    # 1. Behaviours ------------------------------------------------------
    @app.post("/behaviors", status_code=201)
    async def add_behavior(spec: BehaviorSpec):
        """Start a standing instruction. The id comes back at once; the
        behaviour is installed between two frames."""
        if spec.kind not in BUILT:
            return _reject(f"behaviour kind {spec.kind!r} is not implemented yet; "
                           f"available: {sorted(BUILT)}")
        bad = _vocab_error(svc, spec)
        if bad:
            return _reject(bad)
        return BehaviorCreated(id=svc.builder.add_behavior(spec))

    @app.get("/behaviors")
    async def list_behaviors():
        state = svc.loop.last_state
        return {
            "behaviors": [b.model_dump() for b in (state.behaviors if state else [])],
            "kinds": {"available": sorted(BUILT), "planned": sorted(set(KINDS) - BUILT)},
        }

    @app.delete("/behaviors/{behavior_id}", status_code=202)
    async def remove_behavior(behavior_id: str):
        if behavior_id not in svc.loop.behaviors:
            return JSONResponse(status_code=404,
                                content={"detail": f"no behaviour {behavior_id!r}"})
        svc.builder.remove_behavior(behavior_id)
        return {"removed": behavior_id}

    @app.delete("/behaviors", status_code=202)
    async def clear_behaviors():
        svc.builder.clear()
        return {"cleared": True}

    # 2. Model -----------------------------------------------------------
    @app.post("/model", status_code=202)
    async def set_model(choice: ModelChoice):
        """Swap the detector. Behaviours the new model cannot see are paused
        with a reason and resume when a model that can see them returns."""
        try:
            entry = svc.registry.entry(choice.name)
        except KeyError:
            return _reject(f"unknown model {choice.name!r}; "
                           f"available: {svc.registry.names()}")
        if not entry.available:
            return _reject(f"model {choice.name!r} unavailable: {entry.unavailable_reason}")
        svc.builder.set_model(choice.name)
        return {"accepted": True, "model": choice.name}

    @app.get("/models")
    async def models():
        """What this machine can do. The agent's device manifest."""
        return {"models": svc.registry.manifest(), "device": resolve_device()}

    # 3. Introspection ---------------------------------------------------
    @app.get("/health")
    async def health() -> HealthView:
        return HealthView(
            status=svc.health.status, fps=round(svc.loop.timings.fps, 1),
            model=svc.loop.model_name, camera_ok=svc.health.camera_ok,
            behaviors=len(svc.loop.behaviors), device=resolve_device(),
            detail=svc.health.detail,
        )

    @app.get("/state")
    async def state():
        s = svc.loop.last_state
        return s.model_dump(mode="json") if s else {}

    @app.post("/hud")
    async def set_hud(body: HudText):
        """The instruction the operator typed, drawn on the frame."""
        svc.loop.hud_text = body.text or None
        return {"hud": svc.loop.hud_text}

    # 4. Queries ---------------------------------------------------------
    @app.post("/query/count")
    async def query_count(q: CountQuery) -> CountResult:
        """Median over a window, so one bad frame cannot change the answer."""
        counts = await _sample(svc, q.selector, q.window_s)
        return CountResult(count=int(statistics.median(counts)) if counts else 0,
                           samples=len(counts), window_s=q.window_s)

    @app.post("/query/look")
    async def query_look(q: LookQuery) -> LookResult:
        """What is in front of the camera right now, with attribute scores."""
        state = svc.loop.last_state
        tracks = list(state.tracks) if state else []
        if q.selector:
            wanted = set(q.selector.detect)
            tracks = [t for t in tracks if t.label in wanted]
        return LookResult(tracks=tracks)

    @app.get("/snapshot")
    async def snapshot():
        """The raw frame, for the agent's vision model. No overlays: they
        would be read as part of the scene."""
        data = streamer.raw()
        if not data:
            return JSONResponse(status_code=503, content={"detail": "no frame"})
        return Response(content=data, media_type="image/jpeg")

    @app.get("/snapshots/{event_id}.jpg")
    async def event_snapshot(event_id: str):
        data = svc.snapshots.get(event_id)
        if not data:
            return JSONResponse(status_code=404, content={"detail": "no such snapshot"})
        return Response(content=data, media_type="image/jpeg")

    # 5. Live channels ----------------------------------------------------
    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        """One StateView per frame. Level-triggered and lossy: a client that
        falls behind gets the newest, not a backlog of stale frames."""
        await _pump(ws, svc.states, replay=False)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        """Things that happened. Edge-triggered, replayed to a late client."""
        await _pump(ws, svc.events, replay=True)

    # 6. Video ------------------------------------------------------------
    @app.get("/video")
    async def video():
        return StreamingResponse(
            streamer.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/frame.jpg")
    async def frame_jpg():
        return Response(content=streamer.jpeg(), media_type="image/jpeg")

    @app.get("/", response_class=HTMLResponse)
    async def debug():
        """Dev page. Not the operator UI, which lives with the orchestrator."""
        return DEBUG_PAGE

    return app


# --- helpers -------------------------------------------------------------
async def _sample(svc: Service, selector, window_s: float) -> list[int]:
    """Count matching tracks over a window of published frames."""
    wanted = set(selector.detect)
    deadline = time.time() + window_s
    counts, seen = [], None
    while time.time() < deadline:
        state = svc.loop.last_state
        if state is not None and state.ts != seen:
            seen = state.ts
            counts.append(sum(1 for t in state.tracks if t.label in wanted))
        await asyncio.sleep(0.02)
    return counts


def _vocab_error(svc: Service, spec: BehaviorSpec) -> str | None:
    """Fixed-vocabulary models are checked before anything is queued, so the
    agent is told at once rather than by an event a second later."""
    world = svc.loop.world
    detector = world.detector if world else None
    if detector is None or detector.classes is None:
        return None
    missing = [c for c in spec.subject.prompts() if c not in set(detector.classes)]
    if missing:
        return (f"label(s) {missing} are not in the {detector.name} vocabulary; "
                f"it knows {len(detector.classes)} classes, "
                f"e.g. {detector.classes[:8]}")
    return None


def _reject(reason: str) -> JSONResponse:
    log.warning("rejected: %s", reason)
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
