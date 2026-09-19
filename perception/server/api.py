"""HTTP and WebSocket surface. Port 8001.

The shape of this is set by who is on the other end:

- The orchestrator posts instructions and wants an answer immediately, so
  `POST /spec` validates what it can synchronously and returns 202 without
  waiting for a model. Anything it could not check up front arrives later as a
  `failed` event rather than a stalled request.
- The controller reads target states and must never be held up by the UI, so
  it gets its own channel with its own drop policy.
- A human watches the video and the status board, which tolerate a lag the
  controller would not.

Nothing here mutates pipeline state. Requests become messages the inference
loop picks up, which keeps that loop the only writer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from config import CFG
from contracts import Phase, SpecRequest, StatusEvent
from detectors.base import UnknownClassError, validate_vocab
from detectors.registry import Registry
from runtime.capture import Capture
from runtime.events import Bus
from runtime.loader import Loader
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
    loader: Loader
    loop: InferenceLoop
    machine: Machine
    events: Bus
    targets: Bus

    def start(self) -> None:
        self.capture.start()
        self.loader.start()
        self.loop.start()

    def stop(self) -> None:
        self.loop.stop()
        self.loader.stop()
        self.capture.stop()


def create_app(svc: Service, *, run_threads: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        svc.events.attach(loop)
        svc.targets.attach(loop)
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

    # 1. Instructions --------------------------------------------------
    @app.post("/spec", status_code=202)
    async def post_spec(req: SpecRequest):
        """Accept a new TaskSpec. Preparation happens in the background.

        Rejects here only what can be known without touching a model: an
        unknown model name, a model this machine cannot run, or a class a
        fixed-vocabulary model does not have. Everything else is answered by a
        `failed` event, because making the orchestrator wait on a model load
        would defeat the point.
        """
        spec = req.spec
        try:
            entry = svc.registry.entry(spec.model)
        except KeyError:
            return _reject(svc, req, f"unknown model {spec.model!r}; "
                                     f"available: {svc.registry.names()}")
        if not entry.available:
            return _reject(svc, req, f"model {spec.model!r} unavailable: "
                                     f"{entry.unavailable_reason}")
        if entry.detector is not None and entry.detector.classes is not None:
            try:
                validate_vocab(spec.prompt_union(), entry.detector.classes, spec.model)
            except UnknownClassError as exc:
                return _reject(svc, req, str(exc))

        svc.loader.submit(req.instruction_id, spec)
        return {"accepted": True, "instruction_id": req.instruction_id,
                "spec_id": spec.spec_id}

    @app.delete("/spec", status_code=202)
    async def delete_spec():
        """Stop tracking. The car stops as soon as the loop sees this."""
        svc.loader.clear()
        return {"cleared": True}

    # 2. Introspection -------------------------------------------------
    @app.get("/models")
    async def models():
        """What this machine can actually do.

        The orchestrator builds the LLM's device manifest from this, so a
        fixed-vocabulary model reports its real class list and an unavailable
        one says why.
        """
        return {"models": svc.registry.manifest(), "device": _device()}

    @app.get("/health")
    async def health():
        state = svc.loop.last_state
        task = svc.loop.active
        return {
            "phase": svc.machine.phase.value,
            "fps": round(svc.loop.timings.fps, 1),
            "model": task.model_name if task else None,
            "spec_id": svc.machine.spec_id,
            "instruction_id": svc.machine.instruction_id,
            "visible": bool(state.visible) if state else False,
            "last_frame_age_ms": round(svc.capture.age() * 1000, 1)
            if svc.capture.age() != float("inf") else None,
            "source_open": svc.capture.opened,
            "timings_ms": {k: round(v, 2) for k, v in svc.loop.timings.ewma.items()},
            "frames": svc.loop.timings.frames,
            "device": _device(),
        }

    # 3. Live channels --------------------------------------------------
    @app.websocket("/ws/target")
    async def ws_target(ws: WebSocket):
        """One TargetState per frame, for the controller.

        Level-triggered: a client that drops a message self-corrects on the
        next one, and a slow client is given the newest state rather than a
        backlog of stale positions.
        """
        await _pump(ws, svc.targets, replay=False)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        """Status events, for humans via the orchestrator. Replays the recent
        backlog so a status board that connects late still has context."""
        await _pump(ws, svc.events, replay=True)

    # 4. Video ----------------------------------------------------------
    @app.get("/video")
    async def video():
        return StreamingResponse(
            streamer.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/frame.jpg")
    async def frame_jpg():
        """Single annotated frame. Easier than MJPEG to poke at with curl."""
        return StreamingResponse(iter([streamer.jpeg()]), media_type="image/jpeg")

    @app.get("/", response_class=HTMLResponse)
    async def debug():
        """Dev-only page: video, live state, and a box to send a spec.

        Not the operator UI, which lives in frontend/. This exists so the
        pipeline can be driven and watched before anything else is built.
        """
        return DEBUG_PAGE

    return app


# --- helpers -------------------------------------------------------------
def _device() -> str:
    from config import resolve_device

    return resolve_device()


def _reject(svc: Service, req: SpecRequest, reason: str) -> JSONResponse:
    """422 plus a `failed` event, so a rejection shows on the status board even
    when the caller swallowed the HTTP error."""
    log.warning("rejected %s: %s", req.instruction_id, reason)
    svc.events.publish(StatusEvent(
        stage="failed", instruction_id=req.instruction_id,
        spec_id=req.spec.spec_id, detail=reason, ts=time.time(),
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
