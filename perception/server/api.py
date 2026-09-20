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
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from behaviors.kinds import BUILT, KINDS
from config import CFG, resolve_device
from contracts import (
    BehaviorCreated,
    BehaviorSpec,
    CountQuery,
    CountResult,
    DescribeRequest,
    DescribeResult,
    Health as HealthView,
    HudText,
    LookQuery,
    LookResult,
    PhraseResult,
    ProbeRequest,
    ProbeResult,
    ModelChoice,
)
from zoo.registry import Registry
from runtime.capture import Capture
from runtime.events import Bus
from runtime.health import Health
from runtime.loop import InferenceLoop
from runtime.workers import Builder
from server.debug_page import DEBUG_PAGE
from skills.pose import GESTURES

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
    # The operator UI is served from somewhere else entirely: a file:// page, a
    # static host, or another laptop. Everything here is read-mostly control of
    # a camera on a LAN for one demo, so the permissive setting is the honest
    # one rather than a guess at the right origin on the day.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
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
        bad = _vocab_error(svc, spec) or _params_error(spec)
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
        """Swap the model filling a role. Defaults to the detector.

        Behaviours the new model cannot serve are paused with a reason and
        resume when one that can returns.
        """
        try:
            entry = svc.registry.entry(choice.name)
        except KeyError:
            return _reject(f"unknown model {choice.name!r}; "
                           f"available: {svc.registry.names()}")
        if not entry.available:
            return _reject(f"model {choice.name!r} unavailable: {entry.unavailable_reason}")
        if entry.role != "detector":
            # Aux roles are swapped in place: nothing about the frame loop or
            # the tracker depends on which pose model is active.
            svc.registry.set_active(entry.role, choice.name)
            return {"accepted": True, "model": choice.name, "role": entry.role}
        svc.builder.set_model(choice.name)
        return {"accepted": True, "model": choice.name, "role": "detector"}

    @app.get("/models")
    async def models():
        """What this machine can do. The agent's device manifest.

        `roles` is the quick answer to "is a gesture trigger possible at all"
        without reading the whole model list. `gestures` is the next question
        down — *which* ones — and it has to be reported rather than written
        into the agent's prompt, because the vocabulary is code: it is whatever
        `skills/pose.py` implements, and it grows when a pose model does.

        Empty when no pose model is loaded: a gesture the device cannot
        observe is not a capability, and the agent needs to tell "no gestures
        at all" apart from "not that gesture".
        """
        return {"models": svc.registry.manifest(), "roles": svc.registry.roles(),
                "gestures": sorted(GESTURES) if svc.registry.has_role("pose") else [],
                "device": resolve_device()}

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

    @app.post("/probe")
    async def probe(req: ProbeRequest) -> ProbeResult:
        """Does this wording actually find anything, right now?

        The failure that looks like a broken pipeline and is really a word the
        detector has no match for. Measured here: "duck" finds nothing in
        frames where "yellow duck" finds it every time, and the same phrase
        stops working in a different room. Rather than guessing, ask.

        Runs on the spare detector, so probing never disturbs tracking.
        """
        from starlette.concurrency import run_in_threadpool

        import scene as scene_mod

        frame = svc.loop.view.frame
        if frame is None:
            return ProbeResult(detail="no frame to look at")
        try:
            raw = await run_in_threadpool(
                scene_mod.probe, svc.registry, frame, req.phrases, frame.shape)
        except Exception as exc:
            return ProbeResult(detail=str(exc))
        best, advice = scene_mod.probe_advice(raw)
        return ProbeResult(results=[PhraseResult(**r) for r in raw],
                           best=best, advice=advice)

    @app.post("/describe")
    async def describe(req: DescribeRequest) -> DescribeResult:
        """Everything needed to answer a question about the scene.

        Perception does not call a vision model — it has no LLM and should not
        grow one. This returns the frame, what the detector can actually see,
        and a prompt built from both; the agent supplies the model.

        With `sweep`, a dedicated detector takes one pass over a broad everyday
        vocabulary, because an open-vocabulary detector only finds what it is
        named and an open question names nothing. Without it, only what the
        running behaviours already track is reported.
        """
        from starlette.concurrency import run_in_threadpool

        import scene as scene_mod

        state = svc.loop.last_state
        behaviors = [b.label or b.kind for b in (state.behaviors if state else [])
                     if b.state != "PAUSED"]
        tracks = list(state.tracks) if state else []
        swept, detail = False, None

        if req.sweep:
            frame = svc.loop.view.frame
            if frame is None:
                detail = "no frame to look at"
            else:
                candidates = scene_mod.candidates_for(req.candidates)
                try:
                    # Off the event loop: this is a model call, and a query
                    # that blocks the server also blocks the video feed.
                    tracks = await run_in_threadpool(
                        scene_mod.sweep, svc.registry, frame, candidates,
                        frame.shape)
                    swept = True
                except Exception as exc:
                    # Fall back to what the behaviours already see rather than
                    # failing the question outright. The message, not the
                    # exception type: the agent reads this and may act on it.
                    log.warning("scene sweep failed: %s", exc)
                    detail = f"{exc}; reporting tracked objects only"

        objects = scene_mod.to_objects(tracks)
        return DescribeResult(
            snapshot_url="/snapshot", objects=objects,
            summary=scene_mod.summarize(objects),
            prompt=scene_mod.build_prompt(req.question, objects, behaviors),
            behaviors=behaviors, swept=swept, detail=detail,
        )

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
        data = svc.loop.snapshots.get(event_id)
        if not data:
            return JSONResponse(status_code=404, content={"detail": "no such snapshot"})
        return Response(content=data, media_type="image/jpeg")

    # 4b. References ------------------------------------------------------
    @app.post("/references", status_code=201)
    async def add_reference(file: UploadFile | None = File(default=None),
                            label: str | None = Form(default=None),
                            body: dict | None = None):
        """Register an appearance to match against: "track *that* one".

        Either an uploaded photo, or `{"from": "largest_person"}` to take the
        biggest thing on screen right now, which is how "follow him" works
        with nothing to upload.
        """
        import cv2
        import numpy as np

        if file is not None:
            raw = np.frombuffer(await file.read(), np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                return _reject("could not decode that image")
            ref_id = svc.builder.add_reference(image=image, label=label or file.filename)
            return {"ref_id": ref_id, "thumb_url": f"/references/{ref_id}.jpg"}

        source = (body or {}).get("from", "largest_person")
        vector, detail = _embedding_from_frame(svc, source)
        if vector is None:
            return _reject(detail)
        ref_id = svc.builder.add_reference(vector=vector, label=label or source)
        return {"ref_id": ref_id, "thumb_url": f"/references/{ref_id}.jpg"}

    @app.get("/references")
    async def list_references():
        store = svc.builder.references
        return {"references": [
            {"ref_id": r.ref_id, "label": r.label,
             "thumb_url": f"/references/{r.ref_id}.jpg"}
            for r in store.items.values()]}

    @app.get("/references/{ref_id}.jpg")
    async def reference_thumb(ref_id: str):
        ref = svc.builder.references.items.get(ref_id)
        if ref is None or ref.thumb is None:
            return JSONResponse(status_code=404, content={"detail": "no such reference"})
        return Response(content=ref.thumb, media_type="image/jpeg")

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


def _embedding_from_frame(svc: Service, source: str):
    """Take an appearance straight off the current frame.

    The embedding already exists: the attribute cache computes one per track
    for re-identification, so "that one there" costs a lookup, not a model
    call.
    """
    tracks = svc.loop.last_tracks
    if tracks is None or len(tracks) == 0:
        return None, "nothing on screen to reference"
    wanted = source.replace("largest_", "").replace("_", " ")
    names = tracks.data.get("class_name")
    idx = [i for i in range(len(tracks))
           if names is None or wanted in ("", str(names[i]))]
    if not idx:
        return None, f"nothing matching {wanted!r} on screen"
    import numpy as np

    best = int(max(idx, key=lambda i: float(tracks.box_area[i])))
    ids = tracks.tracker_id
    if ids is None:
        return None, "tracks have no ids yet"
    entry = svc.loop.shared.bank.get(int(ids[best]))
    if entry is None or entry.embedding is None:
        return None, "that track has not been scored yet; is CLIP loaded?"
    return np.asarray(entry.embedding), ""


def _params_error(spec: BehaviorSpec) -> str | None:
    """Check kind-specific params up front by building the behaviour and
    throwing it away.

    Construction is pure and costs microseconds, and without this a malformed
    `params` is accepted with a 201 and only fails later as an event. An agent
    that has to notice an event to learn its call was wrong will not notice.
    """
    from behaviors.kinds import build

    try:
        build("preflight", spec)
    except Exception as exc:
        return str(exc)
    return None


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
