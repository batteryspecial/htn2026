"""HTTP and WebSocket surface. Port 8001.

Shaped by who is on the other end. The **agent** starts and stops behaviours
and asks questions; behaviour changes return immediately and are applied in
the background. Reference uploads wait for their encoder result so a returned
id is already usable. The **frontend** watches the enriched feed and a state
channel.

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

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from behaviors.kinds import BUILT, KINDS
from behaviors.base import Behavior, Frame
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
from runtime.cameras import CameraChoice
from runtime.capture import Capture
from runtime.events import Bus
from runtime.health import Health
from runtime.loop import InferenceLoop
from runtime.ops import Rejected
from runtime.workers import Builder
from server.debug_page import DEBUG_PAGE
from skills import gestures as gesture_vocab

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

    @app.get("/behaviors/{behavior_id}/status")
    async def behavior_status(behavior_id: str):
        """Return accepted/pending/installed/failed for one behavior id."""
        status = svc.loop.operation_status(behavior_id)
        if status is None:
            if behavior_id in svc.loop.behaviors:
                return {"id": behavior_id, "status": "installed", "detail": None}
            return JSONResponse(status_code=404,
                                content={"detail": f"unknown behaviour {behavior_id!r}"})
        return status

    @app.get("/operations/{operation_id}")
    async def operation_status(operation_id: str):
        status = svc.loop.operation_status(operation_id)
        if status is None:
            return JSONResponse(status_code=404,
                                content={"detail": f"unknown operation {operation_id!r}"})
        return status

    @app.delete("/behaviors/{behavior_id}", status_code=202)
    async def remove_behavior(behavior_id: str):
        if behavior_id not in svc.loop.behaviors:
            return JSONResponse(status_code=404,
                                content={"detail": f"no behaviour {behavior_id!r}"})
        op = svc.builder.remove_behavior(behavior_id)
        return {"removed": behavior_id, "operation_id": f"op-{op.seq}"}

    @app.delete("/behaviors", status_code=202)
    async def clear_behaviors():
        op = svc.builder.clear()
        return {"cleared": True, "operation_id": f"op-{op.seq}"}

    @app.post("/behaviors/{behavior_id}/advance", status_code=202)
    async def advance_behavior(behavior_id: str):
        """Step a `keyboard` in `step_mode: manual` to the next key.

        A counter bump, not a mutation: the loop reads and clears it on the
        next frame, so it stays the only thing that changes what is drawn.
        """
        behavior = svc.loop.behaviors.get(behavior_id)
        if behavior is None:
            return JSONResponse(status_code=404,
                                content={"detail": f"no behaviour {behavior_id!r}"})
        if not hasattr(behavior, "advance"):
            return _reject(f"{behavior.kind!r} does not step; "
                           f"only 'keyboard' in step_mode 'manual' does")
        behavior.advance()
        return {"advanced": behavior_id}

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
        op = svc.builder.set_model(choice.name)
        return {"accepted": True, "model": choice.name, "role": "detector",
                "operation_id": f"op-{op.seq}"}

    @app.get("/models")
    async def models():
        """What this machine can do. The agent's device manifest.

        `roles` is the quick answer to "is a gesture trigger possible at all"
        without reading the whole model list. `gestures` is the next question
        down — *which* ones — and it has to be reported rather than written
        into the agent's prompt, because the vocabulary is code: whatever
        `skills/` implements, gated by which aux models are loaded here.

        It spans roles. A body gesture needs `pose` and a finger gesture needs
        `hands`, so this list narrows to what this machine can actually
        observe: a rule whose model is missing is not a capability, and
        reporting it would have the agent install a behaviour that can never
        fire.
        """
        return {"models": svc.registry.manifest(), "roles": svc.registry.roles(),
                "gestures": gesture_vocab.available(svc.registry.has_role),
                "device": resolve_device()}

    # 2b. Camera ---------------------------------------------------------
    #
    # Same shape as the model swap above: GET the manifest, POST the choice.
    # Enumeration is server-side because the camera is opened here, by
    # OpenCV, in a thread. The browser's own device list names cameras this
    # process cannot open and indexes them differently.

    @app.get("/cameras")
    async def cameras(refresh: bool = False):
        """Every camera this machine offers, and which one is live.

        `refresh=1` re-scans and opens each idle device to read its real
        resolution. That is disruptive — opening a device already in use can
        steal it — so the active camera is never probed; its numbers come
        from the live capture instead.
        """
        from starlette.concurrency import run_in_threadpool

        import runtime.cameras as cameras_mod

        source = svc.capture.source
        found = await run_in_threadpool(
            cameras_mod.list_cameras, source, refresh=refresh)

        listed = [c.as_dict() for c in found]
        for entry in listed:
            if entry["active"]:
                # What the driver actually gave us, not what was requested.
                entry.update(width=svc.capture.width, height=svc.capture.height,
                             fps=svc.capture.fps, available=svc.capture.opened,
                             fourcc=svc.capture.fourcc)

        if not any(e["active"] for e in listed):
            # A file or a stream URL is a perfectly good source and is not in
            # any device list. Say what is running rather than showing nothing.
            listed.append({"index": -1, "name": source, "backend": "file",
                           "active": True, "available": svc.capture.opened,
                           "width": svc.capture.width, "height": svc.capture.height,
                           "fps": svc.capture.fps, "detail": "not a camera device"})

        return {"cameras": listed, "source": source,
                "backend": cameras_mod.backend_name()}

    @app.post("/camera", status_code=202)
    async def set_camera(choice: CameraChoice):
        """Point the pipeline at a different camera.

        Behaviours are left alone: this is a change of sensor, not a change of
        objective. Tracking is reset, because track ids from the old lens can
        never reappear on the new one.
        """
        from starlette.concurrency import run_in_threadpool

        import runtime.cameras as cameras_mod

        source, detail = await run_in_threadpool(
            cameras_mod.resolve, choice.source, choice.name)
        if source is None:
            return _reject(detail)
        if source == svc.capture.source:
            return _reject(f"already reading {detail}")

        ok, result = await run_in_threadpool(svc.capture.switch, source)
        cameras_mod.invalidate()
        if not ok:
            return _reject(result)

        svc.loop.request_reset(f"camera -> {detail}")
        # No event is emitted here on purpose. Frames stop and restart, so the
        # loop's own camera_lost / camera_ok pair already tells the story, and
        # "camera_switched" is not in the shared EventType — adding it is a
        # change to linker/schemas.py, which is owned jointly.
        log.info("camera switched to %s (%s)", detail, source)
        return {"accepted": True, "source": source, "name": detail,
                "width": svc.capture.width, "height": svc.capture.height,
                "fps": svc.capture.fps}

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
    async def query_count(q: CountQuery):
        """Median over a window, so one bad frame cannot change the answer."""
        try:
            counts = await _sample(svc, q.selector, q.window_s)
            return CountResult(count=int(statistics.median(counts)) if counts else 0,
                               samples=len(counts), window_s=q.window_s)
        except QueryUnavailable as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.post("/query/look")
    async def query_look(q: LookQuery):
        """What is in front of the camera right now, with attribute scores."""
        try:
            state = svc.loop.last_state
            tracks = (await _query_tracks(svc, q.selector)
                      if q.selector else list(state.tracks if state else []))
            return LookResult(tracks=tracks)
        except QueryUnavailable as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})

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
            return JSONResponse(status_code=503,
                                content={"detail": "no frame to look at"})
        try:
            raw = await run_in_threadpool(
                scene_mod.probe, svc.registry, frame, req.phrases, frame.shape)
        except Exception as exc:
            log.exception("probe failed")
            return JSONResponse(status_code=500,
                                content={"detail": f"probe failed: {exc}"})
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
    async def add_reference(request: Request, file: UploadFile | None = File(default=None),
                            label: str | None = Form(default=None)):
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
            failure = await _reference_failure(svc.builder, ref_id)
            if failure:
                return failure
            return {"ref_id": ref_id, "thumb_url": f"/references/{ref_id}.jpg"}

        body = {}
        if request.headers.get("content-type", "").split(";", 1)[0] == "application/json":
            try:
                body = await request.json()
            except ValueError:
                return _reject("invalid reference JSON")
            if not isinstance(body, dict):
                return _reject("reference JSON must be an object")
            label = body.get("label") or label
        source = body.get("from", "largest_person")
        vector, detail = _embedding_from_frame(svc, source)
        if vector is None:
            return _reject(detail)
        ref_id = svc.builder.add_reference(vector=vector, label=label or source)
        failure = await _reference_failure(svc.builder, ref_id)
        if failure:
            return failure
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
class QueryUnavailable(RuntimeError):
    pass


async def _query_tracks(svc: Service, selector) -> list:
    """Evaluate the complete Selector or fail explicitly."""
    state = svc.loop.last_state
    frame = svc.loop.view.frame
    if not state or not state.camera_ok or frame is None:
        raise QueryUnavailable("no current camera frame")

    if selector.ref_id and selector.ref_id not in set(state.refs):
        raise QueryUnavailable(f"unknown reference {selector.ref_id!r}")

    world = svc.loop.world
    active_prompts = set(world.prompt_union(world.behaviors.values())) if world else set()
    active_attributes = set(svc.loop.shared.bank.phrases)
    covered = (set(selector.prompts()) <= active_prompts
               and set(selector.attribute_texts()) <= active_attributes)

    if covered and svc.loop.last_tracks is not None:
        spec = BehaviorSpec(kind="highlight", subject=selector, notify=False)
        query = Behavior("__query__", spec)
        raw = svc.loop.last_tracks
        query_frame = Frame(
            image=frame, tracks=raw, bank=svc.loop.shared.bank,
            now=state.ts, shape=frame.shape[:2], trails=svc.loop.shared.paths(),
        )
        indices = query.filter(query_frame)
        if selector.pick != "all":
            chosen = query.choose(query_frame, indices)
            indices = [chosen] if chosen is not None else []
        normalized = svc.loop._tracks(raw, frame.shape)
        return [normalized[int(i)] for i in indices]

    if selector.include or selector.exclude or selector.ref_id or selector.relate:
        raise QueryUnavailable(
            "the requested selector is not prepared in the live world; install "
            "a behavior with that selector before querying it"
        )

    from starlette.concurrency import run_in_threadpool
    import scene as scene_mod
    try:
        tracks = await run_in_threadpool(
            scene_mod.sweep, svc.registry, frame, selector.detect, frame.shape)
    except Exception as exc:
        raise QueryUnavailable(f"query detector unavailable: {exc}") from exc
    if selector.pick == "largest" and tracks:
        tracks = [max(tracks, key=lambda item: item.area)]
    elif selector.pick == "most_centered" and tracks:
        tracks = [min(tracks, key=lambda item: item.cx * item.cx + item.cy * item.cy)]
    elif selector.pick == "ref":
        raise QueryUnavailable("pick 'ref' requires a prepared reference selector")
    return tracks


async def _sample(svc: Service, selector, window_s: float) -> list[int]:
    """Count complete selector matches over a window of published frames."""
    deadline = time.time() + window_s
    counts, seen = [], None
    while time.time() < deadline:
        state = svc.loop.last_state
        if state is not None and state.ts != seen:
            seen = state.ts
            counts.append(len(await _query_tracks(svc, selector)))
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


async def _reference_failure(builder: Builder, ref_id: str) -> JSONResponse | None:
    """References are usable when their 201 is returned, or fail explicitly."""
    result = await asyncio.to_thread(builder.wait_reference, ref_id)
    if result is None:
        return JSONResponse(
            status_code=504,
            content={"detail": "reference encoding timed out; try the image again"},
        )
    if isinstance(result, Rejected):
        return _reject(result.reason)
    return None


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
