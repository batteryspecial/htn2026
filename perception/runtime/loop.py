"""The frame loop: the only thread that changes what the pipeline is doing.

One frame in, one pass, out come an enriched frame, a state message and
whatever events fired:

    ops -> DETECT -> TRACK -> ATTRIBUTES -> BEHAVIOURS -> ACTUATOR -> RENDER

Detection runs once on the union of every behaviour's classes, tracking is
shared, and attributes are scored once per track and cached. Behaviours then
filter that one result to their own selector, so two behaviours watching people
cost one forward pass.

Behaviours never draw. They return layers; the renderer composes them here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from actuator.base import Actuator
from behaviors.base import Frame, Outcome
from config import CFG
from contracts import BehaviorView, StateView, TrackView
from render.hud import Hud
from render.layers import Layer
from runtime.capture import Capture
from runtime.events import Bus
from runtime.health import Health
from runtime.ops import Accepted, Applied, ModelLoaded, ModelLoading, Rejected
from runtime.world import Shared, World

log = logging.getLogger("perception.loop")


@dataclass
class View:
    """Latest frame plus the layers to draw on it. Read by the streamer."""

    frame: np.ndarray | None = None
    layers: list[Layer] = field(default_factory=list)
    state: StateView | None = None
    seq: int = 0


@dataclass
class Timings:
    """Rolling per-stage cost, so the slow stage is found rather than guessed."""

    ewma: dict[str, float] = field(default_factory=dict)
    frames: int = 0
    fps: float = 0.0
    _last: float = 0.0

    def mark(self, stage: str, seconds: float, alpha: float = 0.1) -> None:
        ms = seconds * 1000.0
        self.ewma[stage] = ms if stage not in self.ewma else (
            (1 - alpha) * self.ewma[stage] + alpha * ms)

    def tick(self, now: float) -> None:
        if self._last and now > self._last:
            dt = now - self._last
            self.fps = (1.0 / dt) if not self.fps else 0.9 * self.fps + 0.1 / dt
        self._last = now
        self.frames += 1

    def line(self) -> str:
        return f"fps={self.fps:.1f} " + " ".join(f"{k}={v:.1f}ms" for k, v in self.ewma.items())


class InferenceLoop:
    def __init__(self, capture: Capture, builder, health: Health,
                 state_bus: Bus, event_bus: Bus, shared: Shared | None = None,
                 actuator: Actuator | None = None, registry=None) -> None:
        self.capture = capture
        self.builder = builder
        self.health = health
        self.state_bus = state_bus
        self.event_bus = event_bus
        self.shared = shared or Shared()
        self.actuator = actuator
        self.registry = registry or getattr(builder, "registry", None)

        self.world: World | None = None
        self.view = View()
        self.timings = Timings()
        self.last_state: StateView | None = None
        #: Raw tracked detections from the last frame. The published state is
        #: normalized; review tools need the pixels.
        self.last_tracks: sv.Detections | None = None
        self.hud_text: str | None = None
        #: event id -> JPEG. Served by /snapshots/{id}.jpg so the agent can
        #: look at what fired. Bounded: a live stream never ends.
        self.snapshots: OrderedDict[str, bytes] = OrderedDict()
        self._last_seq = -1
        self._last_publish = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def behaviors(self) -> dict:
        return self.world.behaviors if self.world else {}

    @property
    def model_name(self) -> str | None:
        return self.world.model_name if self.world else None

    # 1. Applying changes -----------------------------------------------
    def _drain(self, now: float) -> None:
        for out in self.builder.poll():
            if isinstance(out, Accepted):
                continue  # the id was handed back synchronously
            if isinstance(out, ModelLoading):
                log.info("cold-loading %s", out.model)
            elif isinstance(out, ModelLoaded):
                log.info("cold-loaded %s in %.1fs", out.model, out.seconds)
            elif isinstance(out, Rejected):
                log.warning("rejected %s: %s", out.op.describe(), out.reason)
                self.health.event("behavior_failed", out.reason,
                                  op=out.op.describe(),
                                  behavior_id=out.op.behavior_id)
            elif isinstance(out, Applied):
                self._install(out, now)

    def _install(self, out: Applied, now: float) -> None:
        world = out.world
        try:
            if world.prepared is not None:
                # The detector's cheap half. It runs here because on some
                # models it mutates the model object, which only this thread
                # may touch.
                t0 = time.perf_counter()
                world.detector.apply(world.prepared)
                self.timings.mark("apply", time.perf_counter() - t0)
        except Exception as exc:
            log.exception("apply failed; keeping the running world")
            self.health.event("behavior_failed", f"apply failed: {exc}")
            return

        if out.reset_tracking:
            self.shared.reset(f"model -> {world.model_name}")
        self.shared.bank.set_texts(world.text_vectors, world.baseline_vectors)
        self.shared.bank.set_references(world.ref_vectors)
        self.world = world
        log.info("applied %s -> %s in %.0fms", out.op.describe(), world.summary(),
                 out.seconds * 1000)
        if out.op.kind == "set_model":
            paused = [b.id for b in world.behaviors.values() if b.state == "PAUSED"]
            self.health.model_switched(world.model_name, self.timings.fps, paused)

    # 2. One frame -------------------------------------------------------
    def step(self, now: float | None = None) -> StateView | None:
        """Exactly one frame of work. Returns what was published, if anything."""
        now = time.time() if now is None else now
        self._drain(now)

        frame, _, seq = self.capture.read()
        if frame is None or self.capture.age(now) > CFG.CAMERA_DEAD_S:
            return self._camera_down(now)
        self.health.camera(True)
        if seq == self._last_seq:
            return None  # the source is slower than we are
        self._last_seq = seq
        self.shared.frame_area = frame.shape[0] * frame.shape[1]
        self.timings.tick(now)

        active = [b for b in self.behaviors.values() if b.state != "PAUSED"]
        tracks = self._detect_and_track(frame, now) if self.world else None
        if tracks is None:
            return self._publish(self._state(now, []), frame, [])

        self.last_tracks = tracks
        t0 = time.perf_counter()
        self.shared.bank.update(frame, tracks, now)
        self.timings.mark("attrs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        layers, motor = self._run_behaviors(frame, tracks, now, active)
        self.timings.mark("behaviors", time.perf_counter() - t0)

        if self.actuator is not None:
            layers.extend(self.actuator.command(motor, frame.shape[:2], now))

        if self.timings.frames % CFG.TIMING_EVERY == 0:
            log.info("timings %s", self.timings.line())
        return self._publish(self._state(now, self._tracks(tracks, frame.shape)),
                             frame, layers)

    def _run_skills(self, frame: np.ndarray) -> dict:
        """Run aux models, but only the ones some behaviour actually wants.

        A pipeline with no gesture behaviour pays nothing for one being
        possible. A failure here degrades that behaviour, never the frame.
        """
        wanted = self.world.roles_needed() if self.world else set()
        if not wanted or self.registry is None:
            return {}
        out = {}
        for role in wanted:
            model = self.registry.try_get_active(role)
            if model is None:
                continue
            try:
                t0 = time.perf_counter()
                if role in ("pose", "hands", "wholebody"):
                    # Both are keypoint models — (N, K, 3) in frame pixels,
                    # only K differs — so everything downstream is shared.
                    # ponytail: `ocr` will not be, and gets its own branch.
                    out[role] = model.keypoints(frame)
                self.timings.mark(role, time.perf_counter() - t0)
            except Exception:
                log.exception("%s model failed", role)
        return out

    def _run_behaviors(self, frame, tracks, now, active) -> tuple[list[Layer], object]:
        ctx = Frame(image=frame, tracks=tracks, bank=self.shared.bank, now=now,
                    shape=frame.shape[:2], trails=self.shared.paths(),
                    skills=self._run_skills(frame))
        layers: list[Layer] = []
        motor, motor_started = None, -1.0
        for b in active:
            try:
                outcome: Outcome = b.step(ctx)
            except Exception as exc:
                # One broken behaviour must not stop the others.
                log.exception("behaviour %s failed", b.id)
                b.to("FAILED", str(exc))
                continue
            layers.extend(outcome.layers)
            for event_id, crop in outcome.snapshots:
                self._keep_snapshot(event_id, crop)
            for ev in outcome.events:
                self.event_bus.publish(ev)
            # Only the most recently started behaviour steers, so starting a
            # new "follow that" takes the arrows over from the old one.
            if outcome.motor is not None and b.started > motor_started:
                motor, motor_started = outcome.motor, b.started
        return layers, motor

    def _keep_snapshot(self, event_id: str, crop: np.ndarray, limit: int = 64) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        self.snapshots[event_id] = buf.tobytes()
        while len(self.snapshots) > limit:
            self.snapshots.popitem(last=False)

    def _detect_and_track(self, frame: np.ndarray, now: float) -> sv.Detections | None:
        try:
            t0 = time.perf_counter()
            dets = self.world.detector.infer(frame)
            self.timings.mark("detect", time.perf_counter() - t0)
            self.health.infer_ok()
        except Exception as exc:
            log.warning("inference error: %s", exc)
            if self.health.infer_error(str(exc)):
                self.health.event("camera_lost", f"inference failing: {exc}")
            return None

        t0 = time.perf_counter()
        tracked = self.shared.track(dets)
        self.timings.mark("track", time.perf_counter() - t0)
        return tracked

    # 3. Publishing -------------------------------------------------------
    def _tracks(self, dets: sv.Detections, shape) -> list[TrackView]:
        h, w = shape[:2]
        names, ids, conf = dets.data.get("class_name"), dets.tracker_id, dets.confidence
        out = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            tid = int(ids[i]) if ids is not None else -1
            out.append(TrackView(
                track_id=tid,
                label=str(names[i]) if names is not None else "?",
                conf=float(conf[i]) if conf is not None else 0.0,
                cx=float((x1 + x2) / 2 / w * 2 - 1),
                cy=float((y1 + y2) / 2 / h * 2 - 1),
                area=float((x2 - x1) * (y2 - y1) / (w * h)),
                attributes=self.shared.bank.scores_for(tid),
            ))
        return out

    def _state(self, now: float, tracks: list[TrackView]) -> StateView:
        return StateView(
            ts=now, model=self.model_name, fps=self.timings.fps,
            camera_ok=self.health.camera_ok, hud=self.hud_text,
            behaviors=[b.view() for b in self.behaviors.values()], tracks=tracks,
            refs=list(getattr(self.builder, "references", []).ids())
            if hasattr(getattr(self.builder, "references", None), "ids") else [],
        )

    def _camera_down(self, now: float) -> StateView | None:
        """Keep talking while broken. Silence and failure look identical from
        outside, and only one of them is diagnosable."""
        self.health.camera(False, f"no frame for {self.capture.age(now):.1f}s")
        if now - self._last_publish < 1.0 / CFG.FAULT_HEARTBEAT_HZ:
            return None
        return self._publish(self._state(now, []), None, [])

    def _publish(self, state: StateView, frame, layers: list[Layer]) -> StateView:
        self.last_state = state
        self._last_publish = state.ts
        layers = list(layers)
        layers.append(Hud(model=state.model, fps=state.fps, text=self.hud_text,
                          chips=[b.chip() for b in self.behaviors.values()],
                          camera_ok=state.camera_ok))
        self.view = View(frame=frame, layers=layers, state=state, seq=self._last_seq)
        self.state_bus.publish(state)
        return state

    # 4. Thread -----------------------------------------------------------
    def start(self) -> InferenceLoop:
        self._thread = threading.Thread(target=self.run, name="inference", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def run(self) -> None:
        log.info("frame loop started")
        while not self._stop.is_set():
            if self.step() is None:
                self._stop.wait(0.002)  # capture sets the real pace
