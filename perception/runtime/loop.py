"""The inference loop: the only thread that changes what the pipeline is doing.

One frame in, one pass through five stages, out come an annotated view, a
summary and whatever events fired. The sequence never branches on anything
expensive, so a frame costs the model plus a rounding error.

    apply ops ─▶ DETECT ─▶ TRACK ─▶ ATTRIBUTES ─▶ BEHAVIOURS ─▶ publish

Detection runs once on the union of every behaviour's classes, tracking is
shared, and attributes are scored once per track and cached. Behaviours then
filter that one result to their own subject. Two behaviours watching people
cost one forward pass, not two.

Failure is loud and safe: a dead camera or a broken model trips to FAULT and
the loop keeps publishing, because silence and failure look identical from
outside and only one of them is diagnosable.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from behaviors.base import Frame, Outcome
from config import CFG
from contracts import BehaviorStatus, FrameSummary, Phase, TargetState, Track
from runtime.capture import Capture
from runtime.events import Bus
from runtime.ops import Accepted, Applied, ModelLoaded, ModelLoading, Rejected
from runtime.state import Machine
from runtime.world import Shared, World

log = logging.getLogger("perception.loop")


@dataclass
class View:
    """Latest frame and everything found in it. Read by the renderer."""

    frame: np.ndarray | None = None
    tracks: sv.Detections | None = None
    #: behaviour id -> what that behaviour claimed this frame.
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    summary: FrameSummary | None = None
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
        if self._last:
            dt = now - self._last
            if dt > 0:
                self.fps = (1.0 / dt) if not self.fps else 0.9 * self.fps + 0.1 / dt
        self._last = now
        self.frames += 1

    def line(self) -> str:
        return f"fps={self.fps:.1f} " + " ".join(f"{k}={v:.1f}ms" for k, v in self.ewma.items())


class InferenceLoop:
    def __init__(self, capture: Capture, builder, machine: Machine,
                 state_bus: Bus, event_bus: Bus, shared: Shared | None = None) -> None:
        self.capture = capture
        self.builder = builder
        self.machine = machine
        self.state_bus = state_bus
        self.event_bus = event_bus
        self.shared = shared or Shared()

        self.world: World | None = None
        self.view = View()
        self.timings = Timings()
        self.last_summary: FrameSummary | None = None
        self.last_target: TargetState | None = None
        self._last_seq = -1
        self._last_publish = 0.0
        self._fault_source: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def behaviors(self) -> dict:
        return self.world.behaviors if self.world else {}

    # 1. Applying changes -----------------------------------------------
    def _drain(self, now: float) -> None:
        """Install whatever the builder finished. The swap point."""
        for out in self.builder.poll():
            if isinstance(out, Accepted):
                self.machine.on_spec(out.op.instruction_id, out.op.target)
            elif isinstance(out, ModelLoading):
                self.machine.on_model_loading(out.model)
            elif isinstance(out, ModelLoaded):
                log.info("cold-loaded %s in %.1fs", out.model, out.seconds)
                self.machine.on_model_loaded(out.model)
            elif isinstance(out, Rejected):
                log.warning("rejected %s: %s", out.op.describe(), out.reason)
                self.machine.on_prepare_failed(out.reason)
            elif isinstance(out, Applied):
                self._install(out, now)

    def _install(self, out: Applied, now: float) -> None:
        world = out.world
        self.machine.on_prepared()
        try:
            if world.prepared is not None:
                # The detector's cheap half. It runs here because on some models
                # it mutates the model object, which only this thread may touch.
                t0 = time.perf_counter()
                world.detector.apply(world.prepared)
                self.timings.mark("apply", time.perf_counter() - t0)
        except Exception as exc:
            log.exception("apply failed")
            self.machine.on_prepare_failed(f"apply: {exc}")
            return

        if out.reset_tracking:
            # Only a model change invalidates tracker ids and cached crops.
            # Adding a behaviour must never disturb the others.
            self.shared.reset(f"model -> {world.model_name}")
        self.shared.bank.set_texts(world.text_vectors, world.baseline_vectors)

        self.world = world
        log.info("applied %s -> %s in %.0fms", out.op.describe(), world.summary(),
                 out.seconds * 1000)
        self.machine.on_applied(now)

    # 2. One frame -------------------------------------------------------
    def step(self, now: float | None = None) -> FrameSummary | None:
        """Exactly one frame of work. Returns what was published, if anything."""
        now = time.time() if now is None else now
        self._drain(now)

        frame, _, seq = self.capture.read()
        if frame is None or self.capture.age(now) > CFG.CAMERA_DEAD_S:
            return self._camera_down(now)
        if self.machine.phase is Phase.FAULT and self._fault_source == "camera":
            self._fault_source = None
            self.machine.on_recovered(now)
        if seq == self._last_seq:
            return None  # the source is slower than we are
        self._last_seq = seq
        self.timings.tick(now)

        active = list(self.behaviors.values())
        if not active:
            self.machine.on_frame(False, now)
            return self._publish(FrameSummary(ts=now, phase=self.machine.phase,
                                              model=self._model_name,
                                              fps=self.timings.fps), frame, None, {})

        tracks = self._detect_and_track(frame, now)
        if tracks is None:
            return self._publish(FrameSummary(ts=now, phase=self.machine.phase,
                                              model=self._model_name,
                                              fps=self.timings.fps), frame, None, {})

        t0 = time.perf_counter()
        self.shared.bank.update(frame, tracks, now)
        self.timings.mark("attrs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        ctx = Frame(image=frame, tracks=tracks, bank=self.shared.bank,
                    now=now, shape=frame.shape[:2])
        outcomes, statuses = {}, []
        for b in active:
            try:
                outcomes[b.id] = b.step(ctx)
            except Exception as exc:
                # One broken behaviour must not stop the others.
                log.exception("behaviour %s failed", b.id)
                b.to("failed", str(exc))
                outcomes[b.id] = Outcome()
            statuses.append(b.status())
        self.timings.mark("behaviors", time.perf_counter() - t0)

        for outcome in outcomes.values():
            for ev in outcome.events:
                self.event_bus.publish(ev)

        self.machine.on_frame(any(s.state == "active" for s in statuses), now)
        summary = FrameSummary(ts=now, phase=self.machine.phase, model=self._model_name,
                               fps=self.timings.fps,
                               tracks=self._as_tracks(tracks, frame.shape),
                               behaviors=statuses)
        self._emit_target(now, tracks, outcomes, frame.shape)

        if self.timings.frames % CFG.TIMING_EVERY == 0:
            log.info("timings %s", self.timings.line())
        return self._publish(summary, frame, tracks, outcomes)

    def _detect_and_track(self, frame: np.ndarray, now: float) -> sv.Detections | None:
        try:
            t0 = time.perf_counter()
            dets = self.world.detector.infer(frame)
            self.timings.mark("detect", time.perf_counter() - t0)
            self.machine.on_infer_ok()
            if self.machine.phase is Phase.FAULT and self._fault_source == "infer":
                self._fault_source = None
                self.machine.on_recovered(now)
        except Exception as exc:
            log.warning("inference error: %s", exc)
            before = self.machine.phase
            if self.machine.on_infer_error(str(exc), now) is Phase.FAULT and before is not Phase.FAULT:
                self._fault_source = "infer"
            return None

        t0 = time.perf_counter()
        tracked = self.shared.track(dets)
        self.timings.mark("track", time.perf_counter() - t0)
        return tracked

    # 3. Publishing -------------------------------------------------------
    @property
    def _model_name(self) -> str | None:
        return self.world.model_name if self.world else None

    def _as_tracks(self, dets: sv.Detections, shape) -> list[Track]:
        """Normalized view of the frame, for queries and the UI."""
        h, w = shape[:2]
        names, ids, conf = dets.data.get("class_name"), dets.tracker_id, dets.confidence
        out = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            tid = int(ids[i]) if ids is not None else -1
            out.append(Track(
                track_id=tid,
                label=str(names[i]) if names is not None else "?",
                conf=float(conf[i]) if conf is not None else 0.0,
                cx=float((x1 + x2) / 2 / w * 2 - 1),
                cy=float((y1 + y2) / 2 / h * 2 - 1),
                area=float((x2 - x1) * (y2 - y1) / (w * h)),
                attributes=self.shared.bank.scores_for(tid),
            ))
        return out

    def _emit_target(self, now: float, tracks, outcomes, shape) -> None:
        """A TargetState for whichever behaviour is following one thing.

        This is what the actuator reads: guidance arrows now, a stepper later.
        Only `track` behaviours set `chosen`, so this is empty until one runs.
        """
        for bid, outcome in outcomes.items():
            if outcome.chosen is None:
                continue
            b = self.behaviors[bid]
            h, w = shape[:2]
            x1, y1, x2, y2 = tracks.xyxy[outcome.chosen]
            ids, conf = tracks.tracker_id, tracks.confidence
            names = tracks.data.get("class_name")
            self.last_target = TargetState(
                ts=now, spec_id=bid, mode="follow", visible=True,
                cx=float((x1 + x2) / 2 / w * 2 - 1),
                cy=float((y1 + y2) / 2 / h * 2 - 1),
                area=float((x2 - x1) * (y2 - y1) / (w * h)),
                conf=float(conf[outcome.chosen]) if conf is not None else 0.0,
                label=str(names[outcome.chosen]) if names is not None else None,
                track_id=int(ids[outcome.chosen]) if ids is not None else None,
                phase=self.machine.phase, model=self._model_name, target_ref=b.id,
            )
            return
        if self.last_target is not None:
            self.last_target = self.last_target.model_copy(
                update={"ts": now, "visible": False, "phase": self.machine.phase})

    def _camera_down(self, now: float) -> FrameSummary | None:
        """Keep talking while broken, at a heartbeat rate."""
        if self.machine.phase is not Phase.FAULT:
            self._fault_source = "camera"
            self.machine.on_fault(f"no frame for {self.capture.age(now):.1f}s", now)
        if now - self._last_publish < 1.0 / CFG.FAULT_HEARTBEAT_HZ:
            return None
        return self._publish(FrameSummary(ts=now, phase=Phase.FAULT,
                                          model=self._model_name), None, None, {})

    def _publish(self, summary: FrameSummary, frame, tracks, outcomes) -> FrameSummary:
        self.last_summary = summary
        self._last_publish = summary.ts
        self.view = View(frame=frame, tracks=tracks, outcomes=outcomes,
                         summary=summary, seq=self._last_seq)
        self.state_bus.publish(summary)
        return summary

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
        log.info("inference loop started")
        while not self._stop.is_set():
            if self.step() is None:
                self._stop.wait(0.002)  # capture sets the real pace
