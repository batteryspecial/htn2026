"""The inference loop: the only thread allowed to change what the pipeline is doing.

One frame in, one TargetState out, every time. The sequence never branches on
anything expensive, so the cost of a frame is the cost of the model and nothing
else. Two things arrive from elsewhere and both are picked up at the top of a
frame, never in the middle of one:

1. outcomes from the loader worker, which is how a retask lands
2. frames from the capture thread, which is how the world gets in

Failure is designed to be loud and safe rather than clever: a dead camera or a
broken model trips to FAULT, the car is told to stop, and the loop keeps
publishing so the controller can tell "broken" apart from "disconnected".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from config import CFG
from contracts import Phase, TargetState
from runtime.capture import Capture
from runtime.events import Bus
from runtime.loader import (Accepted, Cleared, Failed, Loader, ModelLoaded,
                            ModelLoading, Prepared)
from runtime.state import Machine
from runtime.task import PreparedTask
from stages.select import candidates_for, pick

log = logging.getLogger("perception.loop")


@dataclass
class View:
    """Latest frame plus what was found in it. Read by the MJPEG stream."""

    frame: np.ndarray | None = None
    detections: sv.Detections | None = None
    state: TargetState | None = None
    chosen: int | None = None
    seq: int = 0


@dataclass
class Timings:
    """Rolling per-stage cost. The point is to find out which stage is slow
    before optimizing the one we assumed was."""

    ewma: dict[str, float] = field(default_factory=dict)
    frames: int = 0
    fps: float = 0.0
    _last: float = 0.0

    def mark(self, stage: str, seconds: float, alpha: float = 0.1) -> None:
        ms = seconds * 1000.0
        self.ewma[stage] = ms if stage not in self.ewma else (
            (1 - alpha) * self.ewma[stage] + alpha * ms
        )

    def tick(self, now: float) -> None:
        if self._last:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
        self._last = now
        self.frames += 1

    def line(self) -> str:
        parts = " ".join(f"{k}={v:.1f}ms" for k, v in self.ewma.items())
        return f"fps={self.fps:.1f} {parts}"


class InferenceLoop:
    def __init__(self, capture: Capture, loader: Loader, machine: Machine,
                 target_bus: Bus, event_bus: Bus) -> None:
        self.capture = capture
        self.loader = loader
        self.machine = machine
        self.target_bus = target_bus
        self.event_bus = event_bus

        self.active: PreparedTask | None = None
        self.view = View()
        self.timings = Timings()
        self.last_state: TargetState | None = None
        # Which failure put us in FAULT, so the right signal clears it.
        self._fault_source: str | None = None
        self._last_seq = -1
        self._last_publish = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # 1. Retask ----------------------------------------------------------
    def _drain_loader(self, now: float) -> None:
        """Pick up whatever the worker finished. This is the swap point."""
        for outcome in self.loader.poll():
            if isinstance(outcome, Accepted):
                self.machine.on_spec(outcome.job.instruction_id, outcome.job.spec.spec_id)
            elif isinstance(outcome, Cleared):
                self.active = None
                self._fault_source = None
                self.machine.on_clear_spec()
            elif isinstance(outcome, ModelLoading):
                self.machine.on_model_loading(outcome.model)
            elif isinstance(outcome, ModelLoaded):
                log.info("cold-loaded %s in %.1fs", outcome.model, outcome.seconds)
                self.machine.on_model_loaded(outcome.model)
            elif isinstance(outcome, Failed):
                if outcome.reason == "superseded":
                    self.machine.on_superseded(outcome.job.instruction_id,
                                               outcome.job.spec.spec_id)
                else:
                    self.machine.on_prepare_failed(outcome.reason)
            elif isinstance(outcome, Prepared):
                self._swap(outcome, now)

    def _swap(self, outcome: Prepared, now: float) -> None:
        """Install a finished task. The one place the active task changes.

        `apply` is the detector's cheap half; it runs here because on some
        models it mutates the model object, which only this thread may touch.
        """
        task = outcome.task
        self.machine.on_prepared()
        try:
            t0 = time.perf_counter()
            task.detector.apply(task.prepared)
            self.timings.mark("apply", time.perf_counter() - t0)
        except Exception as exc:
            # Vanishingly unlikely, but if it happens the old task is intact.
            log.exception("apply failed")
            self.machine.on_prepare_failed(f"apply: {exc}")
            return

        self.active = task
        latency = now - outcome.job.submitted_ts
        log.info("swapped in %s after %.2fs (prepare %.2fs)",
                 task.summary(), latency, outcome.seconds)
        self.machine.on_applied(now)

    # 2. One frame -------------------------------------------------------
    def step(self, now: float | None = None) -> TargetState | None:
        """Exactly one frame of work. Returns what was published, if anything."""
        now = time.time() if now is None else now
        self._drain_loader(now)

        frame, frame_ts, seq = self.capture.read()
        if frame is None or self.capture.age(now) > CFG.CAMERA_DEAD_S:
            return self._camera_down(now)

        if self.machine.phase is Phase.FAULT and self._fault_source == "camera":
            self._fault_source = None
            self.machine.on_recovered(now)

        if seq == self._last_seq:
            return None  # no new frame; the source is slower than we are
        self._last_seq = seq
        self.timings.tick(now)

        if self.active is None:
            return self._publish(TargetState(ts=now, phase=self.machine.phase), frame, None, None)

        dets = self._detect(frame, now)
        if dets is None:
            # Publish anyway: a controller that stops hearing from us cannot
            # tell a broken pipeline from a broken network.
            return self._publish(
                TargetState(ts=now, phase=self.machine.phase, visible=False,
                            spec_id=self.active.spec.spec_id, mode=self.active.spec.mode,
                            model=self.active.model_name),
                frame, None, None,
            )

        t0 = time.perf_counter()
        tracked = self.active.track(dets)
        self.timings.mark("track", time.perf_counter() - t0)

        t0 = time.perf_counter()
        state, chosen = self._resolve(tracked, frame.shape, now)
        self.timings.mark("select", time.perf_counter() - t0)

        if self.timings.frames % CFG.TIMING_EVERY == 0:
            log.info("timings %s", self.timings.line())
        return self._publish(state, frame, tracked, chosen)

    def _detect(self, frame: np.ndarray, now: float) -> sv.Detections | None:
        try:
            t0 = time.perf_counter()
            dets = self.active.detector.infer(frame)
            self.timings.mark("detect", time.perf_counter() - t0)
            self.machine.on_infer_ok()
            if self.machine.phase is Phase.FAULT and self._fault_source == "infer":
                self._fault_source = None
                self.machine.on_recovered(now)
            return dets
        except Exception as exc:
            log.warning("inference error: %s", exc)
            before = self.machine.phase
            if self.machine.on_infer_error(str(exc), now) is Phase.FAULT and before is not Phase.FAULT:
                self._fault_source = "infer"
            return None

    # 3. Target resolution ------------------------------------------------
    def _resolve(self, tracked: sv.Detections, shape, now: float) -> tuple[TargetState, int | None]:
        """Per target: narrow to its classes, pick one. Then arbitrate."""
        task = self.active
        picks: list[int | None] = []
        subsets: list[sv.Detections] = []
        for t in task.spec.targets:
            subset = candidates_for(tracked, t.detect)
            subsets.append(subset)
            picks.append(pick(subset, t.select, shape, task.select_states[t.ref]))

        idx = task.arbiter.choose([p is not None for p in picks], now)
        target = task.spec.targets[idx]
        subset, chosen = subsets[idx], picks[idx]

        phase = self.machine.on_frame(chosen is not None, now)
        state = TargetState(
            ts=now, spec_id=task.spec.spec_id, mode=task.spec.mode,
            visible=chosen is not None, phase=phase,
            model=task.model_name, target_ref=target.ref,
        )
        if chosen is None:
            return state, None

        x1, y1, x2, y2 = subset.xyxy[chosen]
        h, w = shape[:2]
        state.cx = float(((x1 + x2) / 2) / w * 2 - 1)
        state.cy = float(((y1 + y2) / 2) / h * 2 - 1)
        state.area = float((x2 - x1) * (y2 - y1) / (w * h))
        state.conf = float(subset.confidence[chosen]) if subset.confidence is not None else 0.0
        names = subset.data.get("class_name")
        state.label = str(names[chosen]) if names is not None else None
        ids = subset.tracker_id
        state.track_id = int(ids[chosen]) if ids is not None else None
        return state, self._index_in(tracked, subset, chosen)

    @staticmethod
    def _index_in(tracked: sv.Detections, subset: sv.Detections, i: int) -> int | None:
        """Map a subset index back to the full frame, for the annotator."""
        if subset.tracker_id is None or tracked.tracker_id is None:
            return None
        hit = np.flatnonzero(tracked.tracker_id == subset.tracker_id[i])
        return int(hit[0]) if hit.size else None

    # 4. Failure ----------------------------------------------------------
    def _camera_down(self, now: float) -> TargetState | None:
        """Keep talking while broken.

        Silence and failure look identical to a controller, and only one of
        them means "the operator can still see what is wrong".
        """
        if self.machine.phase is not Phase.FAULT:
            self._fault_source = "camera"
            self.machine.on_fault(f"no frame for {self.capture.age(now):.1f}s", now)
        if now - self._last_publish < 1.0 / CFG.FAULT_HEARTBEAT_HZ:
            return None
        return self._publish(
            TargetState(ts=now, phase=Phase.FAULT, visible=False,
                        spec_id=self.machine.spec_id), None, None, None
        )

    # 5. Publishing -------------------------------------------------------
    def _publish(self, state: TargetState, frame, dets, chosen) -> TargetState:
        self.last_state = state
        self._last_publish = state.ts
        self.view = View(frame=frame, detections=dets, state=state,
                         chosen=chosen, seq=self._last_seq)
        self.target_bus.publish(state)
        return state

    # 6. Thread -----------------------------------------------------------
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
                # Nothing new. Yield rather than spin; the capture thread sets
                # the real pace.
                self._stop.wait(0.002)
