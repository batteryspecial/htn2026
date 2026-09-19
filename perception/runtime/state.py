"""The pipeline state machine.

One enum, one `transition()`, and a handful of trigger methods that encode the
table in CLAUDE.md. No state-machine library, no threads, no clock of its own:
every method that cares about time takes `now`, so tests are deterministic and
the inference loop stays the only writer.

Two outputs, deliberately separate:

1. `phase` is level-triggered and goes to the controller on every frame. A
   consumer reading a level cannot get stuck by dropping a message.
2. `StatusEvent`s are edge-triggered and go to humans via the orchestrator.

The machine decides *what* to emit. It does not know how events are delivered;
the caller passes an `emit` callable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from config import CFG
from contracts import RUNNING_PHASES, Phase, Stage, StatusEvent

log = logging.getLogger("perception.state")

Emit = Callable[[StatusEvent], None]


def _noop(_: StatusEvent) -> None:
    pass


@dataclass
class Machine:
    """Phase plus the counters that decide when the phase changes.

    Owned by the inference loop. Nothing else may mutate it.
    """

    emit: Emit = _noop
    acquire_hits: int = CFG.ACQUIRE_HITS
    lost_misses: int = CFG.LOST_MISSES
    acquire_timeout_s: float = CFG.ACQUIRE_TIMEOUT_S
    infer_errors_to_fault: int = CFG.INFER_ERRORS_TO_FAULT

    phase: Phase = Phase.BOOTING
    instruction_id: str | None = None
    spec_id: str | None = None

    # Phase to fall back to when a prepare or a load fails, and the phase to
    # resume from after a fault clears.
    _revert_to: Phase = Phase.IDLE
    _hits: int = 0
    _misses: int = 0
    _acquire_started: float = 0.0
    _active_announced: bool = False
    _infer_errors: int = 0
    _history: list[tuple[Phase, Phase, str | None]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 1. Core
    # ------------------------------------------------------------------
    def transition(self, new: Phase, event: Stage | None = None, detail: str | None = None,
                   data: dict | None = None, now: float | None = None) -> None:
        """Move to `new`, log it, and emit `event` if one is attached.

        Re-entering the same phase is allowed and still emits, because some
        edges (SWITCHING -> prepared -> SWITCHING) are self-loops that carry an
        event the UI needs.
        """
        old = self.phase
        self.phase = new
        self._history.append((old, new, event))
        if old is not new:
            log.info("phase %s -> %s%s", old.value, new.value, f" ({event})" if event else "")
        if event:
            self._emit(event, detail, data, now)

    def _emit(self, stage: Stage, detail: str | None = None, data: dict | None = None,
              now: float | None = None) -> None:
        ev = StatusEvent(
            stage=stage,
            ts=now if now is not None else time.time(),
            instruction_id=self.instruction_id,
            spec_id=self.spec_id,
            detail=detail,
            data=data or {},
        )
        log.info("event %s%s", stage, f" detail={detail}" if detail else "")
        self.emit(ev)

    # ------------------------------------------------------------------
    # 2. Boot
    # ------------------------------------------------------------------
    def on_registry_ready(self) -> None:
        self._revert_to = Phase.IDLE
        self.transition(Phase.IDLE)

    def on_boot_failed(self, detail: str) -> None:
        """YOLOE could not load. The service stays up and reports FAULT."""
        self.transition(Phase.FAULT, "failed", detail)

    # ------------------------------------------------------------------
    # 3. Spec lifecycle
    # ------------------------------------------------------------------
    def on_spec(self, instruction_id: str, spec_id: str) -> None:
        """A validated spec was accepted. Preparation happens off-loop.

        In FAULT the spec is remembered but the phase holds, because the camera
        or the detector is still broken; recovery resumes into ACQUIRING.
        """
        if self.phase in RUNNING_PHASES:
            self._revert_to = self.phase
        self.instruction_id = instruction_id
        self.spec_id = spec_id
        self._reset_counters()
        self._active_announced = False
        if self.phase is Phase.FAULT:
            return
        self.transition(Phase.SWITCHING)

    def on_superseded(self, instruction_id: str, spec_id: str | None) -> None:
        """A spec was replaced before it ever applied. The loser gets told."""
        ev = StatusEvent(
            stage="failed",
            instruction_id=instruction_id,
            spec_id=spec_id,
            detail="superseded",
        )
        log.info("event failed detail=superseded instruction=%s", instruction_id)
        self.emit(ev)

    def on_model_loading(self, model: str) -> None:
        self.transition(Phase.LOADING_MODEL, "model_loading", model, {"model": model})

    def on_model_loaded(self, model: str) -> None:
        self.transition(Phase.SWITCHING, "model_loaded", model, {"model": model})

    def on_prepared(self) -> None:
        """Worker finished. Self-loop: the swap still has to happen in the loop."""
        self.transition(Phase.SWITCHING, "prepared")

    def on_prepare_failed(self, detail: str) -> None:
        """Prepare or load blew up. No-op for the pipeline: old task keeps running."""
        self.transition(self._revert_to, "failed", detail)

    def on_applied(self, now: float) -> None:
        """The loop swapped the new task in at a frame boundary."""
        self._reset_counters()
        self._acquire_started = now
        self.transition(Phase.ACQUIRING, "applied", now=now)

    def on_clear_spec(self) -> None:
        """DELETE /spec."""
        self.instruction_id = None
        self.spec_id = None
        self._reset_counters()
        self._revert_to = Phase.IDLE
        self.transition(Phase.IDLE)

    # ------------------------------------------------------------------
    # 4. Per-frame
    # ------------------------------------------------------------------
    def on_frame(self, target_visible: bool, now: float) -> Phase:
        """Advance one frame's worth of debouncing. Returns the resulting phase.

        Debounced both ways so a single dropped detection never stops the car
        and a single false positive never starts it:
        `acquire_hits` consecutive hits to reach TRACKING, `lost_misses`
        consecutive misses to leave it.
        """
        if self.phase in (Phase.BOOTING, Phase.IDLE, Phase.FAULT,
                          Phase.SWITCHING, Phase.LOADING_MODEL):
            return self.phase

        if target_visible:
            self._hits += 1
            self._misses = 0
        else:
            self._misses += 1
            self._hits = 0

        if self.phase in (Phase.ACQUIRING, Phase.NO_TARGET, Phase.LOST):
            if self._hits >= self.acquire_hits:
                self._to_tracking(now)
            elif self.phase is Phase.ACQUIRING and now - self._acquire_started >= self.acquire_timeout_s:
                self.transition(Phase.NO_TARGET, "no_target", now=now)
        elif self.phase is Phase.TRACKING and self._misses >= self.lost_misses:
            self.transition(Phase.LOST, now=now)

        return self.phase

    def _to_tracking(self, now: float) -> None:
        # `active` is the retask-time metric, so it fires once per spec only.
        first = not self._active_announced
        self._active_announced = True
        self.transition(Phase.TRACKING, "active" if first else None, now=now)

    # ------------------------------------------------------------------
    # 5. Faults
    # ------------------------------------------------------------------
    def on_infer_error(self, detail: str, now: float) -> Phase:
        """Count consecutive inference errors; trip to FAULT at the threshold."""
        self._infer_errors += 1
        if self._infer_errors >= self.infer_errors_to_fault and self.phase is not Phase.FAULT:
            self.on_fault(f"inference: {detail}", now)
        return self.phase

    def on_infer_ok(self) -> None:
        self._infer_errors = 0

    def on_fault(self, detail: str, now: float | None = None) -> None:
        if self.phase in RUNNING_PHASES:
            self._revert_to = self.phase
        self._reset_counters()
        self.transition(Phase.FAULT, "failed", detail, now=now)

    def on_recovered(self, now: float) -> None:
        """Camera or detector is back. Re-acquire if a spec is loaded."""
        self._infer_errors = 0
        self._reset_counters()
        if self.spec_id is None:
            self.transition(Phase.IDLE)
            return
        self._acquire_started = now
        self.transition(Phase.ACQUIRING)

    # ------------------------------------------------------------------
    # 6. Helpers
    # ------------------------------------------------------------------
    def _reset_counters(self) -> None:
        self._hits = 0
        self._misses = 0

    @property
    def history(self) -> list[tuple[Phase, Phase, str | None]]:
        return list(self._history)
