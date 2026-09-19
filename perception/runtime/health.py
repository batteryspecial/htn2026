"""Pipeline health, and the system events that report it.

This replaces a nine-state machine that existed to tell a car when it was safe
to drive. There is no car: each behaviour now owns its own state, and all that
is left at the pipeline level is whether the camera and the model are working.

Debounced in both directions, because a camera that blinks should not produce
a pair of events every second.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from contracts import Event

log = logging.getLogger("perception.health")

#: Consecutive inference errors before the pipeline calls itself broken.
ERRORS_TO_FAULT = 3


def _noop(_: Event) -> None:
    pass


@dataclass
class Health:
    """Owned by the frame loop. Nothing else may mutate it."""

    emit: Callable[[Event], None] = _noop
    camera_ok: bool = False
    status: str = "booting"       # booting | ok | no_camera | fault
    detail: str | None = None
    _errors: int = 0
    _announced_camera: bool | None = field(default=None, repr=False)

    # 1. Camera ---------------------------------------------------------
    def camera(self, ok: bool, detail: str | None = None) -> None:
        self.camera_ok = ok
        if ok and self.status in ("booting", "no_camera"):
            self.status, self.detail = "ok", None
        elif not ok:
            self.status, self.detail = "no_camera", detail
        if self._announced_camera is not ok:
            self._announced_camera = ok
            # Edge-triggered, so the agent hears about it once.
            self.event("camera_ok" if ok else "camera_lost",
                       detail or ("camera back" if ok else "camera lost"))

    # 2. Inference ------------------------------------------------------
    def infer_ok(self) -> None:
        if self._errors or self.status == "fault":
            self._errors = 0
            if self.status == "fault":
                self.status, self.detail = "ok" if self.camera_ok else "no_camera", None
                log.info("inference recovered")

    def infer_error(self, detail: str) -> bool:
        """Returns True when this error tipped the pipeline into fault."""
        self._errors += 1
        if self._errors >= ERRORS_TO_FAULT and self.status != "fault":
            self.status, self.detail = "fault", detail
            log.warning("pipeline fault: %s", detail)
            return True
        return False

    # 3. Model ----------------------------------------------------------
    def model_switched(self, model: str, fps: float, paused: list[str]) -> None:
        self.event("model_switched", f"model is now {model}",
                   model=model, fps=round(fps, 1), paused=paused)

    # 4. Emitting -------------------------------------------------------
    def event(self, type_: str, detail: str, **data) -> Event:
        """System events carry no behaviour id and always wake the agent."""
        ev = Event(id=uuid.uuid4().hex[:10], type=type_, ts=time.time(),
                   behavior_id=None, detail=detail, data=data, notify=True)
        log.info("system event %s: %s", type_, detail)
        self.emit(ev)
        return ev
