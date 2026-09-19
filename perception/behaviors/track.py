"""track: follow one subject, and keep following that one.

The follow-cam, and the only behaviour that drives the actuator. Its state
machine is the one that matters on stage, because it is what the guidance
arrow reads:

    ACQUIRING --seen--> TRACKING <--> EDGE --gone--> LOST --10s--> SEARCHING
                            ^                          |
                            +------ same instance ------+

EDGE is an early warning, not a failure: the subject is still visible but near
the frame edge, so the arrow appears faintly before anything is lost.

The rule that makes or breaks the demo: leaving LOST requires the **same
instance**, not merely another object of the same class. A passerby cancelling
the arrow makes the whole thing look broken, so reacquisition checks appearance
against the embedding of the last good crop.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from contracts import MotorCommand
from render.layers import Boxes, Masks, Trails

#: |cx| past this is "near the edge".
EDGE_AT = 0.7
#: How long LOST waits before admitting it is searching.
SEARCH_AFTER_S = 10.0
#: Cosine an appearance must reach to be accepted as the same instance.
REID_SIM = 0.8


class Track(Behavior):
    kind = "track"
    states = ("ACQUIRING", "TRACKING", "EDGE", "LOST", "SEARCHING", "PAUSED")
    initial = "ACQUIRING"
    emits = ("acquired", "lost", "reacquired")

    def __init__(self, *a, **kw) -> None:
        self.locked_track_id: int | None = None
        self._appearance = None      # embedding of the last good crop
        self._exit_side: str | None = None
        self._last_cx = 0.0
        self._announced = False
        super().__init__(*a, **kw)

    # 1. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        chosen = self._resolve(frame, idx)
        outcome.chosen = chosen
        self.seen(chosen is not None)

        if chosen is not None:
            self._on_seen(frame, chosen, outcome)
        else:
            self._on_missing(frame, outcome)

        outcome.motor = self._motor(frame)
        self._draw(frame, chosen, outcome)

    def _resolve(self, frame: Frame, idx: np.ndarray) -> int | None:
        """The latched instance if it is here, otherwise try to re-identify."""
        chosen = self.choose(frame, idx)
        if chosen is not None:
            return chosen
        if self.state in ("LOST", "SEARCHING") and len(idx):
            return self._reidentify(frame, idx)
        return None

    def _reidentify(self, frame: Frame, idx: np.ndarray) -> int | None:
        """Adopt a candidate only if it *looks like* the one we lost.

        Without this a random passerby clears the guidance arrow and the demo
        reads as broken. With no appearance stored we would rather stay lost.
        """
        if self._appearance is None:
            return None
        ids = frame.tracks.tracker_id
        if ids is None:
            return None
        best, best_sim = None, 0.0
        for i in idx:
            entry = frame.bank.get(int(ids[i]))
            if entry is None or entry.embedding is None:
                continue
            sim = float(np.dot(self._appearance, entry.embedding))
            if sim > best_sim:
                best, best_sim = int(i), sim
        if best is None or best_sim < REID_SIM:
            return None
        self.locked_track_id = int(ids[best])
        self.data["reid_sim"] = round(best_sim, 3)
        return best

    # 2. State ----------------------------------------------------------
    def _on_seen(self, frame: Frame, chosen: int, outcome: Outcome) -> None:
        h, w = frame.shape[:2]
        box = frame.tracks.xyxy[chosen]
        cx = float((box[0] + box[2]) / 2 / w * 2 - 1)
        cy = float((box[1] + box[3]) / 2 / h * 2 - 1)
        self._last_cx = cx
        ids = frame.tracks.tracker_id
        self.data.update(cx=round(cx, 3), cy=round(cy, 3),
                         area=round(float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)), 4),
                         track_id=int(ids[chosen]) if ids is not None else None)
        self._remember(frame, chosen)

        if not self.confirmed and self.state in ("ACQUIRING", "LOST", "SEARCHING"):
            return
        was_lost = self.state in ("LOST", "SEARCHING")
        self.to("EDGE" if abs(cx) > EDGE_AT else "TRACKING")
        self._exit_side = None
        if not self._announced:
            self._announced = True
            outcome.events.append(self.event("acquired", frame, chosen,
                                             detail=f"{self.label}: acquired"))
        elif was_lost:
            outcome.events.append(self.event("reacquired", frame, chosen,
                                             detail=f"{self.label}: reacquired"))

    def _remember(self, frame: Frame, chosen: int) -> None:
        """Keep the appearance of a confident sighting, for re-identification."""
        ids = frame.tracks.tracker_id
        conf = frame.tracks.confidence
        if ids is None or (conf is not None and conf[chosen] < 0.6):
            return
        entry = frame.bank.get(int(ids[chosen]))
        if entry is None or entry.embedding is None:
            return
        if self._appearance is None:
            self._appearance = entry.embedding
        else:
            # Slow EMA: drift with lighting, not with a bad frame.
            blend = 0.9 * self._appearance + 0.1 * entry.embedding
            self._appearance = blend / (np.linalg.norm(blend) + 1e-8)

    def _on_missing(self, frame: Frame, outcome: Outcome) -> None:
        if self.state in ("TRACKING", "EDGE") and self.gone:
            self._exit_side = "right" if self._last_cx > 0 else "left"
            self.to("LOST", f"exited {self._exit_side}")
            outcome.events.append(self.event(
                "lost", detail=f"{self.label}: lost, exited {self._exit_side}"))
        elif self.state == "LOST" and frame.now - self.since > SEARCH_AFTER_S:
            self.to("SEARCHING")

    # 3. Actuator -------------------------------------------------------
    def _motor(self, frame: Frame) -> MotorCommand | None:
        """What the view should do. A virtual motor draws it; a stepper would
        execute it. Same command either way."""
        if not self.param("guidance", True):
            return None
        if self.state == "EDGE":
            return MotorCommand(rate_deg_s=8.0 * self._last_cx,
                                reason=f"{self.label} near the edge")
        if self.state in ("LOST", "SEARCHING"):
            side = self._exit_side or "left"
            return MotorCommand(rate_deg_s=-20.0 if side == "left" else 20.0,
                                reason=f"pan {side}", urgent=True)
        return None

    # 4. Render ---------------------------------------------------------
    def _draw(self, frame: Frame, chosen: int | None, outcome: Outcome) -> None:
        if chosen is None:
            return
        subset = frame.tracks[np.array([chosen])]
        render = self.spec.render
        if render.mask and subset.mask is not None:
            outcome.layers.append(Masks(masks=subset.mask, color=self.color))
        if render.boxes:
            ids = subset.tracker_id
            label = f"{self.label}{f'#{ids[0]}' if ids is not None else ''}"
            outcome.layers.append(Boxes(boxes=subset.xyxy, labels=[label],
                                        color=self.color, emphasis=0))
        if render.trail and frame.tracks.tracker_id is not None:
            tid = int(frame.tracks.tracker_id[chosen])
            outcome.layers.append(Trails(paths=[frame.trails.get(tid, [])],
                                         color=self.color))
