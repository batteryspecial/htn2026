"""watch: guard a thing and say when something happens to it.

The sentinel, and the widest-reach behaviour on the board: "watch the duck and
tell me if anyone approaches", "guard the table", "anyone in a red hoodie is
authorised" are all this with different triggers.

    ARMING --subject stable--> ARMED --trigger--> FIRED --> COOLDOWN --> ARMED

ARMING matters. The behaviour learns where the subject *is* before it will
judge anything, so "it moved" means moved from where it was when you pointed
at it, not from wherever it happened to be on the first frame the detector got
lucky. Until then it stays quiet.

COOLDOWN matters too. A person standing near the duck would otherwise fire
thirty times a second, and the agent is rate-limited anyway: one alert that
arrives is worth more than a hundred that get dropped.

Authorisation is not a separate mechanism. "Anyone in a red hoodie is allowed"
is an `exclude` on the trigger's own selector, so the rule that decides who
counts is the same rule used everywhere else.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from contracts import Selector
from render.layers import Alert, Boxes, Masks, Zone

#: How long the subject must be steadily visible before the guard arms.
ARM_AFTER_S = 1.0
#: How long a fired alert stays on screen.
FLASH_S = 1.0


class Watch(Behavior):
    kind = "watch"
    states = ("ARMING", "ARMED", "FIRED", "COOLDOWN", "PAUSED")
    initial = "ARMING"
    emits = ("armed", "missing", "moved", "near", "appeared")

    def __init__(self, *a, **kw) -> None:
        self.triggers: list[dict] = []
        self.others: dict[int, Selector] = {}
        self._baseline: tuple[float, float] | None = None
        self._last_seen: float = 0.0
        self._fired_at: float = 0.0
        self._stable_since: float | None = None
        super().__init__(*a, **kw)

    # 1. Setup ----------------------------------------------------------
    def validate(self) -> None:
        raw = self.param("triggers", [{"type": "missing", "after_s": 2.0}])
        if not isinstance(raw, list) or not raw:
            raise ValueError("watch needs a non-empty 'triggers' list")
        known = {"missing", "moved", "near", "appeared"}
        for i, t in enumerate(raw):
            kind = t.get("type")
            if kind not in known:
                raise ValueError(f"unknown trigger {kind!r}; expected one of {sorted(known)}")
            if kind in ("near", "appeared"):
                if "other" not in t:
                    raise ValueError(f"trigger {kind!r} needs an 'other' selector")
                # Validated now, on the worker, so a malformed trigger is
                # refused before anything is installed.
                self.others[i] = Selector(**t["other"])
        self.triggers = raw
        self.cooldown_s = float(self.param("cooldown_s", 5.0))

    def selectors(self) -> list:
        return [self.subject, *self.others.values()]

    # 2. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        here = len(idx) > 0
        if here:
            self._last_seen = frame.now
        self._draw(frame, idx, outcome)

        if self.state == "ARMING":
            self._arming(frame, idx, here, outcome)
            return
        if self.state == "COOLDOWN":
            if frame.now - self._fired_at >= self.cooldown_s:
                self.to("ARMED")
            return
        if self.state == "FIRED":
            # A visible beat so the alert is readable, then back to watching.
            if frame.now - self._fired_at >= FLASH_S:
                self.to("COOLDOWN")
            outcome.layers.append(Alert(text=self.detail or "", intensity=0.8))
            return
        self._armed(frame, idx, outcome)

    def _arming(self, frame: Frame, idx, here: bool, outcome: Outcome) -> None:
        """Learn where the subject is before judging whether it moved."""
        if not here:
            self._stable_since = None
            return
        if self._stable_since is None:
            self._stable_since = frame.now
        if frame.now - self._stable_since < ARM_AFTER_S:
            return
        self._baseline = self._centre(frame, int(idx[0]))
        self.data["baseline"] = [round(v, 3) for v in self._baseline]
        self.to("ARMED")
        outcome.events.append(self.event(
            "armed", frame, int(idx[0]),
            detail=f"{self.label}: watching {self.subject.summary()}"))

    def _armed(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        for i, trig in enumerate(self.triggers):
            hit = self._check(trig, i, frame, idx)
            if hit is None:
                continue
            detail, data, index = hit
            ev = self.event(trig["type"], frame, index, detail=detail, **data)
            self._capture(frame, index, ev, outcome)
            outcome.events.append(ev)
            self.detail = detail
            self._fired_at = frame.now
            self.to("FIRED", detail)
            return

    # 3. Triggers -------------------------------------------------------
    def _check(self, trig: dict, i: int, frame: Frame, idx: np.ndarray):
        kind = trig["type"]
        if kind == "missing":
            gone = frame.now - self._last_seen
            if len(idx) == 0 and gone >= float(trig.get("after_s", 2.0)):
                return (f"{self.label}: {self.subject.summary()} is gone",
                        {"missing_for_s": round(gone, 1)}, None)
            return None

        if kind == "moved":
            if len(idx) == 0 or self._baseline is None:
                return None
            cx, cy = self._centre(frame, int(idx[0]))
            shift = float(np.hypot(cx - self._baseline[0], cy - self._baseline[1]))
            if shift >= float(trig.get("min_shift", 0.15)):
                return (f"{self.label}: {self.subject.summary()} moved",
                        {"shift": round(shift, 3)}, int(idx[0]))
            return None

        other = self.filter(frame, self.others[i])
        if len(other) == 0:
            return None

        if kind == "appeared":
            return (f"{self.label}: {self.others[i].summary()} appeared",
                    {"count": int(len(other))}, int(other[0]))

        # near: something came close to the thing being guarded.
        if len(idx) == 0:
            return None
        margin = float(trig.get("margin", 0.1))
        subject_box = frame.tracks.xyxy[int(idx[0])]
        for j in other:
            if self._close(subject_box, frame.tracks.xyxy[int(j)], margin, frame.shape):
                return (f"{self.label}: {self.others[i].summary()} is near "
                        f"{self.subject.summary()}", {"margin": margin}, int(j))
        return None

    @staticmethod
    def _close(a, b, margin: float, shape) -> bool:
        """Boxes within `margin` of the frame's smaller side, or overlapping.

        Distance between edges rather than centres, so a large object standing
        beside a small one still counts as near.
        """
        h, w = shape[:2]
        pad = margin * min(h, w)
        return not (b[0] > a[2] + pad or b[2] < a[0] - pad
                    or b[1] > a[3] + pad or b[3] < a[1] - pad)

    @staticmethod
    def _centre(frame: Frame, i: int) -> tuple[float, float]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = frame.tracks.xyxy[i]
        return (float((x1 + x2) / 2 / w * 2 - 1), float((y1 + y2) / 2 / h * 2 - 1))

    def _capture(self, frame: Frame, index, ev, outcome: Outcome) -> None:
        """Attach a crop, so the agent can look rather than ask."""
        if index is None or index >= len(frame.tracks):
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = frame.tracks.xyxy[index].astype(int)
        pad = 24
        crop = frame.image[max(0, y1 - pad):min(h, y2 + pad),
                           max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size:
            outcome.snapshots.append((ev.id, crop.copy()))
            ev.snapshot_url = f"/snapshots/{ev.id}.jpg"

    # 4. Render ---------------------------------------------------------
    def _draw(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        if len(idx) == 0:
            return
        subset = frame.tracks[idx]
        if self.spec.render.mask and subset.mask is not None:
            outcome.layers.append(Masks(masks=subset.mask, color=self.color))
        if self.spec.render.boxes:
            outcome.layers.append(Boxes(boxes=subset.xyxy,
                                        labels=[f"{self.label} [{self.state}]"],
                                        color=self.color))
        if self._baseline is not None:
            # The guarded spot, so "it moved" is visible rather than asserted.
            h, w = frame.shape[:2]
            cx = (self._baseline[0] + 1) / 2 * w
            cy = (self._baseline[1] + 1) / 2 * h
            r = 0.04 * min(h, w)
            outcome.layers.append(Zone(
                points=np.array([[cx - r, cy - r], [cx + r, cy - r],
                                 [cx + r, cy + r], [cx - r, cy + r]]),
                color=self.color))
