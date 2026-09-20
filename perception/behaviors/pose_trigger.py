"""pose_trigger: fire when someone makes a gesture.

"Tell me when someone raises their hand." The pose model runs only while this
behaviour is active, so a pipeline with no gesture behaviour pays nothing for
one being possible.

The subject narrows *who* counts before the gesture is checked, so "tell me
when someone in a red hoodie raises their hand" is this behaviour plus an
ordinary selector, not a second mechanism.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from render.layers import Alert, Boxes
from skills.pose import GESTURES, detect_gesture, person_box

#: Frames a gesture must hold before it counts. A hand passing through the
#: raised position on its way somewhere else is not a raised hand.
HOLD_FRAMES = 3
#: Quiet period after firing, so one raised hand is one alert.
COOLDOWN_S = 5.0
FLASH_S = 1.5


class PoseTrigger(Behavior):
    kind = "pose_trigger"
    states = ("ACTIVE", "PAUSED")
    initial = "ACTIVE"
    emits = ("hand_raised",)
    #: Without a pose model this behaviour cannot do anything, so the registry
    #: pauses it with a reason rather than letting it sit looking healthy.
    needs_roles = ("pose",)

    def __init__(self, *a, **kw) -> None:
        self._held = 0
        self._fired_at = -1e9
        self._banner = ""
        super().__init__(*a, **kw)

    def validate(self) -> None:
        self.gesture = self.param("gesture", "hand_raised")
        if self.gesture not in GESTURES:
            raise ValueError(
                f"unknown gesture {self.gesture!r}; known: {sorted(GESTURES)}")
        self.cooldown_s = float(self.param("cooldown_s", COOLDOWN_S))
        self.hold_frames = int(self.param("hold_frames", HOLD_FRAMES))

    # 1. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        keypoints = frame.skills.get("pose")
        if keypoints is None:
            # The pose model has not produced anything yet this frame.
            self.data["people"] = 0
            return

        doing = detect_gesture(keypoints, self.gesture)
        doing = [i for i in doing if self._is_subject(frame, idx, keypoints[i])]
        self.data.update(people=int(len(keypoints)), gesturing=len(doing))

        self._draw(frame, keypoints, doing, outcome)
        if self._banner and frame.now - self._fired_at < FLASH_S:
            # Held for a beat: a single-frame flash at 30fps is invisible.
            outcome.layers.append(Alert(text=self._banner, color=self.color, intensity=0.7))

        if not doing:
            self._held = 0
            return
        self._held += 1
        if self._held < self.hold_frames:
            return
        if frame.now - self._fired_at < self.cooldown_s:
            return

        self._fired_at = frame.now
        self._held = 0
        ev = self.event(
            self.gesture,
            frame,
            None,
            detail=f"{self.label}: {len(doing)} "f"{'person' if len(doing) == 1 else 'people'} "f"{self.gesture.replace('_', ' ')}",
            count=len(doing)
        )
        self._capture(frame, keypoints[doing[0]], ev, outcome)
        outcome.events.append(ev)
        self._banner = ev.detail
        outcome.layers.append(Alert(text=ev.detail, color=self.color, intensity=0.7))

    def _is_subject(self, frame: Frame, idx: np.ndarray, person: np.ndarray) -> bool:
        """
        Only count the gesture if it came from someone the selector wants.

        The pose model finds every body; the selector may want only some of
        them. Matched by box overlap, because pose and detection are separate
        models with no shared identity.
        """
        if not self.subject.needs_attributes and self.subject.ref_id is None:
            return True
        box = person_box(person)
        if box is None or len(idx) == 0:
            return False
        px1, py1, px2, py2 = box
        for i in idx:
            x1, y1, x2, y2 = frame.tracks.xyxy[int(i)]
            if not (px1 > x2 or px2 < x1 or py1 > y2 or py2 < y1):
                return True
        return False

    def _capture(self, frame: Frame, person: np.ndarray, ev, outcome: Outcome) -> None:
        box = person_box(person)
        if box is None:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        pad = 32
        crop = frame.image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size:
            outcome.snapshots.append((ev.id, crop.copy()))
            ev.snapshot_url = f"/snapshots/{ev.id}.jpg"

    def _draw(self, frame: Frame, keypoints, doing: list[int], outcome: Outcome) -> None:
        boxes, labels = [], []
        for i in range(len(keypoints)):
            box = person_box(keypoints[i])
            if box is None:
                continue
            boxes.append(box)
            labels.append(self.gesture.replace("_", " ") if i in doing else "")
        if boxes:
            outcome.layers.append(Boxes(
                boxes=np.array(boxes, dtype=np.float32), labels=labels,
                color=self.color,
                emphasis=next((n for n, i in enumerate(range(len(boxes))) if i in doing), None)
            ))
