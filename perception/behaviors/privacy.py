"""privacy: blur what should not be seen.

Two modes, and the interesting one is the inverse. "Blur everyone's face
except mine" is not a list of people to blur, it is everyone *except* a
reference match, which means the safe default matters: a face whose appearance
has not been scored yet is blurred, not shown. Failing closed is the whole
point of a privacy feature.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from render.layers import Blur, Boxes


class Privacy(Behavior):
    kind = "privacy"
    states = ("ACTIVE", "PAUSED")
    initial = "ACTIVE"
    emits = ()

    def validate(self) -> None:
        self.mode = self.param("mode", "blur")
        if self.mode not in ("blur", "pixelate"):
            raise ValueError(f"privacy mode must be 'blur' or 'pixelate', not {self.mode!r}")
        self.keep_ref = self.param("keep_ref", None)

    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        if self.keep_ref:
            self._keep_one(frame, outcome)
            return
        self.data["blurred"] = int(len(idx))
        if len(idx):
            outcome.layers.append(Blur(boxes=frame.tracks[idx].xyxy, mode=self.mode))

    def _keep_one(self, frame: Frame, outcome: Outcome) -> None:
        """Blur everything matching the subject except the reference.

        `filter` is run without the reference so it returns every candidate,
        then the one that matches is spared. Doing it the other way round would
        mean an unscored face is never blurred, which is the wrong way to fail.
        """
        candidates = self.filter(frame, self.subject.model_copy(
            update={"ref_id": None, "pick": "all"}))
        if len(candidates) == 0:
            self.data["blurred"] = 0
            return
        ids = frame.tracks.tracker_id
        keep = []
        for i in candidates:
            tid = int(ids[i]) if ids is not None else -1
            if frame.bank.ref_match(tid, [self.keep_ref],
                                    self.subject.ref_min_sim) is True:
                keep.append(int(i))
        hide = np.array([int(i) for i in candidates if int(i) not in keep], dtype=int)
        self.data["blurred"] = int(len(hide))
        self.data["kept"] = len(keep)
        if len(hide):
            outcome.layers.append(Blur(boxes=frame.tracks[hide].xyxy, mode=self.mode))
        if keep and self.spec.render.boxes:
            outcome.layers.append(Boxes(boxes=frame.tracks[np.array(keep)].xyxy,
                                        labels=[self.label], color=self.color))
