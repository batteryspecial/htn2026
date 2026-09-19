"""highlight: draw every match and report how many there are.

The workhorse of the demo list. "Detect all humans" is a people counter,
"highlight anything red" is attribute search, and both are this behaviour with
a different selector. It is also the reason the selector has to be good: most
items are this plus a sentence.

States: ACTIVE, PAUSED.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from render.layers import Boxes, Masks, Trails


class Highlight(Behavior):
    kind = "highlight"
    states = ("ACTIVE", "PAUSED")
    initial = "ACTIVE"
    emits = ("count_changed",)

    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        count = int(len(idx))
        before = self.data.get("count")
        self.data["count"] = count
        self.data["peak"] = max(self.data.get("peak", 0), count)

        # Edge-triggered: the agent wants to hear that it changed, not that it
        # is still four.
        if before is not None and before != count:
            outcome.events.append(self.event(
                "count_changed", frame, int(idx[0]) if len(idx) else None,
                detail=f"{self.label}: {count}", count=count, previous=before))

        if len(idx) == 0:
            return
        subset = frame.tracks[idx]
        render = self.spec.render
        if render.mask and subset.mask is not None:
            outcome.layers.append(Masks(masks=subset.mask, color=self.color))
        if render.boxes:
            outcome.layers.append(Boxes(boxes=subset.xyxy, labels=self._labels(subset),
                                        color=self.color))
        if render.trail:
            outcome.layers.append(Trails(
                paths=[frame.trails.get(t, []) for t in self._track_ids_of(subset)],
                color=self.color))

    def _labels(self, subset) -> list[str]:
        names, ids = subset.data.get("class_name"), subset.tracker_id
        return [f"{names[i] if names is not None else '?'}"
                f"{f'#{ids[i]}' if ids is not None else ''}"
                for i in range(len(subset))]

    @staticmethod
    def _track_ids_of(subset) -> list[int]:
        ids = subset.tracker_id
        return [int(v) for v in ids] if ids is not None else []
