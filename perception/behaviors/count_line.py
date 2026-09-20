"""count_line: count things crossing a line.

The traffic counter. Crossings are detected from each track's own path rather
than from where things are now: a line test on a single frame tells you which
side something is on, never that it changed sides.

The line is given in normalized coordinates so it survives a change of camera
or resolution, and defaults to a vertical line down the middle.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from render.layers import Boxes, Line


class CountLine(Behavior):
    kind = "count_line"
    states = ("ACTIVE", "PAUSED")
    initial = "ACTIVE"
    emits = ("crossed",)

    def __init__(self, *a, **kw) -> None:
        self._side: dict[int, float] = {}
        super().__init__(*a, **kw)

    def validate(self) -> None:
        line = self.param("line", [[0.5, 0.0], [0.5, 1.0]])
        try:
            (self.ax, self.ay), (self.bx, self.by) = line
        except Exception:
            raise ValueError(
                "count_line needs 'line': [[x1,y1],[x2,y2]] in 0..1 coordinates") from None
        if (self.ax, self.ay) == (self.bx, self.by):
            raise ValueError("count_line needs two different points")
        self.data.setdefault("in", 0)
        self.data.setdefault("out", 0)

    # 1. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        h, w = frame.shape[:2]
        ax, ay = self.ax * w, self.ay * h
        bx, by = self.bx * w, self.by * h
        ids = frame.tracks.tracker_id

        if ids is not None:
            for i in idx:
                tid = int(ids[i])
                x1, y1, x2, y2 = frame.tracks.xyxy[int(i)]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                # 2D cross product: which side of the line the centre is on.
                # numpy 2 dropped np.cross for 2-vectors, and writing it out
                # is clearer about what the sign means anyway.
                side = float((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if side == 0.0:
                    # Exactly on the line. Not a crossing yet, and recording it
                    # would lose which side it came from, so wait for it to
                    # commit to a side.
                    continue
                was = self._side.get(tid)
                self._side[tid] = side
                if was is None or (was > 0) == (side > 0):
                    continue
                # Direction comes from the transition, not from where it ended
                # up: a track that stops on the line and carries on would
                # otherwise be read as crossing the same way twice.
                direction = "in" if side > 0 else "out"
                self.data[direction] += 1
                outcome.events.append(self.event(
                    "crossed", frame, int(i),
                    detail=f"{self.label}: crossed {direction} "
                           f"(in {self.data['in']}, out {self.data['out']})",
                    direction=direction, **{k: self.data[k] for k in ("in", "out")}))
            self._forget(ids)

        outcome.layers.append(Line(
            a=(int(ax), int(ay)), b=(int(bx), int(by)), color=self.color,
            label=f"in {self.data['in']}  out {self.data['out']}"))
        if len(idx) and self.spec.render.boxes:
            outcome.layers.append(Boxes(
                boxes=frame.tracks[idx].xyxy,
                labels=[],
                color=self.color
            ))

    def _forget(self, ids) -> None:
        """A live stream never ends, so stale sides must not accumulate."""
        if len(self._side) <= 256:
            return
        live = {int(v) for v in ids}
        for tid in [t for t in self._side if t not in live]:
            del self._side[tid]
