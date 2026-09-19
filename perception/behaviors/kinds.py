"""The behaviour kinds this build implements, and the registry of all of them.

Every kind in the shared contract appears here. The ones that are not built
yet raise on validation with a message naming what is missing, so the
orchestrator can be written against the full contract today and finds out
immediately, not silently, when it asks for something that does not exist.
"""

from __future__ import annotations

import logging

import numpy as np

from behaviors.base import Frame, Outcome, RuntimeBehavior, UnsupportedBehavior
from contracts import Behavior

log = logging.getLogger("perception.behavior")


class Highlight(RuntimeBehavior):
    """Draw every match and say how many there are.

    The workhorse: "detect all humans" is a people counter, "highlight anything
    red" is attribute search, and both are this with a different subject.
    """

    kind = "highlight"
    default_triggers = ("acquired", "lost")

    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        self.data["count"] = int(len(idx))
        self.data["peak"] = max(self.data.get("peak", 0), int(len(idx)))


class Track(RuntimeBehavior):
    """Follow exactly one match, and keep following that one.

    The follow-cam. `select` decides the first pick; after that `locked` holds
    the same tracked identity, which is what makes "track *that* human"
    different from "track a human".
    """

    kind = "track"
    default_triggers = ("acquired", "lost")

    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        outcome.chosen = self.choose(frame, idx)
        if outcome.chosen is None:
            return
        box = frame.tracks.xyxy[outcome.chosen]
        h, w = frame.shape[:2]
        ids = frame.tracks.tracker_id
        self.data.update(
            cx=float((box[0] + box[2]) / 2 / w * 2 - 1),
            cy=float((box[1] + box[3]) / 2 / h * 2 - 1),
            area=float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)),
            track_id=int(ids[outcome.chosen]) if ids is not None else None,
        )


class NotYetBuilt(RuntimeBehavior):
    """A kind the contract promises and this build does not have.

    Refused at validation, on the worker, before anything is installed, so
    asking for one costs a clear error and changes nothing that is running.
    """

    needs = "not implemented"

    def validate(self) -> None:
        raise UnsupportedBehavior(
            f"behaviour kind {self.spec.kind!r} is not implemented yet ({self.needs}). "
            f"Available now: {sorted(BUILT)}"
        )


class Watch(NotYetBuilt):
    kind = "watch"
    needs = "needs proximity triggers and snapshot capture"


class CountLine(NotYetBuilt):
    kind = "count_line"
    needs = "needs line-crossing geometry"


class Privacy(NotYetBuilt):
    kind = "privacy"
    needs = "needs the blur renderer"


class PanTo(NotYetBuilt):
    kind = "pan_to"
    needs = "needs the actuator and guidance arrows"


class PoseTrigger(NotYetBuilt):
    kind = "pose_trigger"
    needs = "needs the pose skill"


KINDS: dict[str, type[RuntimeBehavior]] = {
    c.kind: c for c in (Highlight, Track, Watch, CountLine, Privacy, PanTo, PoseTrigger)
}
#: Kinds that actually do something, for error messages and GET /behaviors.
BUILT = {k for k, c in KINDS.items() if not issubclass(c, NotYetBuilt)}


def build(spec: Behavior) -> RuntimeBehavior:
    """Construct a runtime behaviour, or raise. Worker thread only."""
    cls = KINDS.get(spec.kind)
    if cls is None:
        raise UnsupportedBehavior(f"unknown behaviour kind {spec.kind!r}")
    return cls(spec)
