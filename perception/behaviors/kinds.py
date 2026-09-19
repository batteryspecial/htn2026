"""Every behaviour kind in the contract, built or not.

Kinds that are not implemented yet are registered anyway and refuse at
validation, naming what they need. That way the orchestrator can be written
against the whole contract today and finds out immediately, rather than
silently, when it asks for something that does not exist.
"""

from __future__ import annotations

from behaviors.base import Behavior, UnsupportedBehavior
from behaviors.highlight import Highlight
from behaviors.track import Track
from contracts import BehaviorSpec


class NotYetBuilt(Behavior):
    needs = "not implemented"

    def validate(self) -> None:
        raise UnsupportedBehavior(
            f"behaviour kind {self.spec.kind!r} is not implemented yet "
            f"({self.needs}). Available now: {sorted(BUILT)}")


class Watch(NotYetBuilt):
    kind = "watch"
    states = ("ARMING", "ARMED", "FIRED", "COOLDOWN", "PAUSED")
    needs = "needs proximity triggers and snapshot capture"


class CountLine(NotYetBuilt):
    kind = "count_line"
    states = ("ACTIVE", "PAUSED")
    needs = "needs line-crossing geometry"


class Privacy(NotYetBuilt):
    kind = "privacy"
    states = ("ACTIVE", "PAUSED")
    needs = "needs reference matching"


class PanTo(NotYetBuilt):
    kind = "pan_to"
    states = ("GUIDING", "REACHED", "PAUSED")
    needs = "needs frame-to-frame odometry"


class PoseTrigger(NotYetBuilt):
    kind = "pose_trigger"
    states = ("ACTIVE", "PAUSED")
    needs = "needs the pose model"


class Keyboard(NotYetBuilt):
    kind = "keyboard"
    states = ("SEARCHING", "LOCKED", "PAUSED")
    needs = "needs OCR and the layout homography"


KINDS: dict[str, type[Behavior]] = {
    c.kind: c for c in
    (Highlight, Track, Watch, CountLine, Privacy, PanTo, PoseTrigger, Keyboard)
}
#: Kinds that actually do something.
BUILT = {k for k, c in KINDS.items() if not issubclass(c, NotYetBuilt)}


def build(behavior_id: str, spec: BehaviorSpec, index: int = 0,
          taken: set | None = None) -> Behavior:
    """Construct a runtime behaviour, or raise. Worker thread only.

    `taken` is the colours already on screen, so a new behaviour does not come
    out looking like one that is already running.
    """
    cls = KINDS.get(spec.kind)
    if cls is None:
        raise UnsupportedBehavior(f"unknown behaviour kind {spec.kind!r}")
    return cls(behavior_id, spec, index, taken)
