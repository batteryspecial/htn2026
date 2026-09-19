"""The behaviour boundary.

A behaviour is one standing instruction that runs on every frame: highlight
everything red, follow the dog, guard the duck. Several run at once and must
not interfere, which drives three rules.

1. **Detection happens once.** The pipeline detects the union of every
   behaviour's classes and every behaviour filters that one result. Two
   behaviours watching people do not cost two forward passes.
2. **Tracking is shared and survives.** Starting a behaviour must never reset
   the tracker, or adding a privacy blur would make the follow-cam lose its
   target. Only a model change resets tracking.
3. **Per-behaviour memory is private.** A behaviour's latch, counters and
   lifecycle live in its own instance, created when it starts and dropped when
   it stops.

Each subclass implements `step`, which sees the frame's tracks already filtered
to its own subject, and returns whatever it wants drawn and whatever happened.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv

from contracts import Behavior, BehaviorState, BehaviorStatus, EventKind, PerceptionEvent
from stages.select import SelectState, pick

if TYPE_CHECKING:
    from stages.attributes import AttributeBank

log = logging.getLogger("perception.behavior")


class UnsupportedBehavior(ValueError):
    """A kind the contract defines but this build does not implement yet."""


@dataclass
class Frame:
    """Everything one frame offers a behaviour."""

    image: np.ndarray
    tracks: sv.Detections      # every track this frame, already attribute-scored
    bank: AttributeBank
    now: float
    shape: tuple[int, int]


@dataclass
class Outcome:
    """What a behaviour did with a frame."""

    #: Indices into `Frame.tracks` this behaviour claims, for the renderer.
    matches: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    #: The one it is following, if it follows one. Index into `Frame.tracks`.
    chosen: int | None = None
    events: list[PerceptionEvent] = field(default_factory=list)


class RuntimeBehavior:
    """Base class. Subclasses override `on_frame` and declare their event kinds."""

    kind: str = "base"
    #: Events this kind emits when the spec does not narrow them.
    default_triggers: tuple[EventKind, ...] = ()
    #: Frames of agreement before a state change, so one dropped detection
    #: never restarts a behaviour and one false positive never triggers it.
    acquire_hits: int = 3
    lose_misses: int = 5

    def __init__(self, spec: Behavior) -> None:
        self.spec = spec
        self.id = spec.behavior_id
        self.subject = spec.subject
        self.state: BehaviorState = "arming"
        self.since = time.time()
        self.select_state = SelectState()
        self.detail: str | None = None
        self.data: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self._matches = 0
        self._track_ids: list[int] = []
        self.validate()

    # 1. Setup ----------------------------------------------------------
    def validate(self) -> None:
        """Check `params` for this kind. Runs on the worker, before anything
        is installed, so a bad behaviour is refused rather than half-started."""

    @property
    def triggers(self) -> tuple[EventKind, ...]:
        return tuple(self.spec.triggers) or self.default_triggers

    # 2. Per frame ------------------------------------------------------
    def step(self, frame: Frame) -> Outcome:
        """Filter to this behaviour's subject, debounce, then run the kind."""
        idx = self.filter(frame)
        outcome = Outcome(matches=idx)
        self._advance(len(idx) > 0, frame, outcome)
        self.on_frame(frame, idx, outcome)
        self._matches = len(idx)
        ids = frame.tracks.tracker_id
        self._track_ids = [int(ids[i]) for i in idx] if ids is not None else []
        return outcome

    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        """Kind-specific work. `idx` indexes the tracks matching this subject."""

    def filter(self, frame: Frame) -> np.ndarray:
        """Indices of the tracks that count as this behaviour's subject.

        Class first because it is free, then the cached attribute verdicts. A
        track whose appearance has not been scored yet is excluded rather than
        assumed to match: acting on an unverified track is how a privacy blur
        misses a face.
        """
        tracks, subject = frame.tracks, self.subject
        if len(tracks) == 0:
            return np.array([], dtype=int)

        names = tracks.data.get("class_name")
        wanted = set(subject.detect)
        ids = tracks.tracker_id
        keep = []
        for i in range(len(tracks)):
            if names is not None and str(names[i]) not in wanted:
                continue
            if not subject.needs_attributes:
                keep.append(i)
                continue
            if ids is None:
                continue
            tid = int(ids[i])
            if frame.bank.matches(tid, subject.include, subject.exclude,
                                  subject.min_score) is not True:
                continue
            if frame.bank.ref_match(tid, subject.reference_ids,
                                    subject.ref_min_sim) is not True:
                continue
            keep.append(i)
        return np.array(keep, dtype=int)

    def choose(self, frame: Frame, idx: np.ndarray) -> int | None:
        """Pick the one track this behaviour follows, in full-frame indices."""
        if len(idx) == 0:
            return None
        subset = frame.tracks[idx]
        local = pick(subset, self.spec.select, frame.shape, self.select_state)
        return None if local is None else int(idx[local])

    # 3. Lifecycle ------------------------------------------------------
    def _advance(self, seen: bool, frame: Frame, outcome: Outcome) -> None:
        if seen:
            self._hits, self._misses = self._hits + 1, 0
        else:
            self._misses, self._hits = self._misses + 1, 0

        if self.state in ("arming", "armed", "lost", "missing"):
            if self._hits >= self.acquire_hits:
                first = self.state in ("arming", "armed")
                self.to("active")
                if first and "acquired" in self.triggers:
                    outcome.events.append(self.event("acquired", frame, outcome))
        elif self.state == "active" and self._misses >= self.lose_misses:
            self.to("lost")
            if "lost" in self.triggers:
                outcome.events.append(self.event("lost", frame, outcome))

    def to(self, state: BehaviorState, detail: str | None = None) -> None:
        if state != self.state:
            log.info("behavior %s (%s) %s -> %s", self.id, self.kind, self.state, state)
            self.state, self.since, self.detail = state, time.time(), detail

    # 4. Reporting ------------------------------------------------------
    def event(self, kind: EventKind, frame: Frame, outcome: Outcome,
              detail: str | None = None, **data) -> PerceptionEvent:
        """Build an event positioned on whatever this behaviour is about."""
        i = outcome.chosen if outcome.chosen is not None else (
            int(outcome.matches[0]) if len(outcome.matches) else None
        )
        ev = PerceptionEvent(
            event_id=uuid.uuid4().hex[:12], kind=kind, ts=frame.now,
            behavior_id=self.id, label=self.spec.label or self.kind,
            detail=detail, data=data,
        )
        if i is not None:
            box = frame.tracks.xyxy[i]
            h, w = frame.shape[:2]
            ev.cx = float((box[0] + box[2]) / 2 / w * 2 - 1)
            ev.cy = float((box[1] + box[3]) / 2 / h * 2 - 1)
            ev.area = float((box[2] - box[0]) * (box[3] - box[1]) / (w * h))
            ids = frame.tracks.tracker_id
            ev.track_id = int(ids[i]) if ids is not None else None
        return ev

    def status(self) -> BehaviorStatus:
        return BehaviorStatus(
            behavior_id=self.id, kind=self.spec.kind, state=self.state,
            label=self.spec.label, matches=self._matches,
            track_ids=list(self._track_ids), since=self.since, detail=self.detail,
            data=dict(self.data),
        )
