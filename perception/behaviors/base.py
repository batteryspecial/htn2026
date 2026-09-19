"""The behaviour boundary.

A behaviour is one standing instruction evaluated on every frame. Several run
at once and must not interfere, which sets three rules:

1. **Detection happens once.** The pipeline detects the union of every
   behaviour's classes; each behaviour filters that one result. Two behaviours
   watching people cost one forward pass.
2. **Tracking is shared and survives.** Starting a behaviour never resets the
   tracker, or adding a privacy blur would make the follow-cam lose its target.
   Only a model change resets tracking.
3. **Private memory.** Latches, counters and lifecycle live on the instance,
   created when it starts and dropped when it stops.

A behaviour never draws. It declares layers and the renderer composes them.

Every kind has its own state machine, but all of them share PAUSED: the active
model cannot see this subject, so the behaviour is held with a readable reason
and resumes by itself when a model that can see it comes back.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv

from contracts import BehaviorSpec, BehaviorState, BehaviorView, Event, EventType
from render.layers import PALETTE, BGR, Layer, distinct, hex_to_bgr

if TYPE_CHECKING:
    from attributes.clip_cache import AttributeBank

log = logging.getLogger("perception.behavior")


class UnsupportedBehavior(ValueError):
    """A kind the contract defines but this build does not implement yet."""


def _assign_color(spec: BehaviorSpec, index: int, taken: set[BGR]) -> BGR:
    """An explicit colour wins; otherwise take the first palette slot nobody
    is using.

    Two behaviours that look the same on the projector are two behaviours the
    audience cannot tell apart, which is the whole point of the multi-target
    items. Indexing the palette blindly collides as soon as one behaviour asks
    for a colour that happens to sit next to the following slot on the wheel.
    """
    if spec.render.color:
        return hex_to_bgr(spec.render.color)
    for offset in range(len(PALETTE)):
        candidate = PALETTE[(index + offset) % len(PALETTE)]
        if all(distinct(candidate, t) for t in taken):
            return candidate
    return PALETTE[index % len(PALETTE)]


@dataclass
class Frame:
    """Everything one frame offers a behaviour."""

    image: np.ndarray
    tracks: sv.Detections          # every track, already attribute-scored
    bank: AttributeBank
    now: float
    shape: tuple[int, int]
    trails: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


@dataclass
class Outcome:
    """What a behaviour did with a frame."""

    matches: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    chosen: int | None = None      # index into Frame.tracks, if it follows one
    events: list[Event] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    motor: Any = None              # MotorCommand, when this behaviour drives


class Behavior:
    """Base class. Subclasses set `kind`, `states`, and implement `on_frame`."""

    kind: str = "base"
    #: States this kind can reach, for documentation and validation.
    states: tuple[BehaviorState, ...] = ("ACTIVE", "PAUSED")
    initial: BehaviorState = "ACTIVE"
    emits: tuple[EventType, ...] = ()
    #: Frames of agreement before a state change. Debounced both ways so one
    #: dropped detection never restarts a behaviour and one false positive
    #: never triggers it.
    acquire_hits: int = 3
    lose_misses: int = 5

    def __init__(self, behavior_id: str, spec: BehaviorSpec, index: int = 0,
                 taken: set[BGR] | None = None) -> None:
        self.id = behavior_id
        self.spec = spec
        self.subject = spec.subject
        self.index = index
        self.color: BGR = _assign_color(spec, index, taken or set())
        self.label = spec.render.label or spec.kind
        self.state: BehaviorState = self.initial
        #: Frame time, not wall time. Every duration this behaviour measures is
        #: against the clock the loop is running on.
        self.now = time.time()
        self.since = self.now
        #: When this behaviour was started. The most recent track or pan_to
        #: owns the actuator, so a new "follow that" takes over the arrows.
        self.started = self.now
        self.detail: str | None = None
        self.data: dict[str, Any] = {}
        self._paused_from: BehaviorState | None = None
        self._hits = 0
        self._misses = 0
        self._matches = 0
        self._track_ids: list[int] = []
        self._ever_matched = False
        self._clock_offset: float | None = None
        self.validate()

    # 1. Setup ----------------------------------------------------------
    def validate(self) -> None:
        """Check `params` for this kind. Runs on the worker, before anything is
        installed, so a bad behaviour is refused rather than half-started."""

    def param(self, name: str, default):
        return self.spec.params.get(name, default)

    # 2. Vocabulary -----------------------------------------------------
    def pause(self, reason: str) -> None:
        """The active model cannot see this subject. Hold, do not drop."""
        if self.state != "PAUSED":
            self._paused_from = self.state
            self.to("PAUSED", reason)
            # It is not matching anything while held, and saying otherwise
            # would leave a stale count on the HUD.
            self._matches = 0
            self._track_ids = []
            self._hits = self._misses = 0

    def resume(self) -> None:
        """A model that can see the subject came back."""
        if self.state == "PAUSED":
            back = self._paused_from or self.initial
            self._paused_from = None
            self.detail = None
            self.to(back)
            self._hits = self._misses = 0

    # 3. Per frame ------------------------------------------------------
    def step(self, frame: Frame) -> Outcome:
        if self._clock_offset is None:
            # A behaviour is constructed on the worker, against wall time, and
            # first runs on the loop, against the loop's clock. Rebase on the
            # first frame so every duration it measures is in one clock.
            self._clock_offset = frame.now - self.now
            self.since += self._clock_offset
            self.started += self._clock_offset
        self.now = frame.now
        if self.state == "PAUSED":
            return Outcome()
        idx = self.filter(frame)
        outcome = Outcome(matches=idx)
        self.on_frame(frame, idx, outcome)
        self._matches = len(idx)
        ids = frame.tracks.tracker_id
        self._track_ids = [int(ids[i]) for i in idx] if ids is not None else []
        self._check_barren(frame)
        return outcome

    #: How long a behaviour may match nothing before it says so.
    barren_after_s: float = 4.0

    def _check_barren(self, frame: Frame) -> None:
        """Say so when a selector has never matched anything.

        Measured on real footage: YOLOE finds a rubber duck for "yellow duck"
        in every frame and for "duck" in none. A behaviour that silently does
        nothing is the worst failure on stage, because it looks like the
        pipeline is working. Surfacing it gives the agent something to read
        and rephrase against.
        """
        if self._matches:
            self._ever_matched = True
            if self.detail and self.detail.startswith("nothing matches"):
                self.detail = None
            return
        if self._ever_matched or self.state == "PAUSED" or self.detail:
            return
        if frame.now - self.started >= self.barren_after_s:
            self.detail = (f"nothing matches {self.subject.summary()!r} — "
                           f"try a more specific phrase")
            log.warning("behaviour %s: %s", self.id, self.detail)

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
            if not (subject.needs_attributes or subject.ref_id):
                keep.append(i)
                continue
            if ids is None:
                continue
            tid = int(ids[i])
            if subject.needs_attributes and frame.bank.matches(
                    tid, subject.include, subject.exclude, subject.min_score) is not True:
                continue
            if subject.ref_id and frame.bank.ref_match(
                    tid, [subject.ref_id], subject.ref_min_sim) is not True:
                continue
            keep.append(i)
        if subject.relate:
            keep = self._contained(tracks, keep, names)
        return np.array(keep, dtype=int)

    def _contained(self, tracks: sv.Detections, keep: list[int], names) -> list[int]:
        """Keep only subjects containing the related class low inside them.

        "the person in red shoes" is a person box with a shoe box near its
        bottom, which is steadier than asking CLIP about a whole-body crop.
        """
        rel = self.subject.relate
        if names is None:
            return keep
        inner = [i for i in range(len(tracks)) if str(names[i]) == rel.contains]
        if not inner:
            return []
        out = []
        for i in keep:
            x1, y1, x2, y2 = tracks.xyxy[i]
            floor = y2 - (y2 - y1) * rel.lower_frac
            for j in inner:
                jx1, jy1, jx2, jy2 = tracks.xyxy[j]
                cx, cy = (jx1 + jx2) / 2, (jy1 + jy2) / 2
                if x1 <= cx <= x2 and floor <= cy <= y2:
                    out.append(i)
                    break
        return out

    def choose(self, frame: Frame, idx: np.ndarray) -> int | None:
        """Pick the one track this behaviour follows, in full-frame indices."""
        from behaviors.picking import pick

        if len(idx) == 0:
            return None
        local = pick(frame.tracks[idx], self.subject.pick, frame.shape, self)
        return None if local is None else int(idx[local])

    # 4. State ----------------------------------------------------------
    def to(self, state: BehaviorState, detail: str | None = None) -> None:
        if state != self.state:
            log.info("behavior %s (%s) %s -> %s%s", self.id, self.kind,
                     self.state, state, f" ({detail})" if detail else "")
            self.state, self.since = state, self.now
        self.detail = detail

    def seen(self, hit: bool) -> None:
        """Advance the debounce counters."""
        if hit:
            self._hits, self._misses = self._hits + 1, 0
        else:
            self._misses, self._hits = self._misses + 1, 0

    @property
    def confirmed(self) -> bool:
        return self._hits >= self.acquire_hits

    @property
    def gone(self) -> bool:
        return self._misses >= self.lose_misses

    # 5. Reporting ------------------------------------------------------
    def event(self, type_: EventType, frame: Frame | None = None,
              index: int | None = None, detail: str = "", **data) -> Event:
        ev = Event(id=uuid.uuid4().hex[:10], type=type_, ts=frame.now if frame else time.time(),
                   behavior_id=self.id, detail=detail or f"{self.label}: {type_}",
                   data=data, notify=self.spec.notify)
        if frame is not None and index is not None and index < len(frame.tracks):
            box = frame.tracks.xyxy[index]
            h, w = frame.shape[:2]
            ids = frame.tracks.tracker_id
            ev.data.setdefault("cx", float((box[0] + box[2]) / 2 / w * 2 - 1))
            ev.data.setdefault("cy", float((box[1] + box[3]) / 2 / h * 2 - 1))
            ev.data.setdefault("area", float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)))
            if ids is not None:
                ev.data.setdefault("track_id", int(ids[index]))
        return ev

    def view(self) -> BehaviorView:
        return BehaviorView(
            id=self.id, kind=self.spec.kind, state=self.state,
            spec=self.subject.summary(), label=self.label, matches=self._matches,
            track_ids=list(self._track_ids), since=self.since, detail=self.detail,
            notify=self.spec.notify, data=dict(self.data),
        )

    def chip(self) -> tuple[str, str, BGR]:
        return (self.label, self.state, self.color)
