"""Shared wire contracts. One source of truth for all four services.

Two generations live here side by side, both current:

- Sections 1-3 are the single-target pipeline: one TaskSpec, one TargetState
  per frame. Still how the follow-cam and the actuator are driven.
- Sections 4-7 are the behaviour layer: many named behaviours running at once
  on the same frame, each with its own subject and lifecycle. This is what the
  agent starts and stops.

Owned jointly: perception (8001), orchestrator (8000), camera (8003), frontend.
Change this only by agreement, because everyone parses it.

Import it from anywhere in the repo:

    from linker.schemas import TargetState, Phase

Everything published here is deliberately hardware-agnostic. Positions are
normalized to [-1, 1] offsets from frame centre and sizes are frame fractions,
so no consumer needs to know the camera resolution, the lens, or what the
frame is eventually pointed at.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 1. Pipeline state
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    """Level-triggered pipeline state, published on every TargetState.

    Level-triggered on purpose: a consumer that drops a message self-corrects
    on the next one. Anything that acts on the world reads this, never events.
    """

    BOOTING = "BOOTING"
    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    LOADING_MODEL = "LOADING_MODEL"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    NO_TARGET = "NO_TARGET"
    FAULT = "FAULT"

    @property
    def drivable(self) -> bool:
        """The one phase in which a consumer may act on the target."""
        return self is Phase.TRACKING


#: Phases in which the pipeline is alive and will accept a new spec.
RUNNING_PHASES = frozenset(
    {
        Phase.IDLE,
        Phase.SWITCHING,
        Phase.LOADING_MODEL,
        Phase.ACQUIRING,
        Phase.TRACKING,
        Phase.LOST,
        Phase.NO_TARGET,
    }
)

#: Progress stages perception reports on its event channel.
Stage = Literal[
    "prepared",
    "model_loading",
    "model_loaded",
    "applied",
    "active",
    "no_target",
    "failed",
]

SelectRule = Literal["largest", "most_centered", "highest_conf", "locked"]
Mode = Literal["follow", "center"]


# ---------------------------------------------------------------------------
# 2. Task specification
# ---------------------------------------------------------------------------


class Verify(BaseModel):
    """Attribute check on a track's crop, scored contrastively by CLIP.

    Scored against `f"a {class}"` rather than thresholded on a raw similarity,
    because raw CLIP scores are too flat to threshold directly.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cls: str = Field(alias="class")
    text: str  # positive prompt, e.g. "a red shoe"
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class Relate(BaseModel):
    """Keep a `keep` box only when it contains an `if_contains` box low in it.

    Expresses "the person wearing the red shoes" as a containment test rather
    than as a second detection problem.
    """

    model_config = ConfigDict(extra="forbid")

    keep: str
    if_contains: str


class Target(BaseModel):
    """One thing to find and follow."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    detect: Annotated[list[str], Field(min_length=1, max_length=6)]
    verify: Verify | None = None
    relate: Relate | None = None
    select: SelectRule = "largest"


class Arbitration(BaseModel):
    """How attention is split when a spec carries two targets."""

    model_config = ConfigDict(extra="forbid")

    alternate_s: float = Field(default=3.0, gt=0.0)


class TaskSpec(BaseModel):
    """What the pipeline should be doing. Compiled from one instruction."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    targets: Annotated[list[Target], Field(min_length=1, max_length=2)]
    mode: Mode = "follow"
    arbitration: Arbitration = Field(default_factory=Arbitration)
    #: Validated against the live registry rather than against this schema, so
    #: adding a model never means editing the shared contract.
    model: str = "yoloe"

    def prompt_union(self) -> list[str]:
        """Every class name this spec needs, deduplicated, order preserved.

        One detector pass covers every target, so the detector is prepared on
        the union rather than once per target.
        """
        names: list[str] = []
        for t in self.targets:
            extra = [t.relate.if_contains] if t.relate else []
            for n in (*t.detect, *extra):
                if n not in names:
                    names.append(n)
        return names


class SpecRequest(BaseModel):
    """Body of POST /spec."""

    model_config = ConfigDict(extra="forbid")

    instruction_id: str
    spec: TaskSpec


# ---------------------------------------------------------------------------
# 3. Published output
# ---------------------------------------------------------------------------


class TargetState(BaseModel):
    """One per frame. The only message a real-time consumer needs."""

    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    spec_id: str | None = None
    mode: Literal["follow", "center", "idle"] = "idle"
    visible: bool = False
    cx: float = 0.0  # [-1, 1] offset from frame centre, right positive
    cy: float = 0.0  # [-1, 1] offset from frame centre, down positive
    area: float = 0.0  # bbox area / frame area
    conf: float = 0.0
    label: str | None = None
    track_id: int | None = None
    phase: Phase = Phase.BOOTING
    model: str | None = None
    target_ref: str | None = None


class StatusEvent(BaseModel):
    """Edge-triggered progress, for humans via the orchestrator.

    Never for a consumer that acts on the world: one missed event and it is
    stuck. Those read `TargetState.phase` instead.
    """

    model_config = ConfigDict(extra="forbid")

    stage: Stage
    ts: float = Field(default_factory=time.time)
    instruction_id: str | None = None
    spec_id: str | None = None
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4. Subjects: what counts as "my thing"
# ---------------------------------------------------------------------------


class Subject(BaseModel):
    """What a behaviour is about.

    One shape covers the whole demo list, which is the point:

        {"detect": ["person"]}                              every human
        {"detect": ["duck"], "include": ["a yellow duck"]}   the yellow one
        {"detect": ["person"], "exclude": ["a person in a
         black jacket"]}                                     everyone but those
        {"detect": ["person"], "reference_ids": ["r1"]}      the person in a photo
        {"detect": ["person"], "include": ["a person
         wearing red shoes"]}                                attribute + relation

    `detect` is what the detector looks for. `include` and `exclude` are CLIP
    checks on each track's crop, scored contrastively rather than thresholded
    on a raw similarity, because raw CLIP scores are too flat to threshold.
    A track must match every `include` and no `exclude`.
    """

    model_config = ConfigDict(extra="forbid")

    detect: Annotated[list[str], Field(min_length=1, max_length=8)]
    include: list[str] = Field(default_factory=list, max_length=4)
    exclude: list[str] = Field(default_factory=list, max_length=4)
    #: Ids from POST /references: "track the human in this photo".
    reference_ids: list[str] = Field(default_factory=list, max_length=4)
    #: Containment, e.g. keep a person only if a shoe sits low inside them.
    relate: Relate | None = None
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    #: Cosine similarity a crop must reach to match a reference image.
    ref_min_sim: float = Field(default=0.75, ge=0.0, le=1.0)

    def prompts(self) -> list[str]:
        """Class names the detector must be prepared for."""
        extra = [self.relate.if_contains] if self.relate else []
        return list(dict.fromkeys([*self.detect, *extra]))

    def attribute_texts(self) -> list[str]:
        """Every phrase CLIP has to encode for this subject."""
        return list(dict.fromkeys([*self.include, *self.exclude]))

    @property
    def needs_attributes(self) -> bool:
        return bool(self.include or self.exclude or self.reference_ids)


# ---------------------------------------------------------------------------
# 5. Behaviours
# ---------------------------------------------------------------------------

#: Every behaviour the agent may start. Perception rejects, with a clear
#: message, any kind it does not implement yet, so the orchestrator can be
#: written against the whole set before the whole set exists.
BehaviorKind = Literal[
    "highlight",      # draw every match; the people counter and attribute search
    "track",          # follow one match; drives arrows and later the motor
    "watch",          # guard a thing, fire when something approaches it
    "count_line",     # count crossings of a line
    "privacy",        # blur every match, or everything except a match
    "pan_to",         # steer the view until a match is centred
    "pose_trigger",   # fire on a body pose, e.g. a raised hand
]

#: Lifecycle of one behaviour. Independent of the global pipeline Phase:
#: several behaviours in different states run on the same frame.
BehaviorState = Literal["arming", "armed", "active", "lost", "missing", "failed"]

#: What perception reports on its event channel.
EventKind = Literal[
    "acquired",     # the subject was seen for the first time
    "lost",         # a tracked subject went away
    "missing",      # a watched subject that should be there is not
    "moved",        # a watched subject changed position
    "near",         # something came close to a watched subject
    "crossed",      # something crossed a counting line
    "hand_raised",  # pose trigger
    "reached",      # pan_to finished, or a guidance target was centred
]


class Behavior(BaseModel):
    """One standing instruction, started and stopped by the agent.

    Behaviours are concurrent and independent: a privacy blur, a line counter
    and a sentinel all run on the same frame, and starting one must never
    disturb another.
    """

    model_config = ConfigDict(extra="forbid")

    behavior_id: str
    kind: BehaviorKind
    subject: Subject
    #: Which match to pick when the behaviour follows exactly one thing.
    select: SelectRule = "largest"
    #: Kind-specific settings: a line's endpoints, a nearness radius, whether
    #: privacy blurs the match or everything else. Validated per kind, not here,
    #: so adding a behaviour never means editing the shared contract.
    params: dict[str, Any] = Field(default_factory=dict)
    #: Event kinds this behaviour may emit. Empty means the kind's default set.
    triggers: list[EventKind] = Field(default_factory=list)
    label: str | None = None  # human-readable, shown on the feed and in the UI


class BehaviorRequest(BaseModel):
    """Body of POST /behaviors."""

    model_config = ConfigDict(extra="forbid")

    instruction_id: str
    behavior: Behavior


class BehaviorStatus(BaseModel):
    """A behaviour as seen from outside: what it is and how it is doing."""

    model_config = ConfigDict(extra="forbid")

    behavior_id: str
    kind: BehaviorKind
    state: BehaviorState
    label: str | None = None
    matches: int = 0           # matching tracks on the last frame
    track_ids: list[int] = Field(default_factory=list)
    since: float = 0.0         # when it entered its current state
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)  # counts, last target, ...


class PerceptionEvent(BaseModel):
    """Something happened in the world. Edge-triggered, for the agent.

    Carries enough context for the agent to act without asking a follow-up:
    which behaviour fired, what it was about, and where to find a crop.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: EventKind
    ts: float = Field(default_factory=time.time)
    behavior_id: str | None = None
    label: str | None = None
    track_id: int | None = None
    #: Normalized [-1, 1] centre and frame-fraction area, as in TargetState.
    cx: float = 0.0
    cy: float = 0.0
    area: float = 0.0
    #: Path or id of a saved crop, when one was captured.
    snapshot: str | None = None
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 6. Observations
# ---------------------------------------------------------------------------


class Track(BaseModel):
    """One tracked thing on one frame, after attributes have been scored.

    This is what queries return and what behaviours filter. Positions are
    normalized exactly as in TargetState.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: int
    label: str
    conf: float
    cx: float
    cy: float
    area: float
    #: Attribute phrase -> contrastive score, for whatever was scored.
    attributes: dict[str, float] = Field(default_factory=dict)


class FrameSummary(BaseModel):
    """What the pipeline is doing right now. Published once per frame."""

    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    phase: Phase = Phase.BOOTING
    model: str | None = None
    fps: float = 0.0
    tracks: list[Track] = Field(default_factory=list)
    behaviors: list[BehaviorStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 7. Agent queries
# ---------------------------------------------------------------------------

QueryKind = Literal["count", "look", "describe"]


class QueryRequest(BaseModel):
    """Body of POST /query. The agent asking the pipeline a question.

    `count` and `look` are answered from the current frame with no model call.
    `describe` hands a snapshot to a vision model and is therefore slow, which
    is why it is a query rather than something in the per-frame path.
    """

    model_config = ConfigDict(extra="forbid")

    kind: QueryKind
    subject: Subject | None = None
    detail: str | None = None  # the question, for `describe`


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: QueryKind
    ts: float = Field(default_factory=time.time)
    count: int = 0
    tracks: list[Track] = Field(default_factory=list)
    answer: str | None = None
    snapshot: str | None = None


__all__ = [
    "Arbitration",
    "Behavior",
    "BehaviorKind",
    "BehaviorRequest",
    "BehaviorState",
    "BehaviorStatus",
    "EventKind",
    "FrameSummary",
    "Mode",
    "PerceptionEvent",
    "Phase",
    "QueryKind",
    "QueryRequest",
    "QueryResult",
    "RUNNING_PHASES",
    "Relate",
    "SelectRule",
    "SpecRequest",
    "Stage",
    "StatusEvent",
    "Subject",
    "TargetState",
    "TaskSpec",
    "Target",
    "Track",
    "Verify",
]
