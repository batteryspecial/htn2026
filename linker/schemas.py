"""Shared wire contracts. One source of truth for all services.

Owned jointly: perception (8001), orchestrator + frontend (8000), camera (8003).
Change this only by agreement, because everyone parses it.

    from linker.schemas import BehaviorSpec, Event

Everything published here is normalized: positions are [-1, 1] offsets from
frame centre, sizes are frame fractions. No consumer needs the resolution or
the lens.

The shape of the system in one line: a **Selector** says what counts as a
subject, a **Behavior** says what to do about it, and an **Event** says what
happened. Breadth comes from combining those, not from adding kinds.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 1. Selector: what counts as a subject
# ---------------------------------------------------------------------------

#: Which matches a behaviour acts on.
Pick = Literal["all", "largest", "most_centered", "ref"]


class Relate(BaseModel):
    """Containment: keep a subject only if another class sits low inside it.

    "the person in red shoes" is a person box containing a shoe box in its
    lower portion, which is cheaper and steadier than asking CLIP about a
    whole-body crop.
    """

    model_config = ConfigDict(extra="forbid")

    contains: str
    #: Fraction of the keep-box height, measured from the bottom, that the
    #: contained box's centre must fall within.
    lower_frac: float = Field(default=0.4, gt=0.0, le=1.0)


class Selector(BaseModel):
    """What counts as a subject. The keystone: one shape for the whole demo list.

        {"detect": ["person"]}                                   every human
        {"detect": ["duck"], "include": ["a yellow duck"]}        the yellow one
        {"detect": ["person"], "exclude": ["a person wearing a
         black jacket"]}                                          everyone else
        {"detect": ["person"], "ref_id": "r1", "pick": "ref"}     the one in a photo
        {"detect": ["person"], "relate": {"contains": "shoe"}}    wearing shoes

    `detect` is what the detector looks for: YOLOE prompts, or labels from a
    fixed vocabulary. `include` and `exclude` are CLIP checks on the subject's
    crop. A subject must match every `include` and no `exclude`.
    """

    model_config = ConfigDict(extra="forbid")

    detect: Annotated[list[str], Field(min_length=1, max_length=8)]
    include: list[str] = Field(default_factory=list, max_length=4)
    exclude: list[str] = Field(default_factory=list, max_length=4)
    #: A reference registered via POST /references.
    ref_id: str | None = None
    relate: Relate | None = None
    pick: Pick = "all"
    #: Contrastive pass mark for include/exclude. Measured on real footage:
    #: at 0.6 a man in a cream coat scored 0.705 for "wearing a dark jacket"
    #: and was wrongly excluded, while the actual dark jackets scored 0.99.
    #: The ranking was right and the bar was too low. Tune per clip with
    #: scripts/calibrate.py; this default errs toward matching too little.
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    #: Cosine a crop must reach to match the reference.
    ref_min_sim: float = Field(default=0.75, ge=0.0, le=1.0)

    def prompts(self) -> list[str]:
        """Class names the detector must be prepared for."""
        extra = [self.relate.contains] if self.relate else []
        return list(dict.fromkeys([*self.detect, *extra]))

    def attribute_texts(self) -> list[str]:
        """Phrases CLIP has to encode for this selector."""
        return list(dict.fromkeys([*self.include, *self.exclude]))

    @property
    def needs_attributes(self) -> bool:
        return bool(self.include or self.exclude)

    @property
    def needs_reference(self) -> bool:
        return self.ref_id is not None or self.pick == "ref"

    def summary(self) -> str:
        """One line for the HUD and the agent trace."""
        bits = "+".join(self.detect)
        if self.include:
            bits += f" ✓{','.join(self.include)}"
        if self.exclude:
            bits += f" ✗{','.join(self.exclude)}"
        if self.relate:
            bits += f" ⊃{self.relate.contains}"
        if self.ref_id:
            bits += f" ref:{self.ref_id}"
        return bits


# ---------------------------------------------------------------------------
# 2. Behaviours: what to do about a subject
# ---------------------------------------------------------------------------

BehaviorKind = Literal[
    "highlight",      # draw every match, report the count
    "track",          # follow one, drive the actuator
    "watch",          # guard a thing, fire when something happens to it
    "count_line",     # count crossings of a line
    "privacy",        # blur matches, or everything except a match
    "pan_to",         # guide the view until it has turned far enough
    "pose_trigger",   # fire on a body pose
    "keyboard",       # find a keyboard and guide typing
]

#: Every state any kind can be in. Which are reachable is per kind:
#:   highlight / count_line / privacy / pose_trigger : ACTIVE
#:   track     : ACQUIRING -> TRACKING <-> EDGE -> LOST -> SEARCHING
#:   watch     : ARMING -> ARMED -> FIRED -> COOLDOWN -> ARMED
#:   pan_to    : GUIDING -> REACHED
#:   keyboard  : SEARCHING <-> LOCKED
#: PAUSED is reachable from anywhere: the active model cannot see the subject.
BehaviorState = Literal[
    "ACTIVE", "PAUSED", "FAILED",
    "ACQUIRING", "TRACKING", "EDGE", "LOST", "SEARCHING",
    "ARMING", "ARMED", "FIRED", "COOLDOWN",
    "GUIDING", "REACHED", "LOCKED",
]


class RenderSpec(BaseModel):
    """How a behaviour wants to look. Chosen from primitives, never code.

    The agent picks colours and which layers to turn on. It never writes
    drawing code: generated drawing code is the fastest way to crash on stage.
    """

    model_config = ConfigDict(extra="forbid")

    color: str | None = None  # "#FFD400"; a palette slot is used when unset
    mask: bool = True
    trail: bool = False
    boxes: bool = True
    label: str | None = None


class BehaviorSpec(BaseModel):
    """Body of POST /behaviors. The id is assigned by the server."""

    model_config = ConfigDict(extra="forbid")

    kind: BehaviorKind
    subject: Selector
    #: Kind-specific settings, validated per kind rather than here, so adding
    #: a behaviour never means editing the shared contract.
    params: dict[str, Any] = Field(default_factory=dict)
    render: RenderSpec = Field(default_factory=RenderSpec)
    #: Whether events from this behaviour should wake the agent.
    notify: bool = True


class BehaviorCreated(BaseModel):
    """Response to POST /behaviors."""

    model_config = ConfigDict(extra="forbid")

    id: str


class BehaviorView(BaseModel):
    """A behaviour as seen from outside."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: BehaviorKind
    state: BehaviorState
    spec: str                  # one-line summary, for the HUD and the trace
    label: str | None = None
    matches: int = 0
    track_ids: list[int] = Field(default_factory=list)
    since: float = 0.0
    detail: str | None = None  # why PAUSED, why FAILED
    notify: bool = True
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. Events: what happened
# ---------------------------------------------------------------------------

EventType = Literal[
    # highlight
    "count_changed",
    # track
    "acquired", "lost", "reacquired",
    # watch
    "armed", "missing", "moved", "near", "appeared",
    # count_line / pan_to / pose
    "crossed", "reached", "hand_raised",
    # keyboard
    "keyboard_locked", "keyboard_lost", "step",
    # system, with behavior_id null
    "model_switched", "camera_lost", "camera_ok",
    #: An operation that was accepted then failed while being prepared. The
    #: synchronous 422 covers what can be known up front; this covers the rest.
    "behavior_failed",
]


class Event(BaseModel):
    """Something happened. Edge-triggered, for the agent and the UI.

    Carries enough for the agent to act without a follow-up call: which
    behaviour, a sentence a human can read, and a crop to look at.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ts: float = Field(default_factory=time.time)
    behavior_id: str | None = None  # null for system events
    type: EventType
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    snapshot_url: str | None = None
    #: False means "show it, do not wake the agent".
    notify: bool = True


# ---------------------------------------------------------------------------
# 4. Observations
# ---------------------------------------------------------------------------


class TrackView(BaseModel):
    """One tracked thing, normalized. What queries return."""

    model_config = ConfigDict(extra="forbid")

    track_id: int
    label: str
    conf: float
    cx: float
    cy: float
    area: float
    attributes: dict[str, float] = Field(default_factory=dict)


class StateView(BaseModel):
    """GET /state, and one message per frame on the state channel."""

    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    model: str | None = None
    fps: float = 0.0
    camera_ok: bool = True
    hud: str | None = None
    behaviors: list[BehaviorView] = Field(default_factory=list)
    tracks: list[TrackView] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list)


class Health(BaseModel):
    """GET /health. Deliberately small: status, fps, model, camera."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["booting", "ok", "no_camera", "fault"] = "booting"
    fps: float = 0.0
    model: str | None = None
    camera_ok: bool = False
    behaviors: int = 0
    device: str = "cpu"
    detail: str | None = None


# ---------------------------------------------------------------------------
# 5. Actuator
# ---------------------------------------------------------------------------


class MotorCommand(BaseModel):
    """What the tracker wants the view to do.

    The abstraction that survives the hardware: today a `VirtualMotor` draws an
    arrow for a person to follow, later a stepper executes it. The tracking
    logic is identical either way.
    """

    model_config = ConfigDict(extra="forbid")

    rate_deg_s: float = 0.0  # positive pans right
    reason: str = ""
    urgent: bool = False     # target is gone, not merely off-centre


# ---------------------------------------------------------------------------
# 6. Agent queries
# ---------------------------------------------------------------------------


class CountQuery(BaseModel):
    """POST /query/count. Median over a window, so one bad frame cannot lie."""

    model_config = ConfigDict(extra="forbid")

    selector: Selector
    window_s: float = Field(default=1.0, gt=0.0, le=10.0)


class CountResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    samples: int = 0
    window_s: float = 1.0


class LookQuery(BaseModel):
    """POST /query/look. What is in front of the camera right now."""

    model_config = ConfigDict(extra="forbid")

    selector: Selector | None = None


class LookResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    tracks: list[TrackView] = Field(default_factory=list)


class SceneObject(BaseModel):
    """One thing in the scene, described in words rather than coordinates.

    The agent and the vision model both read this, and neither wants pixels:
    "a laptop, centre-left, large" is usable in a sentence where
    `cx=-0.31, area=0.18` is not.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    where: str            # "centre", "lower left", "far right"
    size: str             # "small" | "medium" | "large" | "very large"
    conf: float
    count: int = 1        # several of the same thing in the same place
    cx: float = 0.0
    cy: float = 0.0
    area: float = 0.0


class DescribeRequest(BaseModel):
    """POST /describe. Everything needed to answer a question about the scene.

    Perception does not call a vision model — it has no LLM and should not
    grow one. It returns the frame, what it can actually see, and a prompt
    built from both; the agent supplies the model.
    """

    model_config = ConfigDict(extra="forbid")

    #: The operator's question, verbatim. Shapes the returned prompt only.
    question: str = "What do you see?"
    #: Classes to sweep for. Empty uses the built-in everyday vocabulary.
    candidates: list[str] = Field(default_factory=list, max_length=64)
    #: Run a detection sweep. Off means report only what behaviours already
    #: track, which is faster and much narrower.
    sweep: bool = True


class DescribeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    #: Fetch the raw, un-annotated frame here and give it to the vision model.
    snapshot_url: str = "/snapshot"
    objects: list[SceneObject] = Field(default_factory=list)
    #: One English sentence, usable as an answer on its own when the question
    #: is simple enough that no vision model is needed.
    summary: str = ""
    #: Ready to send to a vision model alongside the snapshot. Carries the
    #: grounding, so the model's answer and the overlay agree about the scene.
    prompt: str = ""
    #: What the camera is doing, so the answer can mention it.
    behaviors: list[str] = Field(default_factory=list)
    swept: bool = False
    detail: str | None = None


class PhraseResult(BaseModel):
    """How one wording fared against the frame in front of the camera."""

    model_config = ConfigDict(extra="forbid")

    phrase: str
    found: int = 0
    mean_conf: float = 0.0
    max_conf: float = 0.0
    #: Largest match as a fraction of the frame, so "found but tiny" is
    #: distinguishable from "found properly".
    max_area: float = 0.0


class ProbeRequest(BaseModel):
    """POST /probe. Does this wording actually find anything, right now?

    The debugging loop for the failure that looks like a broken pipeline and
    is really a word the detector has no match for. Measured on this project:
    "duck" finds nothing in frames where "yellow duck" finds it every time.
    """

    model_config = ConfigDict(extra="forbid")

    phrases: Annotated[list[str], Field(min_length=1, max_length=6)]


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    results: list[PhraseResult] = Field(default_factory=list)
    #: The wording that worked best, ready to put straight into `detect`.
    best: str | None = None
    advice: str = ""
    detail: str | None = None


class HudText(BaseModel):
    """POST /hud. The instruction the operator typed, shown on the frame."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""


class ModelChoice(BaseModel):
    """POST /model. The role is inferred from the model unless given."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str | None = None


__all__ = [
    "BehaviorCreated",
    "BehaviorKind",
    "BehaviorSpec",
    "BehaviorState",
    "BehaviorView",
    "CountQuery",
    "CountResult",
    "DescribeRequest",
    "DescribeResult",
    "Event",
    "EventType",
    "Health",
    "HudText",
    "LookQuery",
    "LookResult",
    "ModelChoice",
    "MotorCommand",
    "Pick",
    "Relate",
    "RenderSpec",
    "SceneObject",
    "Selector",
    "StateView",
    "TrackView",
]
