"""Wire contracts for perception.

Mirror of `linker/`, which is currently empty. Four additive proposals live here:

1. TargetState gains `phase`, `model`, `target_ref`
2. StatusEvent.stage gains `model_loading`, `model_loaded`
3. TaskSpec.model is `str`, validated against the registry
4. GET /models exists, so the orchestrator builds its manifest from it

Output is deliberately hardware-agnostic: offsets are normalized to [-1, 1] and
area is a frame fraction, so no consumer needs the resolution, lens, or chassis.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Phase(str, Enum):
    """Level-triggered pipeline state, published on every TargetState.

    The controller drives only in TRACKING. Every other phase means stop.
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
        return self is Phase.TRACKING


# Alive and willing to accept a new spec. BOOTING and FAULT are not.
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


class Verify(BaseModel):
    """Attribute check on a track's crop, scored contrastively by CLIP."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cls: str = Field(alias="class")
    text: str  # positive prompt, e.g. "a red shoe"
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class Relate(BaseModel):
    """Keep a `keep` box only when it contains an `if_contains` box low inside it."""

    model_config = ConfigDict(extra="forbid")

    keep: str
    if_contains: str


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    detect: Annotated[list[str], Field(min_length=1, max_length=6)]
    verify: Verify | None = None
    relate: Relate | None = None
    select: SelectRule = "largest"


class Arbitration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alternate_s: float = Field(default=3.0, gt=0.0)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    targets: Annotated[list[Target], Field(min_length=1, max_length=2)]
    mode: Mode = "follow"
    arbitration: Arbitration = Field(default_factory=Arbitration)
    # Validated against the live registry, not the schema, so adding a model
    # never means editing this file.
    model: str = "yoloe"

    def prompt_union(self) -> list[str]:
        """Every class name this spec needs, deduplicated, order preserved.

        One detector call covers both targets, so the detector is prepared on
        the union rather than once per target.
        """
        names: list[str] = []
        for t in self.targets:
            extra = [t.relate.if_contains] if t.relate else []
            for n in (*t.detect, *extra):
                if n not in names:
                    names.append(n)
        return names


class TargetState(BaseModel):
    """One per frame, for the controller. Level-triggered."""

    model_config = ConfigDict(extra="forbid")

    ts: float = Field(default_factory=time.time)
    spec_id: str | None = None
    mode: Literal["follow", "center", "idle"] = "idle"
    visible: bool = False
    cx: float = 0.0  # [-1, 1], right positive
    cy: float = 0.0  # [-1, 1], down positive
    area: float = 0.0  # bbox area / frame area
    conf: float = 0.0
    label: str | None = None
    track_id: int | None = None
    phase: Phase = Phase.BOOTING
    model: str | None = None
    target_ref: str | None = None


class StatusEvent(BaseModel):
    """Edge-triggered progress, for humans via the orchestrator.

    Never for the controller: a consumer that misses one event gets stuck.
    """

    model_config = ConfigDict(extra="forbid")

    stage: Stage
    ts: float = Field(default_factory=time.time)
    instruction_id: str | None = None
    spec_id: str | None = None
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class SpecRequest(BaseModel):
    """Body of POST /spec."""

    model_config = ConfigDict(extra="forbid")

    instruction_id: str
    spec: TaskSpec
