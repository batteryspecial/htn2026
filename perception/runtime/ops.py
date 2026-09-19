"""Operations: the things the agent can ask the pipeline to change.

Every change funnels through one queue and one worker, which is what keeps the
swap atomic. An operation is prepared completely in the background and then
installed between two frames; if preparing it fails, nothing changed and the
pipeline keeps running exactly as it was.

Unlike the single-spec model this replaces, operations are **cumulative**:
adding a behaviour leaves the others alone. So the newest operation does not
supersede the pending one, it queues behind it. What is superseded is only an
operation that is still pending when the same behaviour is changed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from contracts import Behavior

OpKind = Literal["add_behavior", "remove_behavior", "clear", "set_model"]


@dataclass(frozen=True)
class Op:
    kind: OpKind
    instruction_id: str
    seq: int = 0
    behavior: Behavior | None = None
    behavior_id: str | None = None
    model: str | None = None

    @property
    def target(self) -> str:
        """What this operation is about, for superseding and for logs."""
        if self.kind == "set_model":
            return "model"
        if self.kind == "clear":
            return "all"
        return self.behavior_id or (self.behavior.behavior_id if self.behavior else "?")

    def describe(self) -> str:
        return f"{self.kind}({self.target})"


@dataclass(frozen=True)
class Applied:
    """The world was rebuilt and installed."""

    op: Op
    world: object
    seconds: float
    reset_tracking: bool = False


@dataclass(frozen=True)
class Rejected:
    op: Op
    reason: str


@dataclass(frozen=True)
class Accepted:
    op: Op


@dataclass(frozen=True)
class ModelLoading:
    op: Op
    model: str


@dataclass(frozen=True)
class ModelLoaded:
    op: Op
    model: str
    seconds: float


Outcome = Accepted | Applied | Rejected | ModelLoading | ModelLoaded
