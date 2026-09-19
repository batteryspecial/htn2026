"""Operations: the things the agent can ask the pipeline to change.

Every change funnels through one queue and one worker, which is what keeps the
swap atomic. An operation is prepared completely in the background and then
installed between two frames; if preparing it fails, nothing changed.

Operations are **cumulative**: adding a behaviour leaves the others alone. So
a new operation does not supersede a pending one, it queues behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from contracts import BehaviorSpec

OpKind = Literal["add_behavior", "remove_behavior", "clear", "set_model",
                 "add_reference"]


@dataclass(frozen=True)
class Op:
    kind: OpKind
    seq: int = 0
    behavior_id: str | None = None
    spec: BehaviorSpec | None = None
    model: str | None = None
    ref_id: str | None = None
    image: Any = None          # np.ndarray, for add_reference
    vector: Any = None         # a pre-computed embedding, for "that one there"
    label: str | None = None

    def describe(self) -> str:
        target = self.model or self.behavior_id or "all"
        return f"{self.kind}({target})"


@dataclass(frozen=True)
class Accepted:
    op: Op


@dataclass(frozen=True)
class Applied:
    op: Op
    world: object
    seconds: float
    reset_tracking: bool = False


@dataclass(frozen=True)
class Rejected:
    op: Op
    reason: str


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
