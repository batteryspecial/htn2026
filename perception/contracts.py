"""Perception's view of the shared contracts.

The definitions live in `linker/schemas.py`, which all four services share.
This module re-exports them so perception can keep importing `contracts`
regardless of where the process was started from, and so there is exactly one
copy of the wire format in the repo. Two copies of a wire contract drift, and
the drift only shows up during integration.

Anything perception needs that the team has not agreed to yet goes below the
re-export and is flagged in the reply.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from linker.schemas import (  # noqa: E402
    RUNNING_PHASES,
    Arbitration,
    Behavior,
    BehaviorKind,
    BehaviorRequest,
    BehaviorState,
    BehaviorStatus,
    EventKind,
    FrameSummary,
    Mode,
    PerceptionEvent,
    Phase,
    QueryKind,
    QueryRequest,
    QueryResult,
    Relate,
    SelectRule,
    SpecRequest,
    Stage,
    StatusEvent,
    Subject,
    Target,
    TargetState,
    TaskSpec,
    Track,
    Verify,
)

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
