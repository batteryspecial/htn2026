"""Perception's view of the shared contracts.

The definitions live in `linker/schemas.py`, which every service shares. This
re-exports them so perception imports `contracts` regardless of where the
process was started from, and so the repo holds exactly one copy of the wire
format. Two copies drift, and the drift surfaces during integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from linker.schemas import (  # noqa: E402,F401
    BehaviorCreated,
    BehaviorKind,
    BehaviorSpec,
    BehaviorState,
    BehaviorView,
    CountQuery,
    CountResult,
    Event,
    EventType,
    Health,
    HudText,
    LookQuery,
    LookResult,
    ModelChoice,
    MotorCommand,
    Pick,
    Relate,
    RenderSpec,
    Selector,
    StateView,
    TrackView,
)

__all__ = [
    "BehaviorCreated", "BehaviorKind", "BehaviorSpec", "BehaviorState",
    "BehaviorView", "CountQuery", "CountResult", "Event", "EventType",
    "Health", "HudText", "LookQuery", "LookResult", "ModelChoice",
    "MotorCommand", "Pick", "Relate", "RenderSpec", "Selector", "StateView",
    "TrackView",
]
