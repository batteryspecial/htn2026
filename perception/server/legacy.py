"""Compatibility with the TaskSpec contract the frontend was built against.

The frontend and its in-browser compiler predate the behaviour layer: they
emit a `TaskSpec` with `targets[]`, `verify`, `relate` and `select`, and they
listen for stage events named `prepared`, `applied`, `active`.

Rather than ask that side to change while it is being finished, this translates
in both directions. It is deliberately a thin edge adapter, not a second code
path: a TaskSpec becomes ordinary behaviours the moment it arrives, and nothing
downstream knows this file exists.

Delete it once both sides speak `BehaviorSpec` natively.
"""

from __future__ import annotations

import logging
from typing import Any

from contracts import BehaviorSpec, Event

log = logging.getLogger("perception.legacy")

#: Old select rules to new pick rules. `highest_conf` has no equivalent —
#: nothing downstream ranked by confidence — so it degrades to `largest` and
#: says so rather than silently picking something else.
PICK = {
    "largest": "largest",
    "most_centered": "most_centered",
    "highest_conf": "largest",
    "locked": "ref",
}

#: Behaviour events mapped onto the stage names the frontend's strip knows.
STAGE = {
    "acquired": "active",
    "behavior_failed": "failed",
    "camera_lost": "failed",
}


def to_behaviors(spec: dict[str, Any]) -> tuple[list[BehaviorSpec], list[str]]:
    """Turn one TaskSpec into behaviours. Returns them plus any notes.

    Each target becomes a `track` behaviour, because that is what the old
    contract meant: one thing, followed. `mode` is dropped — it chose between
    driving and rotating, and there is no car.
    """
    notes: list[str] = []
    targets = spec.get("targets") or []
    if not targets:
        raise ValueError("TaskSpec has no targets")

    out: list[BehaviorSpec] = []
    for i, t in enumerate(targets):
        detect = list(t.get("detect") or [])
        if not detect:
            raise ValueError(f"targets[{i}].detect is empty")

        select = t.get("select", "largest")
        if select == "highest_conf":
            notes.append(f"targets[{i}].select 'highest_conf' has no equivalent; "
                         f"using 'largest'")
        subject: dict[str, Any] = {"detect": detect, "pick": PICK.get(select, "largest")}

        verify = t.get("verify")
        if verify and verify.get("text"):
            # An old `verify` is an include with no competitor, which is the
            # weaker of the two attribute regimes. Worth saying out loud.
            subject["include"] = [verify["text"]]
            if verify.get("min_score") is not None:
                subject["min_score"] = float(verify["min_score"])
            notes.append(f"targets[{i}].verify became include={verify['text']!r}; "
                         f"pair it with an exclude for steadier matching")

        relate = t.get("relate")
        if relate and relate.get("if_contains"):
            subject["relate"] = {"contains": relate["if_contains"]}

        out.append(BehaviorSpec(
            kind="track", subject=subject,
            render={"label": t.get("ref") or "+".join(detect)},
        ))
    return out, notes


def to_stage(event: Event) -> dict[str, Any] | None:
    """Translate a behaviour event into a legacy stage message, or None.

    Most events have no old equivalent. Inventing a stage name for them would
    put unknown entries on the frontend's strip, so they are dropped here and
    remain available in full on /ws/events.
    """
    stage = STAGE.get(event.type)
    if stage is None:
        return None
    return {
        "instruction_id": event.data.get("instruction_id"),
        "stage": stage,
        "ts": event.ts,
        "detail": event.detail,
        "data": {"behavior_id": event.behavior_id, **event.data},
    }
