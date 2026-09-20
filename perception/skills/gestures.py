"""Every gesture the device knows, and which model has to be loaded to see it.

Two skills produce keypoints — body (`pose`, 17 COCO points) and hands
(`hands`, 21 MediaPipe points) — and a gesture reads exactly one of them. This
is the only place that knows which, so `pose_trigger` can declare the role it
needs per gesture rather than per kind, and `/models` can report one honest
vocabulary instead of two partial ones.

That vocabulary is what the agent reads. Without it the compiler sees only
that `pose_trigger` exists, still has to fill in a `gesture`, and settles for
the nearest implemented value — which is how "report a raised index finger"
became `hand_raised`, armed cleanly, and then watched wrists forever.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from skills import hands, pose

#: gesture name -> (role that produces its keypoints, the rule)
_RULES: dict[str, tuple[str, Callable[[np.ndarray], bool]]] = {
    **{name: ("pose", fn) for name, fn in pose.GESTURES.items()},
    **{name: ("hands", fn) for name, fn in hands.GESTURES.items()},
}

# A name defined by both skills would silently resolve to one of them, and the
# wrong model would be loaded for it. Cheap to assert, miserable to debug.
assert len(_RULES) == len(pose.GESTURES) + len(hands.GESTURES), (
    "a gesture name is defined in both skills/pose.py and skills/hands.py")

#: gesture name -> the aux role it needs. `pose_trigger` reads this to decide
#: what to declare in `needs_roles`, so an unloaded model pauses it with a
#: reason instead of leaving it armed and blind.
ROLE_OF: dict[str, str] = {name: role for name, (role, _) in _RULES.items()}


def known() -> list[str]:
    """Every gesture that is implemented, loaded or not."""
    return sorted(_RULES)


def available(has_role: Callable[[str], bool]) -> list[str]:
    """Every gesture this machine can actually observe right now.

    Narrower than `known()`: a rule whose model is not loaded is not a
    capability, and reporting it would have the agent install a behaviour that
    can never fire.
    """
    return sorted(name for name, (role, _) in _RULES.items() if has_role(role))


def detect(keypoints: np.ndarray, gesture: str) -> list[int]:
    """Indices of the subjects performing a gesture.

    `keypoints` must be the array for `ROLE_OF[gesture]` — hand landmarks for
    a finger gesture, body keypoints for a body one.
    """
    entry = _RULES.get(gesture)
    if entry is None:
        raise ValueError(f"unknown gesture {gesture!r}; known: {known()}")
    _, rule = entry
    return [i for i in range(len(keypoints)) if rule(keypoints[i])]
