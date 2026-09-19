"""Choosing one subject out of several.

`pick` on the selector decides which. `ref` is the one with memory: it latches
onto an instance and stays on it, which is the difference between "track a
person" and "track *that* person".
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv

log = logging.getLogger("perception.pick")


def pick(dets: sv.Detections, rule: str, frame_shape, behavior=None) -> int | None:
    """Index of the chosen detection, or None when there is nothing to choose."""
    if len(dets) == 0:
        return None
    if rule in ("all", "largest"):
        return int(np.argmax(dets.box_area))
    if rule == "most_centered":
        h, w = frame_shape[:2]
        cx = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2 / w - 0.5
        cy = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2 / h - 0.5
        return int(np.argmin(cx * cx + cy * cy))
    if rule == "ref":
        return _latched(dets, behavior)
    raise ValueError(f"unknown pick rule {rule!r}")


def _latched(dets: sv.Detections, behavior) -> int | None:
    """Stay on one instance rather than one class.

    Holds a tracker id for as long as the tracker keeps producing it. When the
    id disappears the behaviour is LOST, and reacquiring is deliberately *not*
    handled here: matching appearance is what stops a passerby from being
    adopted as the target, and that lives in the track behaviour.
    """
    if behavior is None:
        return int(np.argmax(dets.box_area))

    ids = dets.tracker_id
    locked = getattr(behavior, "locked_track_id", None)
    if locked is not None and ids is not None:
        hit = np.flatnonzero(ids == locked)
        if hit.size:
            return int(hit[0])
        return None  # the latched instance is not here; do not substitute

    idx = int(np.argmax(dets.box_area))
    if ids is not None and idx < len(ids):
        behavior.locked_track_id = int(ids[idx])
        log.info(
            "%s latched onto track %s",
            getattr(behavior, "id", "?"),
            behavior.locked_track_id
        )
    return idx
