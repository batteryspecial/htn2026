"""Turning a frame's worth of tracks into at most one target.

Everything here is a pure function over `supervision.Detections` plus a small
piece of per-target memory, so the whole stage is testable without a model, a
camera or a thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import supervision as sv

log = logging.getLogger("perception.select")


@dataclass
class SelectState:
    """Per-target memory that survives between frames but not between specs.

    A fresh one is built for every PreparedTask, which is what makes a retask
    forget the previous target instead of chasing it.
    """

    locked_track_id: int | None = None
    embedding: Any = None  # CLIP image embedding for re-ID; not yet populated
    extra: dict = field(default_factory=dict)


def candidates_for(dets: sv.Detections, classes: list[str]) -> sv.Detections:
    """Narrow a frame's detections to the classes one target cares about.

    Both targets are detected in a single pass on the union of their prompts,
    so this is where the single result gets split back apart.
    """
    if len(dets) == 0:
        return dets
    names = dets.data.get("class_name")
    if names is None:
        return dets
    wanted = set(classes)
    return dets[np.array([n in wanted for n in names], dtype=bool)]


def _areas(dets: sv.Detections) -> np.ndarray:
    return dets.box_area


def _center_offsets(dets: sv.Detections, frame_shape: tuple[int, int]) -> np.ndarray:
    """Squared distance of each box centre from the frame centre, normalized."""
    h, w = frame_shape[:2]
    cx = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2 / w - 0.5
    cy = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2 / h - 0.5
    return cx * cx + cy * cy


def pick(
    dets: sv.Detections,
    rule: str,
    frame_shape: tuple[int, int],
    state: SelectState | None = None,
) -> int | None:
    """Index of the chosen detection, or None when there is nothing to choose."""
    if len(dets) == 0:
        if state is not None:
            state.locked_track_id = state.locked_track_id  # kept: re-ID may recover it
        return None

    if rule == "largest":
        return int(np.argmax(_areas(dets)))
    if rule == "most_centered":
        return int(np.argmin(_center_offsets(dets, frame_shape)))
    if rule == "highest_conf":
        conf = dets.confidence
        return 0 if conf is None else int(np.argmax(conf))
    if rule == "locked":
        return _pick_locked(dets, frame_shape, state)
    raise ValueError(f"unknown select rule {rule!r}")


def _pick_locked(dets: sv.Detections, frame_shape, state: SelectState | None) -> int | None:
    """Stay on one instance rather than one class.

    Latches onto the first target it selects and holds that tracker id for as
    long as the tracker keeps producing it. When the id disappears it falls
    back to the largest candidate and latches again.

    ponytail: identity here is only the tracker id, so an id swap between two
    similar objects will follow the wrong one. The CLIP re-ID step, and Cutie
    behind it, both slot in at exactly this point.
    """
    if state is None:
        return int(np.argmax(_areas(dets)))

    ids = dets.tracker_id
    if state.locked_track_id is not None and ids is not None:
        hit = np.flatnonzero(ids == state.locked_track_id)
        if hit.size:
            return int(hit[0])

    idx = int(np.argmax(_areas(dets)))
    if ids is not None and idx < len(ids):
        new_id = int(ids[idx])
        if new_id != state.locked_track_id:
            log.info("locked target -> track %s (was %s)", new_id, state.locked_track_id)
        state.locked_track_id = new_id
    return idx
