"""ByteTrack's activation gate, which is not the parameter you set.

`det_thresh` is `track_activation_threshold + 0.1`, and Step 4 of
`update_with_tensors` skips any detection under it. The stock 0.25 therefore
refuses to create a track below **0.35**.

That is right for COCO, where a real object scores 0.8+. It silently breaks
open-vocabulary prompts, which are the point of this service: measured on
yoloe-11s with the object genuinely in frame, "yellow duck" scores 0.18-0.27
and "person's face" 0.12-0.28, while "person" scores 0.87. So the detector
finds the subject on every frame, the tracker creates no track, and the
behaviour layer reports no subjects while the logs show inference succeeding.

Nothing else in the suite exercises a confidence between the detector's cutoff
and 0.35, which is why this can regress without anything going red.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from config import CFG
from runtime.world import Shared, new_tracker

#: What an open-vocabulary detector returns for a correct, in-frame subject.
MARGINAL_CONF = 0.20
FRAME_AREA = 640 * 480


def detection(conf: float, name: str = "yellow duck") -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[260.0, 180.0, 380.0, 300.0]], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
        class_id=np.array([0]),
        data={"class_name": np.array([name], dtype=object)},
    )


def tracked_frames(update, conf: float, frames: int = 6) -> int:
    """How many frames a stable box of this confidence survives tracking."""
    return sum(1 for _ in range(frames) if len(update(detection(conf))) > 0)


def test_the_gate_is_the_threshold_plus_one_tenth():
    """Pin the surprise itself rather than trusting the parameter's name."""
    assert ByteTrack(track_activation_threshold=0.25).det_thresh == pytest.approx(0.35)
    assert ByteTrack(track_activation_threshold=0.02).det_thresh == pytest.approx(0.12)


def test_stock_bytetrack_discards_open_vocab_confidences():
    """The bug, pinned. If this starts failing, supervision changed under us."""
    stock = ByteTrack()
    assert tracked_frames(stock.update_with_detections, MARGINAL_CONF) == 0


def test_new_tracker_keeps_them():
    assert CFG.TRACK_ACTIVATION_THRESHOLD + 0.1 <= MARGINAL_CONF, (
        "TRACK_ACTIVATION_THRESHOLD + 0.1 must sit below the confidences "
        "open-vocabulary prompts actually produce"
    )
    assert tracked_frames(new_tracker().update_with_detections, MARGINAL_CONF) >= 3


def test_shared_uses_the_configured_tracker():
    """The wiring, not just the value: Shared used to default to ByteTrack()."""
    shared = Shared()
    shared.frame_area = FRAME_AREA
    assert tracked_frames(shared.track, MARGINAL_CONF) >= 3, (
        "Shared.tracker is dropping open-vocabulary detections"
    )


def test_reset_does_not_restore_the_stock_gate():
    """reset() rebuilds the tracker, and used to rebuild it bare."""
    shared = Shared()
    shared.frame_area = FRAME_AREA
    shared.reset("model switch")
    assert tracked_frames(shared.track, MARGINAL_CONF) >= 3


def test_detector_cutoff_admits_them_first():
    """A permissive tracker gate is pointless if the detector filters earlier."""
    assert CFG.CONF_THRESHOLD <= MARGINAL_CONF, (
        f"CONF_THRESHOLD={CFG.CONF_THRESHOLD} rejects correct open-vocabulary "
        f"detections before the tracker is reached"
    )
