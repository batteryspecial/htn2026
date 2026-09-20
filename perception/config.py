"""Environment-driven config. Import this, never os.environ directly.

Nothing here imports torch at module level; `resolve_device()` does it lazily so
the contracts and the test suite stay fast to import.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _f(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(_env(name, str(default)))


class Config:
    # 1. Video
    # - webcam index ("0")
    # - a file path (loops forever)
    # - a stream URL
    VIDEO_SOURCE: str = _env("VIDEO_SOURCE", "0")
    IMGSZ: int = _i("IMGSZ", 640)  # 320 on the Mac
    CAPTURE_FPS_CAP: float = _f("CAPTURE_FPS_CAP", 60.0)
    STREAM_FPS: float = _f("STREAM_FPS", 15.0)  # MJPEG encode rate
    JPEG_QUALITY: int = _i("JPEG_QUALITY", 70)

    # 2. Compute
    DEVICE: str = _env("DEVICE", "auto")  # auto | cuda | mps | cpu
    # 0.25 is the right cutoff for a COCO detector, where a real object scores
    # 0.8+. Open-vocabulary matching is not that: measured on yoloe-11s with the
    # subject genuinely in frame, "yellow duck" scores 0.18-0.27 and
    # "person's face" 0.12-0.28, so 0.25 rejects about half of the correct
    # detections for exactly the objects a judge will name.
    #
    # The trade-off is real and points the other way for counting: a lower
    # cutoff admits more texture boxes, which MIN_BOX_FRAC only partly catches
    # because junk is small *and* low-confidence. `track` and `highlight` want
    # this low; `count_line` wants it high. Worth revisiting per behaviour kind
    # rather than leaving one global number to serve both.
    CONF_THRESHOLD: float = _f("CONF_THRESHOLD", 0.15)

    # ByteTrack will not *create* a track below (this + 0.1): det_thresh is
    # track_activation_threshold + 0.1, and Step 4 of update_with_tensors skips
    # anything under it. The stock 0.25 therefore gates at 0.35, which is right
    # for COCO (a real object scores 0.8+) and wrong for open-vocabulary text
    # matching, where a correct detection often sits at 0.15-0.30. Measured on
    # yoloe-11s: a duck present scores 0.18-0.27 and is detected on every
    # frame, yet the stock tracker creates no track at all, so the behaviour
    # layer sees nothing and reports no subjects while the logs show inference
    # succeeding. Keep this at or below CONF_THRESHOLD - 0.1 so the detector's
    # cutoff is the only one that bites.
    TRACK_ACTIVATION_THRESHOLD: float = _f("TRACK_ACTIVATION_THRESHOLD", 0.02)
    LOST_TRACK_BUFFER: int = _i("LOST_TRACK_BUFFER", 60)

    # ByteTrack does not match a track to a detection on overlap alone. It
    # multiplies them (matching.fuse_score): cost = 1 - IoU * score. Even at a
    # perfect overlap of 1.0 that collapses to cost = 1 - score, so a track
    # only re-matches itself when
    #
    #     score >= 1 - MIN_MATCHING_THRESHOLD
    #
    # This is a *second* gate, independent of det_thresh, and it decides
    # whether a track survives rather than whether it is born. At the stock
    # 0.8 the floor is 0.20: measured, a detection at 0.199 creates a track on
    # one frame and is then abandoned forever, while 0.21 tracks perfectly.
    # Everything in the 0.15-0.20 band therefore flickered — detected every
    # frame, tracked for one — which reads as "it only works up close".
    #
    # Left at stock. Raising it was the first fix attempted and it works, but
    # TRACKER_CONF_FLOOR below supersedes it: once confidences are mapped into
    # the range ByteTrack expects, the stock gate is already comfortably below
    # them. Stock is the better place to be — raising this makes matching more
    # permissive, so two heavily overlapping objects grow likelier to swap ids,
    # and the follow-cam depends on them not doing that.
    MIN_MATCHING_THRESHOLD: float = _f("MIN_MATCHING_THRESHOLD", 0.8)
    TRACK_FRAME_RATE: int = _i("TRACK_FRAME_RATE", 30)

    # A third gate, and the only one that cannot be configured: ByteTrack
    # confirms a newly created track by re-matching it on the next frame with
    # `thresh=0.7` written inline (core.py, "Deal with unconfirmed tracks").
    # Fused with score as everywhere else, that demands `score >= 0.30`. Below
    # it a track is born, fails to confirm, is removed, and is born again the
    # next frame — which looks like a mask flashing rather than like a
    # threshold. Measured: 0.2 never confirms, 0.3 confirms on frame two.
    #
    # ByteTrack's constants assume a COCO-style detector where a real object
    # scores 0.8+. Open-vocabulary matching is a text-image similarity on a
    # different scale entirely, where 0.15-0.30 is a correct detection. So
    # rather than patch a hardcoded constant in a dependency, confidences are
    # mapped into the range ByteTrack was designed for on the way in and
    # restored on the way out. Monotonic, so nothing downstream sees a
    # different ordering, and version-independent.
    TRACKER_CONF_FLOOR: float = _f("TRACKER_CONF_FLOOR", 0.35)

    # 3. Timing / state machine
    ACQUIRE_TIMEOUT_S: float = _f("ACQUIRE_TIMEOUT_S", 3.0)
    ACQUIRE_HITS: int = _i("ACQUIRE_HITS", 3)  # consecutive frames -> TRACKING
    LOST_MISSES: int = _i("LOST_MISSES", 5)  # consecutive frames -> LOST
    CAMERA_DEAD_S: float = _f("CAMERA_DEAD_S", 2.0)
    INFER_ERRORS_TO_FAULT: int = _i("INFER_ERRORS_TO_FAULT", 3)
    FAULT_HEARTBEAT_HZ: float = _f("FAULT_HEARTBEAT_HZ", 5.0)

    # 4. Stages
    VERIFY_TTL_S: float = _f("VERIFY_TTL_S", 1.0)  # re-verify a track this often
    # Score clothing on the upper body rather than the whole box.
    ATTR_TIGHTEN: bool = _env("ATTR_TIGHTEN", "1") == "1"
    # Drop detections smaller than this fraction of the frame. Open-vocabulary
    # detectors invent small boxes on texture, and a junk box sails through any
    # exclusion and inflates the count.
    # Reject degenerate boxes: a minimum *side*, in pixels, not an area.
    #
    # This replaces a fraction-of-frame-area floor that was wrong twice over.
    # It scaled quadratically with resolution, so a sharper camera — which
    # sees more detail — got a *higher* cutoff: at 1080p the old 0.0015
    # demanded a 56x56 box, which throws away any face past about 3.5 m.
    # And it never caught the junk it was added for, because open-vocabulary
    # texture detections are tall narrow strips with plenty of area.
    #
    # 12 px kills 3-pixel noise and nothing a person would call an object.
    MIN_BOX_PX: int = _i("MIN_BOX_PX", 12)
    # Area floor as a fraction of the frame. Off by default; the reason it was
    # wrong is above. Raise it only for a fixed camera where you know how big
    # the things you care about are.
    MIN_BOX_FRAC: float = _f("MIN_BOX_FRAC", 0.0)
    VERIFY_BATCH: int = _i("VERIFY_BATCH", 16)
    RELATE_LOWER_FRAC: float = _f("RELATE_LOWER_FRAC", 0.4)
    LOCK_SIM_THRESHOLD: float = _f("LOCK_SIM_THRESHOLD", 0.8)
    LOCK_EMA_ALPHA: float = _f("LOCK_EMA_ALPHA", 0.1)
    LOCK_EMA_MIN_CONF: float = _f("LOCK_EMA_MIN_CONF", 0.6)
    ARBITRATE_STARVE_S: float = _f("ARBITRATE_STARVE_S", 1.0)

    # 4b. Aux keypoint models
    #
    # Every number here is a calibration knob, not a constant: hand and body
    # detection thresholds are exactly what shifts between a bright demo table
    # and a dim conference hall, and they are the first thing to reach for when
    # a gesture reads as flaky on the day.
    #
    # Hands are counted per frame, not per person: two people each showing one
    # hand costs the same as one person showing two.
    MAX_HANDS: int = _i("MAX_HANDS", 4)
    # MediaPipe's own defaults are 0.5/0.5. Detection is lowered because a hand
    # entering the frame is partly cut off; tracking is left high because the
    # cheap frames are the tracked ones.
    HAND_DETECTION_CONF: float = _f("HAND_DETECTION_CONF", 0.4)
    HAND_TRACKING_CONF: float = _f("HAND_TRACKING_CONF", 0.5)
    # rtmlib: lightweight | balanced | performance. `balanced` holds ~30fps on
    # CPU; drop to `lightweight` if the whole-body model starves the detector.
    WHOLEBODY_MODE: str = _env("WHOLEBODY_MODE", "balanced")

    # 5. Server
    HOST: str = _env("HOST", "0.0.0.0")
    PORT: int = _i("PORT", 8001)
    MODELS_CONFIG: Path = Path(_env("MODELS_CONFIG", str(_HERE / "models.yaml")))
    WEIGHTS_DIR: Path = Path(_env("WEIGHTS_DIR", str(_HERE / "weights")))
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
    TIMING_EVERY: int = _i("TIMING_EVERY", 100)  # log stage timings every N frames


CFG = Config()


@functools.cache
def resolve_device() -> str:
    """Resolve auto -> cuda -> mps -> cpu. Never hardcode a device elsewhere."""
    want = CFG.DEVICE.lower()
    if want != "auto":
        return want
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quantize() -> int | None:
    """Precision for ultralytics predict: 16 is fp16, None is fp32.

    fp16 only on CUDA. MPS half produces NaNs and CPU half is slower than fp32.
    """
    return 16 if resolve_device() == "cuda" else None


def setup_logging() -> None:
    """Called once from main. Tests leave logging alone."""
    logging.basicConfig(
        level=getattr(logging, CFG.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
