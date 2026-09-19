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
    MIN_MATCHING_THRESHOLD: float = _f("MIN_MATCHING_THRESHOLD", 0.8)
    TRACK_FRAME_RATE: int = _i("TRACK_FRAME_RATE", 30)

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
    MIN_BOX_FRAC: float = _f("MIN_BOX_FRAC", 0.0015)
    VERIFY_BATCH: int = _i("VERIFY_BATCH", 16)
    RELATE_LOWER_FRAC: float = _f("RELATE_LOWER_FRAC", 0.4)
    LOCK_SIM_THRESHOLD: float = _f("LOCK_SIM_THRESHOLD", 0.8)
    LOCK_EMA_ALPHA: float = _f("LOCK_EMA_ALPHA", 0.1)
    LOCK_EMA_MIN_CONF: float = _f("LOCK_EMA_MIN_CONF", 0.6)
    ARBITRATE_STARVE_S: float = _f("ARBITRATE_STARVE_S", 1.0)

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
