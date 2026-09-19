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
    CONF_THRESHOLD: float = _f("CONF_THRESHOLD", 0.25)

    # 3. Timing / state machine
    ACQUIRE_TIMEOUT_S: float = _f("ACQUIRE_TIMEOUT_S", 3.0)
    ACQUIRE_HITS: int = _i("ACQUIRE_HITS", 3)  # consecutive frames -> TRACKING
    LOST_MISSES: int = _i("LOST_MISSES", 5)  # consecutive frames -> LOST
    CAMERA_DEAD_S: float = _f("CAMERA_DEAD_S", 2.0)
    INFER_ERRORS_TO_FAULT: int = _i("INFER_ERRORS_TO_FAULT", 3)
    FAULT_HEARTBEAT_HZ: float = _f("FAULT_HEARTBEAT_HZ", 5.0)

    # 4. Stages
    VERIFY_TTL_S: float = _f("VERIFY_TTL_S", 1.0)  # re-verify a track this often
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
