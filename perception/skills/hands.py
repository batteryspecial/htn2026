"""Hand landmarks, for finger gestures.

Aux model #2, and it follows every rule `skills/pose.py` set: it runs only
while a behaviour wants it, its failure degrades that behaviour rather than
the frame, and a behaviour whose model is missing is paused with a reason.

**Why a separate role from `pose` rather than a second pose model.** Exactly
one model per role is active at a time, so sharing the role would make "tell
me when someone raises their hand" and "tell me when someone raises their
index finger" mutually exclusive — installing one would pause the other. They
answer different questions and both want to run.

MediaPipe rather than an Ultralytics checkpoint because it is CPU-only at
~30fps, so it costs the detector no GPU, and needs no weights file. COCO pose
has wrists and no fingers, which is why a finger gesture was not merely hard
before this — it was unrepresentable.

Landmarks come back in MediaPipe order, 21 per hand.
"""

from __future__ import annotations

import logging

import numpy as np

from config import CFG, resolve_device

log = logging.getLogger("perception.hands")

#: MediaPipe hand landmark indices. Only the ones the rules use are named.
WRIST = 0
THUMB_TIP = 4

#: finger -> (mcp, pip, tip). The thumb is deliberately absent: it folds
#: sideways across the palm rather than curling toward the wrist, so the
#: distance test below does not describe it and every rule here ignores it.
FINGERS: dict[str, tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

#: Landmarks per hand, as MediaPipe emits them.
N_LANDMARKS = 21


class MediaPipeHands:
    """MediaPipe Hands. No weights file: the graph ships with the package.

    Held open across frames — the tracker inside it is what makes it cheap,
    so it is constructed once at load and reused, never per frame.
    """

    role = "hands"

    def __init__(self, name: str, weights=None) -> None:
        self.name = name
        # Accepted and ignored, so a models.yaml entry looks like every other.
        self.weights = weights
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        import mediapipe as mp

        log.info("loading mediapipe hands (max %d)", CFG.MAX_HANDS)
        self._model = mp.solutions.hands.Hands(
            static_image_mode=False,      # video: track between frames, cheaper
            max_num_hands=CFG.MAX_HANDS,
            min_detection_confidence=CFG.HAND_DETECTION_CONF,
            min_tracking_confidence=CFG.HAND_TRACKING_CONF,
        )
        self._loaded = True
        if resolve_device() != "cpu":
            # Not a warning: this is the point of it. It leaves the GPU to the
            # detector, which is the model that actually needs one.
            log.info("mediapipe hands runs on CPU regardless of DEVICE")

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._model is None:
            return
        self.keypoints(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    def keypoints(self, frame: np.ndarray) -> np.ndarray:
        """(hands, 21, 3) of x, y, confidence in frame pixels.

        Same shape and units as the pose model's keypoints, so everything
        downstream — the subject test, the boxes, the event crop — is shared
        rather than reimplemented. Empty when no hand is found, so callers
        never branch on None.

        The third column is 1.0 throughout. MediaPipe reports a score per
        *hand*, not per landmark, and only returns a hand it is already
        confident about; a per-landmark number here would be invented. The
        rules below therefore do not filter on it, unlike the pose rules.
        """
        import cv2

        # MediaPipe wants RGB; every frame in this service is BGR from OpenCV.
        result = self._model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        found = getattr(result, "multi_hand_landmarks", None)
        if not found:
            return np.zeros((0, N_LANDMARKS, 3), dtype=np.float32)

        h, w = frame.shape[:2]
        return np.array(
            [[(lm.x * w, lm.y * h, 1.0) for lm in hand.landmark] for hand in found],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------


def _extended(hand: np.ndarray, finger: str) -> bool:
    """Whether a finger is straight, by distance from the wrist.

    The obvious test — tip above the middle joint — silently assumes an
    upright hand, and a hand pointing sideways or held palm-down is still a
    hand. Distance from the wrist holds in any orientation: curling a finger
    brings its tip back toward the palm, so an extended tip is always the
    farther of the two and a curled one is always the nearer.
    """
    _, pip, tip = FINGERS[finger]
    wrist = hand[WRIST][:2]
    return bool(
        np.linalg.norm(hand[tip][:2] - wrist) > np.linalg.norm(hand[pip][:2] - wrist)
    )


def index_finger_raised(hand: np.ndarray) -> bool:
    """Index out, the other three curled.

    The other three matter: without them an open palm also has an extended
    index, so "raise your index finger" would fire on a wave.
    """
    return _extended(hand, "index") and not any(
        _extended(hand, f) for f in ("middle", "ring", "pinky")
    )


def open_palm(hand: np.ndarray) -> bool:
    """All four fingers out. The thumb is not consulted."""
    return all(_extended(hand, f) for f in FINGERS)


def fist(hand: np.ndarray) -> bool:
    """All four fingers curled. The thumb is not consulted."""
    return not any(_extended(hand, f) for f in FINGERS)


GESTURES = {
    "index_finger_raised": index_finger_raised,
    "open_palm": open_palm,
    "fist": fist,
}
