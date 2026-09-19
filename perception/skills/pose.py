"""Body keypoints, for gesture triggers.

A pose model is an aux model: it runs on top of the detector rather than
instead of it, so it only runs while a behaviour actually wants keypoints.
With no gesture behaviour active it costs nothing.

Keypoints come back in COCO order, which is what every Ultralytics pose model
emits and what the gesture rules below assume.
"""

from __future__ import annotations

import logging

import numpy as np

from config import CFG, quantize, resolve_device

log = logging.getLogger("perception.pose")

#: COCO keypoint indices. The only ones the gestures use are named.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

#: Below this a keypoint is a guess, not an observation.
MIN_KP_CONF = 0.5


class UltralyticsPose:
    """Any Ultralytics `-pose` checkpoint. Swapping the gesture model for a
    full-body one is a different weights file and nothing else."""

    role = "pose"

    def __init__(self, name: str, weights) -> None:
        self.name = name
        self.weights = weights
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        from detectors.base import use_weights_dir
        from ultralytics import YOLO

        use_weights_dir()
        log.info("loading pose model %s from %s", self.name, self.weights)
        self._model = YOLO(str(self.weights))
        if resolve_device() != "cpu":
            self._model.to(resolve_device())
        self._loaded = True

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._model is None:
            return
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.keypoints(blank)

    def keypoints(self, frame: np.ndarray) -> np.ndarray:
        """(people, 17, 3) of x, y, confidence in frame pixels.

        An empty array when nobody is found, so callers never branch on None.
        """
        result = self._model.predict(
            frame, imgsz=CFG.IMGSZ, conf=CFG.CONF_THRESHOLD,
            quantize=quantize(), verbose=False,
        )[0]
        kp = getattr(result, "keypoints", None)
        if kp is None or kp.data is None or len(kp.data) == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return kp.data.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------


def hand_raised(person: np.ndarray) -> bool:
    """A wrist above the shoulder on the same side.

    Image y grows downward, so "above" is a smaller y. Comparing against the
    person's own shoulder rather than a fixed height is what makes this work
    for someone sitting, standing, near or far, without calibration.
    """
    for wrist, shoulder in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        w, s = person[wrist], person[shoulder]
        if w[2] < MIN_KP_CONF or s[2] < MIN_KP_CONF:
            continue
        if w[1] < s[1]:
            return True
    return False


GESTURES = {"hand_raised": hand_raised}


def detect_gesture(keypoints: np.ndarray, gesture: str) -> list[int]:
    """Indices of the people performing a gesture."""
    rule = GESTURES.get(gesture)
    if rule is None:
        raise ValueError(f"unknown gesture {gesture!r}; known: {sorted(GESTURES)}")
    return [i for i in range(len(keypoints)) if rule(keypoints[i])]


def person_box(person: np.ndarray) -> tuple[float, float, float, float] | None:
    """A bounding box around the confident keypoints, for drawing and events."""
    good = person[person[:, 2] >= MIN_KP_CONF]
    if len(good) < 2:
        return None
    return (float(good[:, 0].min()), float(good[:, 1].min()),
            float(good[:, 0].max()), float(good[:, 1].max()))
