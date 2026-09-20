"""Whole-body keypoints: body, feet, face and both hands from one model.

Aux model #3, and the first one that overlaps the others. RTMPose in the
COCO-WholeBody layout, 133 points per person:

    0-16     body      COCO-17, the same order `skills/pose.py` assumes
    17-22    feet      big toe, small toe, heel, per side
    23-90    face      the standard 68-point layout
    91-111   left hand   21 points, the same topology `skills/hands.py` uses
    112-132  right hand  21 points, likewise

Two of those blocks are layouts this service already has rules for, which is
why this file is short: `body()` and `left_hand()`/`right_hand()` slice out
arrays the existing rules already accept, so nothing is reimplemented. The
gestures defined here are only the ones the extra points make possible at all.

Via rtmlib, which is ONNX and runs on CPU, so it does not contend with the
detector for the GPU. Weights download on first load.

**This does not yet stand in for `pose` or `hands`.** A behaviour needs the
role its gesture is registered under, so loading this one does not satisfy a
finger guard. Letting it substitute means `needs_roles` becoming "any of
these" rather than "all of these", which is a change to `world.revalidate`
and not one to make on the way past.
"""

from __future__ import annotations

import logging

import numpy as np

from config import CFG, resolve_device

log = logging.getLogger("perception.wholebody")

#: Where each block starts in the 133-point array.
BODY, FEET, FACE, LEFT_HAND, RIGHT_HAND = 0, 17, 23, 91, 112
N_KEYPOINTS = 133

#: Offsets inside the 68-point face block. The inner lip is what opens; the
#: outer one moves when you smile too.
FACE_INNER_LIP_TOP = FACE + 62
FACE_INNER_LIP_BOTTOM = FACE + 66
#: Outer eye corners, for scale. Interocular distance is the standard way to
#: normalise a face measurement: it barely changes with expression, so it
#: holds whether someone is close to the camera or across the room.
FACE_RIGHT_EYE_OUTER = FACE + 36
FACE_LEFT_EYE_OUTER = FACE + 45

#: Below this a keypoint is a guess, not an observation. RTMPose scores each
#: point, unlike MediaPipe, so this is a real filter rather than a formality.
MIN_KP_CONF = 0.3

#: Inner-lip gap as a fraction of interocular distance. Measured against a
#: closed mouth sitting near zero and a deliberate open one well above it;
#: worth tuning on the demo machine, which is why it is a module constant.
MOUTH_OPEN_RATIO = 0.35


class RTMPoseWholeBody:
    """RTMPose whole-body via rtmlib. No weights file: it fetches its own."""

    role = "wholebody"

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
        from rtmlib import Wholebody

        device = resolve_device()
        # rtmlib speaks cuda or cpu; mps is neither, and onnxruntime has no
        # mps provider worth using, so anything that is not cuda runs on cpu.
        device = "cuda" if device == "cuda" else "cpu"
        log.info("loading rtmpose wholebody (%s, %s)", CFG.WHOLEBODY_MODE, device)
        self._model = Wholebody(
            mode=CFG.WHOLEBODY_MODE, backend="onnxruntime", device=device)
        self._loaded = True

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._model is None:
            return
        self.keypoints(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    def keypoints(self, frame: np.ndarray) -> np.ndarray:
        """(people, 133, 3) of x, y, confidence in frame pixels.

        Same shape and units as the other keypoint models, so the subject
        test, the boxes and the event crop are shared rather than rewritten.
        Empty when nobody is found, so callers never branch on None.
        """
        points, scores = self._model(frame)
        if points is None or len(points) == 0:
            return np.zeros((0, N_KEYPOINTS, 3), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        return np.concatenate([points, scores[..., None]], axis=-1)


# ---------------------------------------------------------------------------
# Slices: the blocks other skills already have rules for
# ---------------------------------------------------------------------------


def body(person: np.ndarray) -> np.ndarray:
    """The COCO-17 block, in the order `skills/pose.py` expects."""
    return person[BODY:FEET]


def left_hand(person: np.ndarray) -> np.ndarray:
    """The left hand's 21 points, in the order `skills/hands.py` expects."""
    return person[LEFT_HAND:RIGHT_HAND]


def right_hand(person: np.ndarray) -> np.ndarray:
    return person[RIGHT_HAND:N_KEYPOINTS]


# ---------------------------------------------------------------------------
# Gestures — only what the extra points make possible
# ---------------------------------------------------------------------------


def mouth_open(person: np.ndarray) -> bool:
    """Inner lip gap, as a fraction of the distance between the eyes.

    Normalised because the raw gap in pixels means nothing on its own: a
    closed mouth up close measures wider than an open one across the room.
    """
    top = person[FACE_INNER_LIP_TOP]
    bottom = person[FACE_INNER_LIP_BOTTOM]
    left_eye = person[FACE_LEFT_EYE_OUTER]
    right_eye = person[FACE_RIGHT_EYE_OUTER]
    if min(top[2], bottom[2], left_eye[2], right_eye[2]) < MIN_KP_CONF:
        return False
    interocular = float(np.linalg.norm(left_eye[:2] - right_eye[:2]))
    if interocular <= 0:
        return False
    gap = float(np.linalg.norm(top[:2] - bottom[:2]))
    return gap / interocular > MOUTH_OPEN_RATIO


def both_hands_raised(person: np.ndarray) -> bool:
    """Both wrists above their own shoulders.

    Distinct from `hand_raised`, which fires on either one. Surrender, a
    touchdown, "I give up" — and much harder to trigger by accident, which
    makes it the better one to put in front of a judge.
    """
    from skills.pose import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST

    kp = body(person)
    for wrist, shoulder in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        w, s = kp[wrist], kp[shoulder]
        if w[2] < MIN_KP_CONF or s[2] < MIN_KP_CONF:
            return False
        if w[1] >= s[1]:          # image y grows downward: above is smaller
            return False
    return True


GESTURES = {
    "mouth_open": mouth_open,
    "both_hands_raised": both_hands_raised,
}
