"""Where the keys are, given a handful of characters OCR could read.

OCR finds *some* keys — never all of them, and never the same ones twice. The
job here is to turn that partial, noisy scatter into every key's position,
including the ones OCR missed entirely.

That works because a keyboard is rigid and its layout is known. Fitting the
QWERTY template to the characters that *were* read gives one transform, and
the transform gives every other key for free. Reading six keys locates all
forty-seven.

A flat keyboard seen at an angle is a plane under perspective, so the
transform is a homography and `cv2.findHomography` with RANSAC is the fit.
RANSAC matters more than the model does: OCR reads a stray letter off a mug or
a sticker, and one bad correspondence drags a least-squares fit across the
table. Below the four points a homography needs, a partial affine still beats
giving up.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("perception.keyboard")

#: The QWERTY template, in key units: one unit is one key pitch. The origin is
#: the 'q' key, y grows downward, and the row offsets are the real stagger.
_ROWS = (
    ("1234567890", -1.0, -0.5),
    ("qwertyuiop", 0.0, 0.0),
    ("asdfghjkl", 1.0, 0.25),
    ("zxcvbnm", 2.0, 0.75),
)

TEMPLATE: dict[str, tuple[float, float]] = {
    char: (x0 + i, y)
    for row, y, x0 in _ROWS
    for i, char in enumerate(row)
}
#: The space bar is not a character OCR will ever read, but it is a key the
#: skill has to point at, so it lives in the template like any other.
TEMPLATE[" "] = (4.5, 3.0)

#: RANSAC tolerance, in key units. Half a key: a correspondence further out
#: than that is a different key, not a noisy read of this one.
REPROJECT_KEY_UNITS = 0.5
#: Below this many agreeing points the fit is not trustworthy.
MIN_INLIERS = 6
#: A homography needs four; fewer than this and not even affine is worth it.
MIN_AFFINE = 3
#: How much of the old fit to keep each update. The template is rigid and the
#: keyboard barely moves, so heavy smoothing costs nothing and takes the
#: jitter out of the badges between reads.
EMA_ALPHA = 0.35


def fit(seen: list[tuple[str, float, float]]) -> np.ndarray | None:
    """A 3x3 transform from template units to image pixels, or None.

    `seen` is what OCR read: (character, x, y) in pixels. Duplicate readings
    of the same character are kept — RANSAC is better placed to decide which
    one is the real key than any rule here would be.
    """
    import cv2

    pairs = [(TEMPLATE[c], (x, y)) for c, x, y in seen if c in TEMPLATE]
    if len(pairs) < MIN_AFFINE:
        return None

    src = np.array([p[0] for p in pairs], dtype=np.float32)
    dst = np.array([p[1] for p in pairs], dtype=np.float32)

    if len(pairs) >= 4:
        matrix, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, REPROJECT_KEY_UNITS * _scale(src, dst))
        if matrix is not None and mask is not None:
            inliers = int(mask.sum())
            if inliers >= MIN_INLIERS:
                return matrix.astype(np.float32)
            log.debug("homography rejected: %d inliers of %d", inliers, len(pairs))

    # Three to five points, or a homography nothing agreed on. A partial
    # affine cannot represent the perspective, so it is wrong for a keyboard
    # seen from an angle — but it is right for one seen from above, and a
    # slightly wrong badge is worth more than no badge.
    affine, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if affine is None:
        return None
    return np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)


def _scale(src: np.ndarray, dst: np.ndarray) -> float:
    """Pixels per key unit, roughly, so the RANSAC tolerance is in pixels.

    A fixed pixel tolerance would be far too tight on a keyboard filling the
    frame and far too loose on one across the room.
    """
    spread_src = float(np.ptp(src, axis=0).max())
    spread_dst = float(np.ptp(dst, axis=0).max())
    if spread_src <= 0:
        return 1.0
    return max(spread_dst / spread_src, 1.0)


def project(matrix: np.ndarray, chars: str) -> list[tuple[int, int] | None]:
    """Image position of each character, None for ones not on the template."""
    import cv2

    wanted = [(i, c) for i, c in enumerate(chars) if c in TEMPLATE]
    out: list[tuple[int, int] | None] = [None] * len(chars)
    if not wanted:
        return out
    src = np.array([[TEMPLATE[c] for _, c in wanted]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0]
    for (i, _), (x, y) in zip(wanted, dst):
        out[i] = (int(round(float(x))), int(round(float(y))))
    return out


def smooth(previous: np.ndarray | None, current: np.ndarray,
           alpha: float = EMA_ALPHA) -> np.ndarray:
    """Blend a new fit into the old one.

    On the matrix rather than on the projected points, so a key OCR missed
    this time does not jump when it comes back. Normalised by the bottom-right
    term first: two homographies are only comparable at the same scale, and
    averaging them at different ones bends the result.
    """
    if previous is None:
        return current
    a, b = previous, current
    if abs(b[2, 2]) > 1e-9:
        b = b / b[2, 2]
    if abs(a[2, 2]) > 1e-9:
        a = a / a[2, 2]
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)
