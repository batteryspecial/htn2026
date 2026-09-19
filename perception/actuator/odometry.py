"""Measuring how far the view has actually turned.

"Turn 45 degrees left and count the people" needs an answer to *have I turned
45 degrees yet*, and there is no encoder on a human holding a camera. So the
turn is measured from the picture: when the camera pans, the whole scene slides
across the frame, and how far it slid is how far we turned.

The measurement is `cv2.phaseCorrelate`, which finds the global translation
between two frames in the frequency domain. Both frames are Fourier
transformed, their cross-power spectrum is taken, and the inverse transform of
that is a surface with one sharp peak at the shift. Two properties make it the
right tool here rather than tracking features:

- It uses only phase and discards magnitude, so it is nearly immune to the
  exposure and white-balance changes a camera makes while it is being swung.
- It reports the *dominant* motion of the whole image, so people walking
  through the shot perturb it rather than capturing it, which is exactly
  backwards from feature tracking.

It is also sub-pixel accurate and costs about a millisecond at 320px, measured.

Turning pixels into degrees is one similar triangle: a shift of `dx` pixels
across a frame `w` wide is `dx / w` of the horizontal field of view.

Known limits, all of which the guards below bound rather than solve:

- It measures *translation*, so a camera that is carried sideways reads the
  same as one that is turned. For a person pivoting on the spot that is fine.
- Error accumulates, because each frame's estimate is added to the last. Over
  a 45 degree turn taking a couple of seconds it stays within a few degrees;
  it is not an IMU and should not be trusted over minutes.
- The field of view is a guess unless the camera is calibrated, and it scales
  the answer linearly. `hfov_deg` is the knob to turn when the measured angle
  is consistently short or long.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("perception.odometry")

#: Frames are downscaled to this width before correlating. Cheap, and the
#: extra resolution buys nothing: the shift is sub-pixel accurate anyway.
WORK_W = 320
#: Correlation confidence below this means the frames did not really match --
#: a cut, a hand over the lens, violent motion blur. Ignored rather than
#: integrated, because a wrong reading is worse than a missing one.
MIN_RESPONSE = 0.10
#: Shifts below this are noise, and integrating noise is how a stationary
#: camera slowly claims to have turned ninety degrees.
DEADBAND_PX = 0.35
#: A shift larger than this fraction of the frame in one frame is not a pan,
#: it is a dropped frame or a scene change.
MAX_SHIFT_FRAC = 0.25
#: Below this standard deviation there is no texture to correlate. A blank
#: wall has no measurable motion, and saying so is better than guessing.
MIN_TEXTURE_STD = 1.0


class PanOdometry:
    """Accumulates how far the view has turned, in degrees. Right is positive."""

    def __init__(self, hfov_deg: float = 70.0) -> None:
        self.hfov_deg = float(hfov_deg)
        self.yaw_deg = 0.0
        self.samples = 0
        self.rejected = 0
        self._prev: np.ndarray | None = None
        self._window: np.ndarray | None = None
        self._flat = False

    def reset(self) -> None:
        """Start measuring from here. Called when a turn begins."""
        self.yaw_deg = 0.0
        self.samples = 0
        self.rejected = 0
        self._prev = None

    def update(self, frame: np.ndarray) -> float:
        """Fold one frame in. Returns the cumulative yaw in degrees."""
        current = self._prepare(frame)
        previous, self._prev = self._prev, current
        if previous is None or previous.shape != current.shape:
            return self.yaw_deg

        # No window argument: both frames are windowed already by _prepare.
        # Passing one here makes OpenCV multiply the inputs *in place*, which
        # would re-window the stored previous frame on every call, darken its
        # edges a little more each time, and manufacture a steady drift out of
        # a stationary camera. Measured at 3.4 degrees over 30 still frames.
        (dx, _dy), response = cv2.phaseCorrelate(previous, current)
        width = current.shape[1]

        if self._flat or response < MIN_RESPONSE or abs(dx) > width * MAX_SHIFT_FRAC:
            # Keep the frame as the new reference but do not trust the number.
            self.rejected += 1
            return self.yaw_deg
        if abs(dx) < DEADBAND_PX:
            return self.yaw_deg

        # The scene slides the opposite way to the camera: pan right and the
        # content moves left. Hence the sign flip.
        self.yaw_deg += -dx / width * self.hfov_deg
        self.samples += 1
        return self.yaw_deg

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Grey, small, float, windowed -- what phaseCorrelate wants."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = grey.shape[:2]
        if w != WORK_W:
            grey = cv2.resize(grey, (WORK_W, max(1, int(h * WORK_W / w))))
        out = grey.astype(np.float32)
        # Nothing to correlate against: a blank wall, a covered lens.
        self._flat = bool(out.std() < MIN_TEXTURE_STD)
        if self._window is None or self._window.shape != out.shape:
            # Without a window the frame edges act like a hard rectangular
            # aperture and put a cross-shaped artefact through the correlation
            # peak. Hanning tapers them away.
            self._window = cv2.createHanningWindow(
                (out.shape[1], out.shape[0]), cv2.CV_32F)
        # Windowed here, once, into a fresh array, so the frame we keep as the
        # reference is never mutated by OpenCV.
        return out * self._window
