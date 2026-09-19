"""The HUD: what the judges read without being told what to look for.

A top bar with the model and frame rate, the instruction the operator typed,
and one chip per behaviour showing its kind and state. Colour matches the
behaviour's own layers, so a chip and its boxes are obviously the same thing.
"""

from __future__ import annotations

import cv2
import numpy as np

from render.layers import DIM, PANEL, TEXT, BGR, Layer, ui_scale

# States that mean the behaviour is doing its job right now.
LIVE = {"ACTIVE", "TRACKING", "ARMED", "LOCKED", "REACHED"}
# States that mean it is trying.
TRYING = {"ACQUIRING", "EDGE", "ARMING", "GUIDING", "SEARCHING"}
WARN: BGR = (60, 190, 250)
BAD: BGR = (70, 70, 240)


class Hud(Layer):
    """Composed last, over everything."""

    order = 90

    def __init__(self, model: str | None, fps: float, text: str | None,
                 chips: list[tuple[str, str, BGR]], camera_ok: bool = True) -> None:
        self.model = model
        self.fps = fps
        self.text = text
        self.chips = chips  # (label, state, colour)
        self.camera_ok = camera_ok

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        # Everything scales with the frame: a 1080p phone clip and a webcam
        # must both be readable from the back of a room.
        k = ui_scale(frame)
        self._top_bar(frame, w, k)
        self._chips(frame, h, k)
        # The frame centre is what a track behaviour steers toward.
        cv2.drawMarker(frame, (w // 2, h // 2), DIM, cv2.MARKER_CROSS,
                       int(18 * k), max(1, int(k)))

    def _top_bar(self, frame: np.ndarray, w: int, k: float) -> None:
        bar, base = int(34 * k), int(23 * k)
        cv2.rectangle(frame, (0, 0), (w, bar), PANEL, -1)
        x = int(10 * k)
        if not self.camera_ok:
            x = _text(frame, "NO CAMERA", x, base, BAD, 0.6 * k, max(2, int(2 * k)))
        elif self.model:
            x = _text(frame, self.model, x, base, TEXT, 0.6 * k, max(2, int(2 * k)))
        fps_text = f"{self.fps:.0f} fps"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5 * k, max(1, int(k)))
        if self.text:
            # The instruction, trimmed to whatever space is left.
            room = max(w - x - fw - int(40 * k), int(40 * k))
            _text(frame, _fit(self.text, room, 0.5 * k), x + int(14 * k), base,
                  DIM, 0.5 * k, max(1, int(k)))
        _text(frame, fps_text, w - fw - int(14 * k), base, TEXT, 0.5 * k, max(1, int(k)))

    def _chips(self, frame: np.ndarray, h: int, k: float) -> None:
        if not self.chips:
            return
        rows = self.chips[:6]
        row_h = int(22 * k)
        top = h - row_h * len(rows) - int(8 * k)
        cv2.rectangle(frame, (0, top), (frame.shape[1], h), PANEL, -1)
        for i, (label, state, colour) in enumerate(rows):
            y = top + row_h * i + int(16 * k)
            live = state in LIVE
            dot = colour if live else (WARN if state in TRYING else BAD)
            cv2.circle(frame, (int(16 * k), y - int(5 * k)), max(3, int(5 * k)), dot, -1)
            x = _text(frame, label, int(28 * k), y, TEXT if live else DIM,
                      0.48 * k, max(1, int(k)))
            _text(frame, state, x + int(10 * k), y, dot, 0.44 * k, max(1, int(k)))


def _text(frame, s, x, y, colour, scale, thick) -> int:
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    return x + tw


def _fit(s: str, px: int, scale: float = 0.5) -> str:
    """Trim to fit, with an ellipsis. Roughly 9px per character at scale 0.5."""
    limit = max(int(px / max(9 * scale / 0.5, 1e-6)), 8)
    return s if len(s) <= limit else s[: limit - 1] + "..."
