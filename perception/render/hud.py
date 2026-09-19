"""The HUD: what the judges read without being told what to look for.

A top bar with the model and frame rate, the instruction the operator typed,
and one chip per behaviour showing its kind and state. Colour matches the
behaviour's own layers, so a chip and its boxes are obviously the same thing.
"""

from __future__ import annotations

import cv2
import numpy as np

from render.layers import DIM, PANEL, TEXT, BGR, Layer

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
        self._top_bar(frame, w)
        self._chips(frame, h)
        # The frame centre is what a track behaviour steers toward.
        cv2.drawMarker(frame, (w // 2, h // 2), DIM, cv2.MARKER_CROSS, 18, 1)

    def _top_bar(self, frame: np.ndarray, w: int) -> None:
        cv2.rectangle(frame, (0, 0), (w, 34), PANEL, -1)
        x = 10
        if not self.camera_ok:
            x = _text(frame, "NO CAMERA", x, 23, BAD, 0.6, 2)
        elif self.model:
            x = _text(frame, self.model, x, 23, TEXT, 0.6, 2)
        if self.text:
            # The instruction, trimmed to whatever space is left.
            room = max(w - x - 90, 40)
            _text(frame, _fit(self.text, room), x + 14, 23, DIM, 0.5, 1)
        _text(frame, f"{self.fps:.0f} fps", w - 78, 23, TEXT, 0.5, 1)

    def _chips(self, frame: np.ndarray, h: int) -> None:
        if not self.chips:
            return
        rows = self.chips[:6]
        top = h - 22 * len(rows) - 8
        cv2.rectangle(frame, (0, top), (frame.shape[1], h), PANEL, -1)
        for i, (label, state, colour) in enumerate(rows):
            y = top + 22 * i + 16
            live = state in LIVE
            dot = colour if live else (WARN if state in TRYING else BAD)
            cv2.circle(frame, (16, y - 5), 5, dot, -1)
            x = _text(frame, label, 28, y, TEXT if live else DIM, 0.48, 1)
            _text(frame, state, x + 10, y, dot, 0.44, 1)


def _text(frame, s, x, y, colour, scale, thick) -> int:
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    return x + tw


def _fit(s: str, px: int, scale: float = 0.5) -> str:
    """Trim to fit, with an ellipsis. Roughly 9px per character at 0.5."""
    limit = max(int(px / (9 * scale / 0.5)) // 1, 8)
    return s if len(s) <= limit else s[: limit - 1] + "…"
