"""Render layers: the only vocabulary a behaviour has for drawing.

Behaviours declare what they want shown; this module is the only place that
knows how to show it. That split is deliberate. A behaviour that could draw
directly would eventually draw something expensive, or something that throws,
inside the frame loop. And nothing generates drawing code at runtime: the
agent picks colours and toggles layers, never geometry.

Composition order is fixed by layer type, not by which behaviour spoke first,
so two behaviours can never fight over who paints last:

    privacy blur -> masks -> boxes/labels/ids -> trails -> zones/lines
                 -> badges -> arrows -> alert flash -> HUD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import cv2
import numpy as np

BGR = tuple[int, int, int]

# Fallback palette, assigned by behaviour order when a spec names no colour.
PALETTE: list[BGR] = [
    (90, 220, 90),    # green
    (60, 190, 250),   # amber
    (250, 160, 90),   # blue
    (200, 120, 250),  # violet
    (250, 130, 130),  # red
    (230, 230, 120),  # cyan
]
DIM: BGR = (110, 110, 110)
TEXT: BGR = (235, 235, 235)
PANEL: BGR = (24, 24, 24)


#: Overlays are authored against this frame height and scaled from it, so the
#: feed reads the same on a webcam, a 1080p phone clip and a projector.
REFERENCE_H = 720


def ui_scale(frame: np.ndarray) -> float:
    """How much to enlarge overlays for this frame."""
    return max(0.75, min(frame.shape[0] / REFERENCE_H, 3.0))


def hue_of(color: BGR) -> int:
    """Hue on OpenCV's 0-179 circle."""
    return int(cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def distinct(a: BGR, b: BGR, min_hue_gap: int = 18) -> bool:
    """Whether two colours read as different across a room.

    Compared by hue rather than by channel values: #FFD400 and the amber in
    the palette differ by 87 summed across BGR and are four degrees apart on
    the colour wheel, which is to say indistinguishable on a projector.
    """
    ha, hb = hue_of(a), hue_of(b)
    gap = abs(ha - hb)
    return min(gap, 180 - gap) >= min_hue_gap


def hex_to_bgr(value: str | None, fallback: BGR = PALETTE[0]) -> BGR:
    """"#FFD400" -> BGR. Bad input falls back rather than raising: a typo in a
    colour must never take down the frame loop."""
    if not value:
        return fallback
    s = value.lstrip("#")
    if len(s) != 6:
        return fallback
    try:
        r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback
    return (b, g, r)


# ---------------------------------------------------------------------------
# 1. Layer types
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """Base. `order` fixes composition, `draw` does the work."""

    order: ClassVar[int] = 50

    def draw(self, frame: np.ndarray) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Blur(Layer):
    """Privacy. First, so everything else stays legible on top."""

    order: ClassVar[int] = 10
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    invert: bool = False      # blur everything *except* the boxes
    mode: str = "blur"        # blur | pixelate

    def draw(self, frame: np.ndarray) -> None:
        if self.invert:
            keep = frame.copy()
            frame[:] = self._obscure(frame)
            for x1, y1, x2, y2 in self.boxes.astype(int):
                frame[y1:y2, x1:x2] = keep[y1:y2, x1:x2]
            return
        for x1, y1, x2, y2 in self.boxes.astype(int):
            region = frame[y1:y2, x1:x2]
            if region.size:
                frame[y1:y2, x1:x2] = self._obscure(region)

    def _obscure(self, region: np.ndarray) -> np.ndarray:
        if self.mode == "pixelate":
            h, w = region.shape[:2]
            small = cv2.resize(region, (max(w // 12, 1), max(h // 12, 1)))
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        k = max(9, (min(region.shape[:2]) // 4) | 1)
        return cv2.GaussianBlur(region, (k, k), 0)


@dataclass
class Masks(Layer):
    """Segmentation tint. YOLOE-seg gives these away, and they read much
    better than boxes on a projector."""

    order: ClassVar[int] = 20
    masks: np.ndarray | None = None  # (n, h, w) bool
    color: BGR = PALETTE[0]
    alpha: float = 0.45

    def draw(self, frame: np.ndarray) -> None:
        if self.masks is None or len(self.masks) == 0:
            return
        union = np.any(self.masks, axis=0)
        if not union.any():
            return
        tint = np.zeros_like(frame)
        tint[:] = self.color
        frame[union] = cv2.addWeighted(frame, 1 - self.alpha, tint, self.alpha, 0)[union]


@dataclass
class Boxes(Layer):
    order: ClassVar[int] = 30
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    labels: list[str] = field(default_factory=list)
    color: BGR = PALETTE[0]
    thickness: int = 2
    emphasis: int | None = None  # index drawn thicker: the chosen subject

    def draw(self, frame: np.ndarray) -> None:
        k = ui_scale(frame)
        for i, box in enumerate(self.boxes.astype(int)):
            x1, y1, x2, y2 = box
            t = self.thickness + 2 if i == self.emphasis else self.thickness
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, max(1, int(t * k)))
            if i < len(self.labels) and self.labels[i]:
                _tag(frame, self.labels[i], (x1, y1), self.color)


@dataclass
class Trails(Layer):
    """Where a track has been. Cheap, and it makes motion legible in a still."""

    order: ClassVar[int] = 40
    paths: list[list[tuple[int, int]]] = field(default_factory=list)
    color: BGR = PALETTE[0]

    def draw(self, frame: np.ndarray) -> None:
        thick = max(2, int(2 * ui_scale(frame)))
        for path in self.paths:
            if len(path) < 2:
                continue
            pts = np.array(path, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, self.color, thick, cv2.LINE_AA)


@dataclass
class Line(Layer):
    """A counting line, with its tallies."""

    order: ClassVar[int] = 50
    a: tuple[int, int] = (0, 0)
    b: tuple[int, int] = (0, 0)
    color: BGR = PALETTE[1]
    label: str | None = None

    def draw(self, frame: np.ndarray) -> None:
        cv2.line(frame, self.a, self.b, self.color, 2, cv2.LINE_AA)
        if self.label:
            _tag(frame, self.label, self.a, self.color)


@dataclass
class Zone(Layer):
    """A watched region, drawn as an outline."""

    order: ClassVar[int] = 50
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    color: BGR = PALETTE[1]
    label: str | None = None

    def draw(self, frame: np.ndarray) -> None:
        if len(self.points) < 2:
            return
        pts = self.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, self.color, 2, cv2.LINE_AA)
        if self.label:
            _tag(frame, self.label, tuple(self.points[0].astype(int)), self.color)


@dataclass
class Badges(Layer):
    """Numbered markers on points, in order. The keyboard skill draws with these."""

    order: ClassVar[int] = 60
    points: list[tuple[int, int]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    color: BGR = PALETTE[1]
    path: bool = False
    current: int | None = None

    def draw(self, frame: np.ndarray) -> None:
        if self.path and len(self.points) > 1:
            cv2.polylines(frame, [np.array(self.points, np.int32).reshape(-1, 1, 2)],
                          False, self.color, 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.points):
            hot = i == self.current
            cv2.circle(frame, (x, y), 15 if hot else 12,
                       self.color if hot else PANEL, -1)
            cv2.circle(frame, (x, y), 15 if hot else 12, self.color, 2)
            text = self.texts[i] if i < len(self.texts) else str(i + 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, text, (x - tw // 2, y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        PANEL if hot else TEXT, 1, cv2.LINE_AA)


@dataclass
class Arrow(Layer):
    """Guidance. The human is the motor, so this is the motor driver's output."""

    order: ClassVar[int] = 70
    direction: str = "left"        # left | right | up | down
    strength: float = 1.0          # 0..1; faint early warning to bold hold
    label: str | None = None
    pulse: float = 1.0             # 0..1 brightness, animated by the caller
    color: BGR = (90, 220, 90)

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        size = int(min(h, w) * (0.06 + 0.06 * self.strength))
        pad = 24
        cy, cx = h // 2, w // 2
        anchor = {"left": (pad + size, cy), "right": (w - pad - size, cy),
                  "up": (cx, pad + size), "down": (cx, h - pad - size)}[self.direction]
        dx, dy = {"left": (-1, 0), "right": (1, 0),
                  "up": (0, -1), "down": (0, 1)}[self.direction]
        colour = tuple(int(c * (0.35 + 0.65 * self.pulse)) for c in self.color)
        tip = (anchor[0] + dx * size, anchor[1] + dy * size)
        tail = (anchor[0] - dx * size, anchor[1] - dy * size)
        cv2.arrowedLine(frame, tail, tip, colour,
                        max(3, int((3 + 6 * self.strength) * ui_scale(frame))),
                        cv2.LINE_AA, tipLength=0.4)
        if self.label:
            _tag(frame, self.label, (anchor[0] - size, anchor[1] + size + 6), colour)


@dataclass
class Alert(Layer):
    """A flash border plus a banner. One second on a trigger, hard to miss."""

    order: ClassVar[int] = 80
    text: str = ""
    color: BGR = (70, 70, 240)
    intensity: float = 1.0

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        k = ui_scale(frame)
        t = max(2, int(14 * self.intensity * k))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), self.color, t)
        if self.text:
            scale, thick = 0.7 * k, max(2, int(2 * k))
            (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            x, y = (w - tw) // 2, int(70 * k)
            cv2.rectangle(frame, (x - int(12 * k), y - th - int(10 * k)),
                          (x + tw + int(12 * k), y + int(10 * k)), self.color, -1)
            cv2.putText(frame, self.text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)


def _tag(frame: np.ndarray, text: str, at: tuple[int, int], color: BGR) -> None:
    """A filled label chip that stays readable over any background."""
    k = ui_scale(frame)
    scale, thick, pad = 0.45 * k, max(1, int(k)), int(8 * k)
    x, y = at
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + pad)
    cv2.rectangle(frame, (x, y - th - pad), (x + tw + pad + 4, y), color, -1)
    cv2.putText(frame, text, (x + int(5 * k), y - int(5 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, PANEL, thick, cv2.LINE_AA)


def no_signal(shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A placeholder frame. A blank feed with a reason beats a broken image
    icon on stage, and this keeps every drawing call inside render/."""
    frame = np.zeros((*shape, 3), np.uint8)
    h, w = shape
    text, k = "NO SIGNAL", max(0.75, min(h / REFERENCE_H, 3.0))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, max(2, int(2 * k)))
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (70, 70, 240),
                max(2, int(2 * k)), cv2.LINE_AA)
    return frame


def compose(frame: np.ndarray, layers: list[Layer]) -> np.ndarray:
    """Draw every layer in type order. One bad layer must not lose the frame."""
    import logging

    for layer in sorted(layers, key=lambda l: type(l).order):
        try:
            layer.draw(frame)
        except Exception:
            logging.getLogger("perception.render").exception(
                "layer %s failed", type(layer).__name__)
    return frame
