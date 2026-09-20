"""keyboard: find a keyboard and show where to type something.

    SEARCHING --a fit that holds--> LOCKED --no fit for 2s--> SEARCHING

"Show me how to type HACK THE NORTH" is this behaviour. OCR reads whatever
characters it can off the frame, `skills/keyboard.py` fits the QWERTY template
to them, and every key in the requested text gets a numbered badge in typing
order — including the keys OCR never read, because the fit supplies them.

Repeated letters carry every position they occupy, so the 'h' in HACK THE
NORTH is badged `1·7·14` rather than three overlapping circles.

Three ways to present it, because a wall of badges is a photograph and a demo
wants a rhythm: `all` shows the whole word at once, `auto` walks it on a
timer, and `manual` steps on POST /behaviors/{id}/advance.
"""

from __future__ import annotations

import numpy as np

from behaviors.base import Behavior, Frame, Outcome
from render.layers import Alert, Badges
from skills import keyboard as layout

#: No successful fit for this long and the keyboard is gone, not just missed.
#: Comfortably longer than the OCR interval, or a single slow read would
#: unlock a keyboard that never moved.
LOST_AFTER_S = 2.0
#: Seconds per key in `auto`.
STEP_S = 0.8
FLASH_S = 1.5

STEP_MODES = ("all", "auto", "manual")


class Keyboard(Behavior):
    kind = "keyboard"
    states = ("SEARCHING", "LOCKED", "PAUSED")
    initial = "SEARCHING"
    emits = ("keyboard_locked", "keyboard_lost", "step")
    #: OCR is what turns a picture of a keyboard into key positions. Without
    #: it the behaviour is paused with a reason rather than left searching
    #: forever against nothing.
    needs_roles = ("ocr",)

    def __init__(self, *a, **kw) -> None:
        self._matrix: np.ndarray | None = None
        self._fit_at: float = 0.0
        self._stepped_at: float = 0.0
        self._locked_at: float = 0.0
        #: Which key of `typeable` is current. Not `self.step`: that is the base
        #: class's per-frame entry point, and shadowing it with an int makes
        #: every frame raise "'int' object is not callable".
        self.cursor: int = 0
        #: Bumped by the API thread, consumed here. One integer write under
        #: the GIL, the same shape as `loop.hud_text`, so the loop stays the
        #: only thing that acts on it.
        self.pending_advance: int = 0
        super().__init__(*a, **kw)

    def validate(self) -> None:
        text = str(self.param("text", "") or "")
        if not text.strip():
            raise ValueError("keyboard needs 'text': what to show how to type")
        self.text = text.lower()
        self.typeable = [c for c in self.text if c in layout.TEMPLATE]
        if not self.typeable:
            raise ValueError(
                f"none of {text!r} is on the keyboard template; "
                f"letters, digits and space only")
        self.step_mode = str(self.param("step_mode", "all"))
        if self.step_mode not in STEP_MODES:
            raise ValueError(
                f"unknown step_mode {self.step_mode!r}; known: {list(STEP_MODES)}")
        self.step_s = float(self.param("step_s", STEP_S))

    # 1. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        seen = frame.skills.get("ocr")
        if seen is None:
            # The reader has not produced anything yet this frame.
            return

        self._update_fit(seen, frame.now)
        self.data.update(chars=len(seen), keys=len(self.typeable),
                         step=self.cursor, mode=self.step_mode,
                         locked=self.state == "LOCKED")

        if self.state == "LOCKED" and frame.now - self._fit_at > LOST_AFTER_S:
            self.to("SEARCHING", "lost the keyboard")
            outcome.events.append(self.event(
                "keyboard_lost", frame, None,
                detail=f"{self.label}: lost the keyboard"))

        if self.state != "LOCKED":
            return

        self._advance(frame, outcome)
        self._draw(frame, outcome)
        if frame.now - self._locked_at < FLASH_S:
            outcome.layers.append(Alert(
                text=f"{self.label}: {self.text.strip()}",
                color=self.color, intensity=0.5))

    def _update_fit(self, seen, now: float) -> None:
        """Fold a fresh read into the fit. A failed fit keeps the old one."""
        matrix = layout.fit(seen)
        if matrix is None:
            return
        self._matrix = layout.smooth(self._matrix, matrix)
        self._fit_at = now
        if self.state != "LOCKED":
            self._locked_at = now
            self.cursor = 0
            self._stepped_at = now
            self.to("LOCKED", f"{len(seen)} characters read")
            self._pending_lock = True

    def _advance(self, frame: Frame, outcome: Outcome) -> None:
        """Move through the word, however this behaviour was asked to."""
        if getattr(self, "_pending_lock", False):
            self._pending_lock = False
            outcome.events.append(self.event(
                "keyboard_locked", frame, None,
                detail=f"{self.label}: keyboard locked, "
                       f"{len(self.typeable)} keys to type",
                keys=len(self.typeable)))

        if self.step_mode == "all":
            return

        wanted = self.cursor
        if self.step_mode == "manual":
            if self.pending_advance:
                self.pending_advance -= 1
                wanted = self.cursor + 1
        elif frame.now - self._stepped_at >= self.step_s:
            wanted = self.cursor + 1

        if wanted == self.cursor:
            return
        # Wrapping rather than stopping: the demo runs until it is stopped,
        # and a badge frozen on the last key looks like a crash.
        self.cursor = wanted % len(self.typeable)
        self._stepped_at = frame.now
        outcome.events.append(self.event(
            "step", frame, None,
            detail=f"{self.label}: key {self.cursor + 1} of {len(self.typeable)} "
                   f"({self.typeable[self.cursor]!r})",
            step=self.cursor, key=self.typeable[self.cursor], notify=False))

    def advance(self) -> None:
        """Called from the API thread for `step_mode: manual`."""
        self.pending_advance += 1

    # 2. Drawing --------------------------------------------------------
    def _draw(self, frame: Frame, outcome: Outcome) -> None:
        if self._matrix is None:
            return
        points = layout.project(self._matrix, "".join(self.typeable))

        # One badge per key, carrying every position it occupies. Three
        # separate badges on the same 'h' would overlap into an unreadable
        # smudge, which is exactly what the demo needs to be legible.
        order: dict[tuple[int, int], list[int]] = {}
        for i, point in enumerate(points):
            if point is not None:
                order.setdefault(point, []).append(i)

        h, w = frame.shape[:2]
        spots, texts, current = [], [], None
        for point, steps in order.items():
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                continue          # projected off-frame: a bad fit, not a key
            if self.step_mode != "all" and self.cursor not in steps:
                shown = [s for s in steps if s <= self.cursor]
                if not shown:
                    continue
                steps = shown
            if self.cursor in steps:
                current = len(spots)
            spots.append(point)
            texts.append("·".join(str(s + 1) for s in steps))

        if spots:
            outcome.layers.append(Badges(
                points=spots, texts=texts, color=self.color,
                path=True, current=current))
