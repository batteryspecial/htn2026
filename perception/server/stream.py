"""The annotated MJPEG feed.

Annotation and JPEG encoding happen once per frame and are shared by every
viewer, so a second browser tab costs nothing. The feed is throttled well below
the inference rate: nobody can see 60fps and the loop should not pay for pixels
nobody looks at.

Never blocks the loop. It reads the loop's last published view; if the loop is
busy the worst case is a stale frame on screen.

Each behaviour gets its own colour, so a frame with a counter and a follow-cam
running at once is readable rather than a pile of boxes.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import supervision as sv

from config import CFG
from contracts import Phase
from runtime.loop import InferenceLoop

log = logging.getLogger("perception.stream")

BOUNDARY = b"--frame"

# Phase colours, readable across a room.
PHASE_BGR = {
    Phase.TRACKING: (80, 220, 80),
    Phase.ACQUIRING: (60, 200, 250),
    Phase.LOST: (60, 200, 250),
    Phase.NO_TARGET: (60, 200, 250),
    Phase.SWITCHING: (250, 190, 60),
    Phase.LOADING_MODEL: (250, 190, 60),
    Phase.FAULT: (70, 70, 240),
}
DIM = (160, 160, 160)
# One colour per behaviour, assigned in order.
PALETTE = [(90, 220, 90), (250, 190, 60), (200, 120, 250), (80, 210, 250), (250, 130, 130)]


class Streamer:
    def __init__(self, loop: InferenceLoop) -> None:
        self.loop = loop
        self.label = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
        self._jpeg: bytes | None = None
        self._seq = -1
        self._at = 0.0
        self.encoded = 0

    # 1. Encoding ------------------------------------------------------
    def jpeg(self) -> bytes:
        """Latest annotated frame, re-encoded at most STREAM_FPS times a second."""
        now = time.time()
        view = self.loop.view
        stale, due = view.seq != self._seq, now - self._at >= 1.0 / max(CFG.STREAM_FPS, 1.0)
        if self._jpeg is not None and not (stale and due):
            return self._jpeg

        ok, buf = cv2.imencode(".jpg", self._render(view),
                               [cv2.IMWRITE_JPEG_QUALITY, CFG.JPEG_QUALITY])
        if ok:
            self._jpeg, self._seq, self._at = buf.tobytes(), view.seq, now
            self.encoded += 1
        return self._jpeg or b""

    def _render(self, view) -> np.ndarray:
        frame = view.frame
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "NO SIGNAL", (200, 250), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (70, 70, 240), 2)
        else:
            frame = frame.copy()

        tracks = view.tracks
        if tracks is not None and len(tracks):
            self._draw_unclaimed(frame, tracks, view.outcomes)
            self._draw_behaviors(frame, tracks, view.outcomes)
        return self._overlay(frame, view)

    # 2. Boxes ----------------------------------------------------------
    def _draw_unclaimed(self, frame, tracks, outcomes) -> None:
        """Tracks no behaviour wants, drawn faintly.

        Keeping them visible is what makes an exclusion legible: you can see
        the person in the black jacket being deliberately ignored.
        """
        claimed = {int(i) for o in outcomes.values() for i in o.matches}
        for i in range(len(tracks)):
            if i in claimed:
                continue
            x1, y1, x2, y2 = tracks.xyxy[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)

    def _draw_behaviors(self, frame, tracks, outcomes) -> None:
        for n, (bid, outcome) in enumerate(outcomes.items()):
            colour = PALETTE[n % len(PALETTE)]
            for i in outcome.matches:
                x1, y1, x2, y2 = tracks.xyxy[int(i)].astype(int)
                thick = 4 if outcome.chosen == int(i) else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
            if len(outcome.matches):
                labels, subset = self._labels(tracks, outcome.matches), tracks[outcome.matches]
                self.label.annotate(frame, subset, labels=labels)

    @staticmethod
    def _labels(tracks, idx) -> list[str]:
        names, ids = tracks.data.get("class_name"), tracks.tracker_id
        out = []
        for i in idx:
            i = int(i)
            name = str(names[i]) if names is not None else "?"
            out.append(f"{name}#{ids[i]}" if ids is not None else name)
        return out

    # 3. Overlay --------------------------------------------------------
    def _overlay(self, frame: np.ndarray, view) -> np.ndarray:
        """Phase, behaviours and counts, so a glance explains the whole state."""
        h, w = frame.shape[:2]
        summary = view.summary
        phase = summary.phase if summary else Phase.BOOTING

        cv2.rectangle(frame, (0, 0), (w, 34), (24, 24, 24), -1)
        cv2.putText(frame, phase.value, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    PHASE_BGR.get(phase, DIM), 2)
        if summary and summary.model:
            cv2.putText(frame, summary.model, (150, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (210, 210, 210), 1)
        cv2.putText(frame, f"{self.loop.timings.fps:.0f} fps", (w - 80, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

        if summary and summary.behaviors:
            rows = summary.behaviors[:5]
            cv2.rectangle(frame, (0, h - 18 * len(rows) - 10), (w, h), (24, 24, 24), -1)
            for n, b in enumerate(rows):
                y = h - 18 * (len(rows) - n) + 4
                colour = PALETTE[n % len(PALETTE)] if b.state == "active" else DIM
                text = (f"{b.label or b.kind}  {b.state}  x{b.matches}"
                        + (f"  {b.detail}" if b.detail else ""))
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        cv2.drawMarker(frame, (w // 2, h // 2), DIM, cv2.MARKER_CROSS, 18, 1)
        return frame

    # 4. Serving --------------------------------------------------------
    async def mjpeg(self):
        import asyncio

        interval = 1.0 / max(CFG.STREAM_FPS, 1.0)
        while True:
            data = self.jpeg()
            if data:
                yield (BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            await asyncio.sleep(interval)
