"""The annotated MJPEG feed.

Annotation and JPEG encoding happen once per frame and are shared by every
viewer, so opening a second browser tab costs nothing. The feed is also
throttled well below the inference rate: the operator cannot see 60fps and the
inference loop should not be paying for pixels nobody looks at.

This never blocks the inference loop. It reads the loop's latest published
view; if it is busy or slow, the worst case is a stale frame on screen.
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

# Phase colours, chosen so "is the car allowed to move" is readable across a
# room: green only in TRACKING, red for anything broken.
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


class Streamer:
    def __init__(self, loop: InferenceLoop) -> None:
        self.loop = loop
        self.box = sv.BoxAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self._jpeg: bytes | None = None
        self._seq = -1
        self._at = 0.0
        self.encoded = 0

    # 1. Encoding ------------------------------------------------------
    def jpeg(self) -> bytes:
        """Latest annotated frame, re-encoded at most STREAM_FPS times a second."""
        now = time.time()
        view = self.loop.view
        stale = view.seq != self._seq
        due = now - self._at >= 1.0 / max(CFG.STREAM_FPS, 1.0)
        if self._jpeg is not None and not (stale and due):
            return self._jpeg

        frame = self._render(view)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, CFG.JPEG_QUALITY])
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

        dets = view.detections
        if dets is not None and len(dets):
            frame = self.box.annotate(frame, dets)
            frame = self.label.annotate(frame, dets, labels=self._labels(dets))
            if view.chosen is not None and view.chosen < len(dets):
                x1, y1, x2, y2 = dets.xyxy[view.chosen].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 4)

        return self._overlay(frame, view.state)

    @staticmethod
    def _labels(dets: sv.Detections) -> list[str]:
        names = dets.data.get("class_name")
        ids = dets.tracker_id
        out = []
        for i in range(len(dets)):
            name = str(names[i]) if names is not None else "?"
            out.append(f"{name}#{ids[i]}" if ids is not None else name)
        return out

    # 2. Overlay -------------------------------------------------------
    def _overlay(self, frame: np.ndarray, state) -> np.ndarray:
        """Phase, spec and target readout, so a glance at the feed explains
        what the car is about to do."""
        h, w = frame.shape[:2]
        phase = state.phase if state else Phase.BOOTING
        colour = PHASE_BGR.get(phase, DIM)

        cv2.rectangle(frame, (0, 0), (w, 34), (24, 24, 24), -1)
        cv2.putText(frame, phase.value, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        task = self.loop.active
        if task:
            cv2.putText(frame, task.summary(), (140, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (210, 210, 210), 1)
        if state and state.visible:
            # Own bar, because the readout sits over whatever the camera sees.
            cv2.rectangle(frame, (0, h - 28), (w, h), (24, 24, 24), -1)
            txt = (f"{state.label}#{state.track_id}  cx={state.cx:+.2f} "
                   f"cy={state.cy:+.2f}  area={state.area:.3f}  conf={state.conf:.2f}")
            cv2.putText(frame, txt, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (210, 210, 210), 1)

        cv2.putText(frame, f"{self.loop.timings.fps:.0f} fps", (w - 80, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        # Crosshair: the controller steers to put the target box on this point.
        cv2.drawMarker(frame, (w // 2, h // 2), DIM, cv2.MARKER_CROSS, 18, 1)
        return frame

    # 3. Serving -------------------------------------------------------
    async def mjpeg(self):
        import asyncio

        interval = 1.0 / max(CFG.STREAM_FPS, 1.0)
        while True:
            data = self.jpeg()
            if data:
                yield (BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            await asyncio.sleep(interval)
