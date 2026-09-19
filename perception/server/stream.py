"""The enriched MJPEG feed.

What the judges see is exactly what the pipeline computed: the overlays are
drawn here, server-side, and the frontend is a dumb `<img>`. No drawing logic
lives in the browser and none can drift from the pipeline's view of the world.

Composition and encoding happen once per frame and are shared by every viewer,
so a second tab costs nothing. The feed is throttled well below the inference
rate, because nobody can see 60fps and the loop should not pay for pixels
nobody looks at.
"""

from __future__ import annotations

import asyncio
import logging
import time

import cv2

from config import CFG
from render.layers import compose, no_signal
from runtime.loop import InferenceLoop

log = logging.getLogger("perception.stream")

BOUNDARY = b"--frame"


class Streamer:
    def __init__(self, loop: InferenceLoop) -> None:
        self.loop = loop
        self._jpeg: bytes | None = None
        self._seq = -1
        self._at = 0.0
        self.encoded = 0

    def jpeg(self) -> bytes:
        """Latest enriched frame, re-composed at most STREAM_FPS times a second."""
        now = time.time()
        view = self.loop.view
        stale = view.seq != self._seq
        due = now - self._at >= 1.0 / max(CFG.STREAM_FPS, 1.0)
        if self._jpeg is not None and not (stale and due):
            return self._jpeg

        frame = no_signal() if view.frame is None else view.frame.copy()

        ok, buf = cv2.imencode(".jpg", compose(frame, view.layers),
                               [cv2.IMWRITE_JPEG_QUALITY, CFG.JPEG_QUALITY])
        if ok:
            self._jpeg, self._seq, self._at = buf.tobytes(), view.seq, now
            self.encoded += 1
        return self._jpeg or b""

    def raw(self) -> bytes:
        """The frame with nothing drawn on it, for the agent's vision model.

        Overlays would be read as part of the scene, so `describe()` gets the
        world, not our annotations of it.
        """
        frame = self.loop.view.frame
        if frame is None:
            return b""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else b""

    async def mjpeg(self):
        interval = 1.0 / max(CFG.STREAM_FPS, 1.0)
        while True:
            data = self.jpeg()
            if data:
                yield (BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            await asyncio.sleep(interval)
