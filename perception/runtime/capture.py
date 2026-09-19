"""Frame source. One thread, one slot, newest frame wins.

The slot is the whole point. Queued frames are the usual reason a network
camera feels laggy even on fast hardware: the pipeline ends up steering from
where the target was a second ago. Dropping frames is the correct behaviour
for a control loop, so this thread overwrites rather than buffers, and the
inference loop always reads the most recent frame that exists.

Handles three sources through one interface: a webcam index, a file that loops
forever, and a URL that reconnects on its own.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from config import CFG

log = logging.getLogger("perception.capture")


class Capture:
    def __init__(self, source: str | None = None, fps_cap: float | None = None) -> None:
        self.source = str(source if source is not None else CFG.VIDEO_SOURCE)
        self.fps_cap = fps_cap if fps_cap is not None else CFG.CAPTURE_FPS_CAP
        self._frame: np.ndarray | None = None
        self._frame_ts: float = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.opened = False
        self.reconnects = 0

    # 1. Source handling -----------------------------------------------
    @property
    def is_file(self) -> bool:
        return not self.source.isdigit() and "://" not in self.source

    def _open(self) -> bool:
        target: str | int = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            return False
        if not self.is_file:
            # Ask the driver for the shallowest buffer it will give us. Best
            # effort: several backends ignore it, which is why the slot above
            # exists regardless.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.opened = True
        return True

    def _interval(self) -> float:
        """Files play at their own rate; live sources run as fast as they arrive."""
        if self.is_file and self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if 1.0 < fps < 240.0:
                return 1.0 / fps
        return 1.0 / self.fps_cap if self.fps_cap else 0.0

    # 2. Thread --------------------------------------------------------
    def start(self) -> Capture:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self.opened = False

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            if self._cap is None and not self._open():
                self.opened = False
                log.warning("cannot open %s, retrying in %.1fs", self.source, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            interval = self._interval()

            ok, frame = self._cap.read()
            if not ok:
                if self.is_file:
                    # A clip is a loop, not an ending.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                log.warning("read failed on %s, reconnecting", self.source)
                self.reconnects += 1
                self._cap.release()
                self._cap = None
                self.opened = False
                continue

            with self._lock:
                self._frame = frame
                self._frame_ts = time.time()
                self._seq += 1
            if interval:
                self._stop.wait(interval)

    # 3. Reading -------------------------------------------------------
    def read(self) -> tuple[np.ndarray | None, float, int]:
        """Latest frame, its timestamp and its sequence number.

        Returns the same frame again if the consumer is faster than the source;
        the sequence number is how a caller tells a repeat from a new frame.
        """
        with self._lock:
            return self._frame, self._frame_ts, self._seq

    def age(self, now: float | None = None) -> float:
        """Seconds since the last frame arrived. Large means the camera died."""
        if self._frame_ts == 0.0:
            return float("inf")
        return (now if now is not None else time.time()) - self._frame_ts

    def wait_for_frame(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._seq > 0:
                return True
            time.sleep(0.01)
        return False
