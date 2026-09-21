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


def _fourcc_of(cap: cv2.VideoCapture) -> str:
    """The four-character pixel format the driver is actually delivering."""
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    if value <= 0:
        return ""
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


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
        #: A source the operator asked for, picked up by the capture thread at
        #: the top of its next pass. None means "carry on".
        self._pending: str | None = None
        self._switch_done = threading.Event()
        self._switch_result: tuple[bool, str] | None = None
        #: Held for the whole of one switch, so two of them queue rather than
        #: overwriting each other's request. Never taken by the capture thread.
        self._switch_lock = threading.Lock()
        #: What the driver actually gave us, which is often not what was asked
        #: for. Reported by GET /cameras rather than the requested values.
        self.width: int | None = None
        self.height: int | None = None
        self.fps: float | None = None
        #: The pixel format the driver actually settled on, which is not
        #: always the one that was asked for.
        self.fourcc: str | None = None
        self.switches = 0

    # 1. Source handling -----------------------------------------------
    @property
    def is_file(self) -> bool:
        return not self.source.isdigit() and "://" not in self.source

    def _open(self, source: str | None = None) -> bool:
        source = self.source if source is None else source
        is_file = not source.isdigit() and "://" not in source
        target: str | int = int(source) if source.isdigit() else source

        from runtime.cameras import backend_flag

        # The backend is not optional on Windows: a DirectShow index means
        # nothing to MSMF, so enumerating with one and opening with the other
        # gets you a different camera. See runtime/cameras.py.
        cap = (cv2.VideoCapture(target, backend_flag()) if not is_file
               else cv2.VideoCapture(target))
        if not cap.isOpened():
            cap.release()
            return False

        if not is_file:
            # Ask the driver for the shallowest buffer it will give us. Best
            # effort: several backends ignore it, which is why the slot above
            # exists regardless.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._configure(cap)

        self._cap = cap
        self.source = source
        self._read_back(cap)
        self.opened = True
        return True

    def _configure(self, cap: cv2.VideoCapture) -> None:
        """Size first, then pixel format — and check the format actually took.

        The order is load-bearing and it is the opposite of what the docs
        suggest. Measured on a C920 over DirectShow:

            FOURCC then size  ->  YUY2  1280x720  10.0 fps
            size then FOURCC  ->  MJPG  1280x720  30.5 fps

        Setting the resolution **resets the pixel format** on this driver, so
        asking for MJPG first has no effect at all. And the format is what
        decides the frame rate: YUY2 at 720p is about 55 MB/s, well past what
        USB 2.0 carries, so the driver simply delivers 10 fps instead.

        Some drivers want it the other way round, so if the format did not
        take, try the other order before giving up. Neither order is right
        everywhere, which is why this verifies instead of assuming.
        """
        if CFG.CAPTURE_WIDTH and CFG.CAPTURE_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAPTURE_HEIGHT)

        if not CFG.CAMERA_FOURCC:
            return

        wanted = CFG.CAMERA_FOURCC[:4].upper()
        code = cv2.VideoWriter_fourcc(*wanted)
        cap.set(cv2.CAP_PROP_FOURCC, code)

        if _fourcc_of(cap) != wanted and CFG.CAPTURE_WIDTH and CFG.CAPTURE_HEIGHT:
            cap.set(cv2.CAP_PROP_FOURCC, code)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAPTURE_HEIGHT)

        if _fourcc_of(cap) != wanted:
            log.warning("camera kept %s rather than the requested %s; at this "
                        "resolution that may cost frame rate",
                        _fourcc_of(cap), wanted)

    def _read_back(self, cap: cv2.VideoCapture) -> None:
        """What we got, not what we asked for. `set` returns True having done
        nothing on plenty of drivers, and a refused pixel format shows up only
        as a frame rate nobody can explain."""
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        self.fourcc = _fourcc_of(cap)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = round(fps, 1) if 0.0 < fps < 1000.0 else None

    def switch(self, source: str, timeout: float = 15.0) -> tuple[bool, str]:
        """Change source, and say whether it worked.

        The capture thread does the opening and closing — **only that thread
        ever touches `_cap`**. Releasing a VideoCapture from here while the
        thread sits inside `read()` on the same handle is undefined behaviour
        in OpenCV, and the crash would land in the middle of a demo. So this
        posts a request and waits for the answer.

        Mutating in place rather than handing back a new Capture is deliberate:
        the frame loop and the service each hold a reference to this object,
        and swapping those under a running loop is a race for no benefit.

        One switch at a time. Two callers sharing one pending slot and one
        completion Event go wrong in two ways: the second overwrites the
        first's request, so both then read the *same* result and one of them
        reports success for a camera it did not select; and the second's
        `clear()` can land after the thread has already signalled, leaving the
        first to wait out its whole timeout. Concurrent switches are rare but
        entirely reachable — two clicks, two tabs, a retry — and "the console
        says C920 while the pipeline reads the laptop" is exactly the kind of
        failure nobody diagnoses on stage. So they queue.
        """
        with self._switch_lock:
            with self._lock:
                self._pending = str(source)
                self._switch_result = None
            self._switch_done.clear()

            if not self._switch_done.wait(timeout):
                return False, (f"the capture thread did not answer within "
                               f"{timeout:.0f}s; the camera may be wedged")
            with self._lock:
                return self._switch_result or (False, "no result")

    def _do_switch(self, source: str) -> tuple[bool, str]:
        """Capture thread only. Open the new source, or put the old one back.

        The old camera is released first. On one USB bus a second 1080p stream
        often cannot be opened alongside the first, so holding both to compare
        them fails for reasons that have nothing to do with the new camera.

        The revert is the point of this method. A pipeline left blind because a
        dropdown was wrong is worse than one that refuses to switch.
        """
        previous = self.source
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.opened = False

        if self._open(str(source)):
            self.switches += 1
            log.info("camera -> %s (%sx%s @ %s)", self.source,
                     self.width, self.height, self.fps)
            return True, self.source

        detail = f"could not open {source!r}"
        if self._open(previous):
            log.warning("%s — stayed on %s", detail, previous)
            return False, f"{detail}; still on {previous!r}"

        # Both gone. The reconnect loop below keeps trying the original.
        self.source = previous
        log.error("%s, and the previous source %r did not come back",
                  detail, previous)
        return False, f"{detail}, and {previous!r} did not reopen either"

    def _take_pending(self) -> str | None:
        with self._lock:
            pending, self._pending = self._pending, None
        return pending

    def _finish_switch(self, result: tuple[bool, str]) -> None:
        with self._lock:
            self._switch_result = result
        self._switch_done.set()

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
            pending = self._take_pending()
            if pending is not None:
                self._finish_switch(self._do_switch(pending))
                backoff = 0.5
                continue

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
