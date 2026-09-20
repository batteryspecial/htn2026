"""Reading characters off the frame, for the keyboard skill.

Aux model #4, and the first one that cannot run inline. Pose, hands and
wholebody are a few tens of milliseconds and the loop calls them directly.
EasyOCR is a detector plus a recogniser over the whole frame — 150-400ms on
CPU — and a loop that waited for it would drop to three frames a second, which
is not a demo.

So this model owns a thread. The loop hands it the newest frame and takes the
most recent finished read, both without blocking, exactly the way `Capture`
hands over frames: one slot, newest wins, no queue. A keyboard does not move
much between frames, so a read that is 300ms old is still true.

Only single characters are kept. A keyboard is covered in words — Enter,
Shift, Backspace, brand names — and none of them are keys the skill can point
at. Filtering to one character throws away almost all of that for free.
"""

from __future__ import annotations

import logging
import string
import threading
import time

import numpy as np

from config import CFG

log = logging.getLogger("perception.ocr")

#: Characters a key can be. Anything else EasyOCR reads is not a key we can
#: locate on the template.
ALLOWED = set(string.ascii_lowercase + string.digits)


class EasyOCR:
    """EasyOCR, English, single characters, on its own thread.

    The thread starts on the first `latest()` rather than at load, so a
    pipeline that never installs a keyboard behaviour never starts it.
    """

    role = "ocr"

    def __init__(self, name: str, weights=None) -> None:
        self.name = name
        # Accepted and ignored: easyocr fetches its own weights.
        self.weights = weights
        self._reader = None
        self._loaded = False
        #: Newest frame in, newest finished read out. Both single slots.
        self._pending: np.ndarray | None = None
        self._result: list[tuple[str, float, float]] = []
        self._result_at: float = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reads = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        import easyocr

        log.info("loading easyocr (english)")
        # gpu=False on purpose: the detector owns the GPU, and OCR is off the
        # critical path anyway because it runs on its own thread.
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self._loaded = True

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._reader is not None:
            self.read(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    # 1. The blocking read ----------------------------------------------
    def read(self, frame: np.ndarray) -> list[tuple[str, float, float]]:
        """(character, x, y) for every single-character result, in pixels.

        Synchronous and slow. The loop never calls this; the thread does.
        """
        if self._reader is None:
            return []
        out: list[tuple[str, float, float]] = []
        for box, text, conf in self._reader.readtext(frame):
            if conf < CFG.OCR_MIN_CONF:
                continue
            char = text.strip().lower()
            if len(char) != 1 or char not in ALLOWED:
                continue
            pts = np.asarray(box, dtype=np.float32)
            out.append((char, float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        return out

    # 2. The non-blocking face the loop sees -----------------------------
    def latest(self, frame: np.ndarray) -> list[tuple[str, float, float]]:
        """Offer a frame, take the newest finished read. Never blocks.

        Returns the empty list until the first read completes, and keeps
        returning the last good one after that — a keyboard that was there
        300ms ago is still there.
        """
        if self._thread is None:
            self._start()
        with self._lock:
            self._pending = frame
            result = self._result
        self._wake.set()
        return result

    def age(self, now: float) -> float:
        """Seconds since the last completed read. `inf` before the first."""
        return (now - self._result_at) if self._result_at else float("inf")

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ocr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        interval = 1.0 / max(CFG.OCR_HZ, 0.1)
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            started = time.time()
            try:
                found = self.read(frame)
            except Exception:
                # A failed read degrades the keyboard behaviour into
                # SEARCHING. It must never take the thread down with it.
                log.exception("ocr read failed")
                continue
            with self._lock:
                self._result = found
                self._result_at = time.time()
            self.reads += 1
            # Rate limit here rather than in the loop: reading flat out would
            # pin a core for no gain, because a keyboard does not move.
            slept = time.time() - started
            if slept < interval:
                time.sleep(interval - slept)
