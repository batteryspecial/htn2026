"""Which cameras this machine has, and which one OpenCV will open for an index.

OpenCV cannot list cameras: `cv2.VideoCapture(i)` either opens or it does not,
and nothing comes back with a name on it. So this module does the listing, and
the whole platform problem is confined to one file that the tests monkeypatch.

**The index and the backend belong together.** On Windows the names come from
DirectShow, and a DirectShow index is only meaningful to `cv2.CAP_DSHOW`.
OpenCV's default backend on Windows is MSMF, which orders devices differently —
enumerate with one and open with the other and you get the wrong camera, which
looks like a bug in everything except the place it actually is. `backend_flag()`
is the single answer to "how do we open a camera here", and both enumeration
and `Capture` go through it.

Listing is cached: probing a device is disruptive (opening an index that is
already in use can steal it or fail), so it happens on request and not on
every poll of the settings drawer.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

import cv2
from pydantic import BaseModel, ConfigDict

from config import CFG

log = logging.getLogger("perception.cameras")

#: How many indices the probe fallback tries before giving up.
PROBE_LIMIT = 8

#: Enumeration is stable for as long as nobody replugs anything.
CACHE_TTL_S = 30.0

# Measured on this machine opening a C920 by index, OpenCV 5.0 on Windows:
#
#     DSHOW    2.35s
#     MSMF    32.90s
#     ANY     55.64s
#
# which is why the backend is chosen rather than left to OpenCV. MSMF is the
# default on Windows and would put half a minute in front of every switch.
# DSHOW does not report FPS (it returns -1); `Capture._read_back` treats that
# as unknown rather than as a number.


class CameraChoice(BaseModel):
    """Body of POST /camera. An index, or a name.

    Not in `linker/schemas.py`: that file is the wire contract every service
    shares and is changed only by agreement. Picking a camera is local to this
    service and to the console's settings drawer.
    """

    model_config = ConfigDict(extra="forbid")

    #: A camera index ("2"), a file path, or a stream URL.
    source: str | None = None
    #: A device name from GET /cameras, resolved to an index when it is opened.
    #: Preferred over an index: indices shuffle when something is replugged.
    name: str | None = None


@dataclass
class CameraInfo:
    """One camera, as the settings drawer will show it."""

    index: int
    name: str
    #: Which OpenCV backend this index is valid for. See the module docstring.
    backend: str = "auto"
    #: True for the one the pipeline is reading right now.
    active: bool = False
    #: False when it was enumerated but would not open.
    available: bool = True
    #: What actually came back, not what was asked for. Only known for the
    #: active camera and for anything a refresh probed.
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    detail: str | None = None

    def label(self) -> str:
        if self.width and self.height:
            return f"{self.name} · {self.width}×{self.height}"
        return self.name

    def as_dict(self) -> dict:
        return {
            "index": self.index, "name": self.name, "backend": self.backend,
            "active": self.active, "available": self.available,
            "width": self.width, "height": self.height, "fps": self.fps,
            "detail": self.detail,
        }


@dataclass
class _Cache:
    cameras: list[CameraInfo] = field(default_factory=list)
    at: float = 0.0


_cache = _Cache()


# 1. Backend ----------------------------------------------------------------

def backend_name() -> str:
    """The backend to enumerate and open with, resolved from config.

    `auto` means DirectShow on Windows. MSMF is OpenCV's default there and is
    the worse choice for this hardware: it takes seconds to open a Logitech,
    sometimes fails outright, and its device order does not match the names
    DirectShow gives us.
    """
    want = (CFG.CAMERA_BACKEND or "auto").strip().lower()
    if want != "auto":
        return want
    if sys.platform == "win32":
        return "dshow"
    if sys.platform == "darwin":
        return "avfoundation"
    return "v4l2"


def backend_flag() -> int:
    """The `cv2.CAP_*` constant for `backend_name()`, or CAP_ANY."""
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "any": cv2.CAP_ANY,
        "auto": cv2.CAP_ANY,
    }.get(backend_name(), cv2.CAP_ANY)


# 2. Enumeration ------------------------------------------------------------

def _windows_names() -> list[str] | None:
    """DirectShow device names, in DirectShow index order.

    None when pygrabber is not installed, which is not an error — the probe
    fallback below still finds the cameras, it just cannot name them.
    """
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        log.info("pygrabber not installed — cameras will be listed without names")
        return None
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:                            # noqa: BLE001
        log.warning("could not enumerate DirectShow devices: %s", exc)
        return None


def _open_briefly(index: int, timeout_s: float) -> tuple[bool, dict]:
    """Open an index, read what the driver reports, close it again.

    Only ever called on a camera the pipeline is *not* holding. Opening a
    device that is already in use fails on some drivers and steals it on
    others, and either would break the running demo to populate a dropdown.
    """
    started = time.monotonic()
    cap = cv2.VideoCapture(index, backend_flag())
    try:
        if not cap.isOpened():
            return False, {"detail": "would not open"}
        facts = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 1) or None,
        }
        if time.monotonic() - started > timeout_s:
            facts["detail"] = "slow to open"
        return True, facts
    finally:
        cap.release()


def list_cameras(active_source: str | None = None, *,
                 refresh: bool = False,
                 probe: bool = False) -> list[CameraInfo]:
    """Every camera this machine offers.

    `probe` opens each inactive device to find out its resolution, which is
    accurate and slow. Without it the list is names only, which is all the
    drawer needs to let someone pick one.
    """
    now = time.monotonic()
    if not refresh and _cache.cameras and now - _cache.at < CACHE_TTL_S:
        cameras = [CameraInfo(**c.as_dict()) for c in _cache.cameras]
    else:
        cameras = _enumerate(probe=probe or refresh, active_source=active_source)
        _cache.cameras = [CameraInfo(**c.as_dict()) for c in cameras]
        _cache.at = now

    for camera in cameras:
        camera.active = active_source is not None and str(camera.index) == active_source
    return cameras


def _enumerate(*, probe: bool, active_source: str | None) -> list[CameraInfo]:
    backend = backend_name()
    names = _windows_names()

    if names is not None:
        cameras = [CameraInfo(index=i, name=name, backend=backend)
                   for i, name in enumerate(names)]
    else:
        # No name source. Find what opens, and call them what they are.
        cameras = []
        for index in range(PROBE_LIMIT):
            if active_source is not None and str(index) == active_source:
                cameras.append(CameraInfo(index=index, name=f"Camera {index}",
                                          backend=backend))
                continue
            ok, facts = _open_briefly(index, CFG.CAMERA_OPEN_TIMEOUT_S)
            if ok:
                cameras.append(CameraInfo(index=index, name=f"Camera {index}",
                                          backend=backend, **facts))
        return cameras

    if probe:
        for camera in cameras:
            if active_source is not None and str(camera.index) == active_source:
                continue  # never touch the one the pipeline is reading
            ok, facts = _open_briefly(camera.index, CFG.CAMERA_OPEN_TIMEOUT_S)
            camera.available = ok
            for key, value in facts.items():
                setattr(camera, key, value)

    return cameras


def invalidate() -> None:
    """Forget the cached listing. Called after a switch."""
    _cache.cameras = []
    _cache.at = 0.0


# 3. Resolving a request ----------------------------------------------------

def resolve(source: str | None = None, name: str | None = None) -> tuple[str | None, str]:
    """Turn what the operator asked for into a source string for `Capture`.

    Returns `(source, detail)`. A None source means it could not be resolved,
    and `detail` says why in a sentence worth showing someone.

    A name is resolved here rather than stored as an index because indices
    shuffle on replug: unplug the webcam and index 2 may become something
    else entirely. Silently switching to the wrong camera is worse than
    refusing, so a name that is no longer present is an error, not a fallback
    to index 0.
    """
    if name:
        wanted = name.strip().lower()
        for camera in list_cameras():
            if camera.name.strip().lower() == wanted:
                return str(camera.index), camera.name
        known = [c.name for c in list_cameras()]
        return None, f"no camera named {name!r} is connected; found {known}"

    if source is None or not str(source).strip():
        return None, "give a source: a camera index, a file path, or a stream URL"

    source = str(source).strip()
    if source.isdigit():
        cameras = list_cameras()
        match = next((c for c in cameras if c.index == int(source)), None)
        if cameras and match is None:
            known = [f"{c.index}: {c.name}" for c in cameras]
            return None, f"no camera at index {source}; found {known}"
        return source, match.name if match else f"Camera {source}"

    # A file or a URL. Capture handles both; there is nothing to enumerate.
    return source, source
