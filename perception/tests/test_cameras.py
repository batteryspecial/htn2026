"""Choosing a camera.

No test here touches a real device. `runtime.cameras.list_cameras` is the one
door to the platform, and every test goes through a fake one — a suite that
opened the webcam would fail on CI, fail on a laptop with the lid shut, and
steal the camera from a demo running on the same machine.

The test that matters most is the revert: a switch that fails must leave the
pipeline reading the camera it was reading before. A pipeline left blind
because a dropdown was wrong is worse than one that refuses to switch.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import runtime.cameras as cameras_mod
from runtime.cameras import CameraInfo
from runtime.capture import Capture
from server.api import Service, create_app
from tests.rig import Rig

DEVICES = [
    CameraInfo(index=0, name="Integrated Camera", backend="dshow"),
    CameraInfo(index=1, name="Lenovo Virtual Camera", backend="dshow"),
    CameraInfo(index=2, name="HD Pro Webcam C920", backend="dshow"),
]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def fake_devices(monkeypatch):
    """Three cameras that exist, and nothing that opens one."""
    def listing(active_source=None, *, refresh=False, probe=False):
        found = [CameraInfo(**c.as_dict()) for c in DEVICES]
        for camera in found:
            camera.active = str(camera.index) == str(active_source)
        return found

    monkeypatch.setattr(cameras_mod, "list_cameras", listing)
    monkeypatch.setattr(cameras_mod, "invalidate", lambda: None)
    return DEVICES


class SwitchableCapture:
    """The rig's fake camera, plus the one method the route needs.

    `refuse` is the point: it is how the revert path gets exercised without a
    device that can be unplugged mid-test.
    """

    def __init__(self, inner, source="0"):
        self._inner = inner
        self.source = source
        self.width, self.height, self.fps = 640, 480, 30.0
        self.fourcc = "MJPG"
        self.opened = True
        self.refuse = False
        self.switches = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def switch(self, source, timeout=15.0):
        if self.refuse:
            # Exactly what Capture._do_switch reports when the new device will
            # not open and the old one comes back.
            return False, f"could not open {source!r}; still on {self.source!r}"
        self.source = str(source)
        self.switches += 1
        return True, self.source


@pytest.fixture
def client(rig, fake_devices):
    capture = SwitchableCapture(rig.capture)
    svc = Service(registry=rig.registry, capture=capture, builder=rig.builder,
                  loop=rig.loop, health=rig.health, events=rig.event_bus,
                  states=rig.state_bus)
    rig.loop.capture = capture
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.rig = rig
        c.capture = capture
        yield c


# 1. Listing ----------------------------------------------------------------

def test_the_cameras_this_machine_has_are_listed_with_their_names(client):
    body = client.get("/cameras").json()

    assert [c["name"] for c in body["cameras"]] == [
        "Integrated Camera", "Lenovo Virtual Camera", "HD Pro Webcam C920"]


def test_the_live_camera_is_marked_and_carries_what_the_driver_reported(client):
    body = client.get("/cameras").json()

    live = [c for c in body["cameras"] if c["active"]]
    assert len(live) == 1
    assert live[0]["name"] == "Integrated Camera"
    # Not what was requested — what came back from the open handle.
    assert (live[0]["width"], live[0]["height"]) == (640, 480)


def test_a_file_source_is_still_reported_as_the_live_one(client):
    """A clip or a stream URL is a valid source and is in no device list."""
    client.capture.source = "clips/street.mp4"

    body = client.get("/cameras").json()

    live = next(c for c in body["cameras"] if c["active"])
    assert live["name"] == "clips/street.mp4"
    assert live["index"] == -1


# 2. Switching --------------------------------------------------------------

def test_switching_by_name_resolves_to_the_right_index(client):
    answer = client.post("/camera", json={"name": "HD Pro Webcam C920"})

    assert answer.status_code == 202
    assert answer.json()["source"] == "2"
    assert client.capture.source == "2"


def test_switching_by_index_works_too(client):
    assert client.post("/camera", json={"source": "2"}).status_code == 202
    assert client.capture.source == "2"


def test_a_name_that_is_not_plugged_in_is_refused_with_what_is(client):
    answer = client.post("/camera", json={"name": "Logitech Brio"})

    assert answer.status_code == 422
    detail = answer.json()["detail"]
    assert "Logitech Brio" in detail
    assert "HD Pro Webcam C920" in detail
    # And nothing moved.
    assert client.capture.source == "0"


def test_an_index_with_no_camera_behind_it_is_refused(client):
    answer = client.post("/camera", json={"source": "7"})

    assert answer.status_code == 422
    assert "no camera at index 7" in answer.json()["detail"]


def test_switching_to_the_one_already_running_is_refused_rather_than_done(client):
    """Re-opening the live camera would drop frames for no reason."""
    answer = client.post("/camera", json={"source": "0"})

    assert answer.status_code == 422
    assert "already reading" in answer.json()["detail"]
    assert client.capture.switches == 0


def test_an_empty_choice_says_what_it_wanted(client):
    answer = client.post("/camera", json={})

    assert answer.status_code == 422
    assert "give a source" in answer.json()["detail"]


# 3. The revert -------------------------------------------------------------

def test_a_camera_that_will_not_open_leaves_the_old_one_running(client):
    """The failure that must never blind the pipeline."""
    client.capture.refuse = True

    answer = client.post("/camera", json={"name": "HD Pro Webcam C920"})

    assert answer.status_code == 422
    assert "still on '0'" in answer.json()["detail"]
    assert client.capture.source == "0"


# 4. What a switch must not disturb ----------------------------------------

def test_behaviours_keep_running_across_a_switch(client):
    """A change of sensor is not a change of objective."""
    client.post("/behaviors", json={"kind": "highlight",
                                    "subject": {"detect": ["thing"]}})
    client.rig.settle()
    before = [b["id"] for b in client.get("/behaviors").json()["behaviors"]]
    assert before

    client.post("/camera", json={"source": "2"})
    client.rig.settle()

    assert [b["id"] for b in client.get("/behaviors").json()["behaviors"]] == before


def test_tracking_is_reset_because_track_ids_do_not_survive_a_new_lens(client):
    """A follow-cam holding track #7 would wait forever for a track that
    belongs to a different camera."""
    client.post("/camera", json={"source": "2"})

    assert client.rig.loop._pending_reset is not None
    assert "camera ->" in client.rig.loop._pending_reset


# 5. Backend selection ------------------------------------------------------

def test_windows_enumerates_and_opens_through_the_same_backend(monkeypatch):
    """A DirectShow index means nothing to MSMF. Measured on this machine,
    MSMF also takes 33s to open a C920 where DirectShow takes 2.3s."""
    monkeypatch.setattr(cameras_mod.sys, "platform", "win32")
    monkeypatch.setattr(cameras_mod.CFG, "CAMERA_BACKEND", "auto", raising=False)

    assert cameras_mod.backend_name() == "dshow"
    assert cameras_mod.backend_flag() == cameras_mod.cv2.CAP_DSHOW


def test_an_explicit_backend_wins_over_the_platform_default(monkeypatch):
    monkeypatch.setattr(cameras_mod.CFG, "CAMERA_BACKEND", "msmf", raising=False)

    assert cameras_mod.backend_name() == "msmf"


# 6. Resolving --------------------------------------------------------------

def test_a_file_path_resolves_to_itself_without_consulting_any_device(fake_devices):
    source, detail = cameras_mod.resolve("clips/street.mp4")

    assert source == "clips/street.mp4"
    assert detail == "clips/street.mp4"


def test_a_stream_url_resolves_to_itself(fake_devices):
    assert cameras_mod.resolve("rtsp://10.0.0.4/stream")[0] == "rtsp://10.0.0.4/stream"


def test_a_capture_reports_what_the_driver_gave_not_what_was_asked(monkeypatch):
    """`cap.set` returns True having done nothing on plenty of drivers, so the
    numbers shown to the operator are read back off the handle."""
    class FakeHandle:
        def isOpened(self): return True
        def set(self, *_): return True
        def release(self): pass

        def get(self, prop):
            import cv2
            return {cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                    cv2.CAP_PROP_FPS: -1.0}.get(prop, 0.0)

    monkeypatch.setattr(cameras_mod.cv2, "VideoCapture",
                        lambda *a, **kw: FakeHandle())
    capture = Capture(source="2")

    assert capture._open() is True
    assert (capture.width, capture.height) == (1280, 720)
    # DirectShow reports -1 for FPS. Unknown is None, never -1.
    assert capture.fps is None


# 7. Two at once ------------------------------------------------------------

def test_concurrent_switches_queue_instead_of_overwriting_each_other():
    """Two clicks, two tabs, or a retry must not make the console lie.

    With one pending slot and one completion Event shared between callers,
    the second request overwrites the first and both then read the same
    result — so one of them reports success for a camera it did not pick.
    """
    import threading

    import numpy as np

    opened: list[str] = []

    class Handle:
        def isOpened(self): return True
        def set(self, *_): return True
        def get(self, *_): return 0.0
        def release(self): pass

        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def slow_open(target, *_a, **_kw):
        # Long enough that the second caller is certainly inside switch().
        time.sleep(0.2)
        opened.append(str(target))
        return Handle()

    capture = Capture(source="0")
    import runtime.capture as capture_mod
    original, capture_mod.cv2.VideoCapture = capture_mod.cv2.VideoCapture, slow_open
    try:
        capture.start()
        answers: dict[str, tuple[bool, str]] = {}

        def ask(source):
            answers[source] = capture.switch(source, timeout=10.0)

        threads = [threading.Thread(target=ask, args=(s,)) for s in ("2", "3")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)

        # Each caller got its own answer, naming the camera it asked for.
        assert answers["2"] == (True, "2")
        assert answers["3"] == (True, "3")
        # And both switches actually happened, rather than one being lost.
        assert "2" in opened and "3" in opened
    finally:
        capture_mod.cv2.VideoCapture = original
        capture.stop()


def test_the_negotiated_pixel_format_is_reported_not_the_requested_one(monkeypatch):
    """A refused FOURCC is otherwise invisible until someone wonders why the
    frame rate is 10.

    Measured on a C920 over DirectShow: setting the resolution resets the
    pixel format, so asking for MJPG *first* leaves you on YUY2 — which at
    720p exceeds USB 2.0 bandwidth and delivers 10fps instead of 30.
    """
    import runtime.capture as capture_mod

    applied: list[tuple[str, object]] = []

    class Handle:
        def __init__(self):
            self.fourcc = 0
            self.w = self.h = 0

        def isOpened(self): return True
        def release(self): pass

        def set(self, prop, value):
            if prop == capture_mod.cv2.CAP_PROP_FOURCC:
                self.fourcc = int(value)
                applied.append(("fourcc", value))
            elif prop == capture_mod.cv2.CAP_PROP_FRAME_WIDTH:
                self.w = int(value)
                applied.append(("width", value))
                # The behaviour that made this bug: size resets the format.
                self.fourcc = 0
            elif prop == capture_mod.cv2.CAP_PROP_FRAME_HEIGHT:
                self.h = int(value)
                applied.append(("height", value))
            return True

        def get(self, prop):
            if prop == capture_mod.cv2.CAP_PROP_FOURCC: return float(self.fourcc)
            if prop == capture_mod.cv2.CAP_PROP_FRAME_WIDTH: return float(self.w)
            if prop == capture_mod.cv2.CAP_PROP_FRAME_HEIGHT: return float(self.h)
            return 0.0

    monkeypatch.setattr(capture_mod.CFG, "CAPTURE_WIDTH", 1280, raising=False)
    monkeypatch.setattr(capture_mod.CFG, "CAPTURE_HEIGHT", 720, raising=False)
    monkeypatch.setattr(capture_mod.CFG, "CAMERA_FOURCC", "MJPG", raising=False)
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", lambda *a, **kw: Handle())

    capture = Capture(source="2")
    assert capture._open() is True

    # Size was applied before the format, so the format survives.
    assert [a for a, _ in applied][:3] == ["width", "height", "fourcc"]
    assert capture.fourcc == "MJPG"
    assert (capture.width, capture.height) == (1280, 720)


# 8. Turning the optional models off without uninstalling -------------------

def test_disable_models_reproduces_a_machine_without_the_packages(monkeypatch):
    """A/B switch. The packages stay on disk; the pipeline behaves as if they
    were never installed, so a suspected regression can be tested both ways
    without a five-minute reinstall in either direction."""
    from zoo.registry import Entry
    import zoo.registry as registry_mod

    monkeypatch.setattr(registry_mod.CFG, "DISABLE_MODELS",
                        "hands, wholebody ,OCR", raising=False)

    for name, role in (("hands", "hands"), ("wholebody", "wholebody"),
                       ("ocr", "ocr")):
        entry = Entry(name=name, role=role, type="mediapipe_hands")
        entry.check_available("cuda")
        assert entry.available is False
        assert entry.unavailable_reason == "disabled by DISABLE_MODELS"


def test_disabling_a_role_switches_off_every_model_in_it(monkeypatch):
    from zoo.registry import Entry
    import zoo.registry as registry_mod

    monkeypatch.setattr(registry_mod.CFG, "DISABLE_MODELS", "pose", raising=False)

    entry = Entry(name="pose_fullbody", role="pose", type="ultralytics_pose")
    entry.check_available("cuda")

    assert entry.available is False


def test_nothing_is_disabled_by_default(monkeypatch):
    """The switch is for testing. It must never be on by accident."""
    import zoo.registry as registry_mod

    monkeypatch.setattr(registry_mod.CFG, "DISABLE_MODELS", "", raising=False)

    assert registry_mod._disabled() == set()
