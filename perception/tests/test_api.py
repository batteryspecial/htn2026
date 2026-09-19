"""Tests of the service as Danny and Kevin will actually meet it: over HTTP.

The orchestrator does not exist yet, so these double as the contract it will
be written against. Nothing here touches internals that a network client
could not see, except to fake the camera.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from contracts import Phase
from detectors.registry import Registry
from runtime.capture import Capture
from runtime.events import make_buses
from runtime.loader import Loader
from runtime.loop import InferenceLoop
from runtime.state import Machine
from server.api import Service, create_app
from tests.test_pipeline import FakeCapture, boxes

PENCIL = ("pencil", 320, 240, 100)


@pytest.fixture
def svc(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "models:\n"
        "  - name: fake\n    type: fake\n    preload: true\n"
        "  - name: fixed\n    type: fake\n    preload: true\n"
        "  - name: gpu_only\n    type: ultralytics_fixed\n"
        "    weights: nope.engine\n    requires_cuda: true\n"
    )
    registry = Registry.from_yaml(p)
    registry.preload()
    events, targets = make_buses()
    machine = Machine(emit=events.publish)
    machine.on_registry_ready()
    capture = FakeCapture()
    loader = Loader(registry).start()
    loop = InferenceLoop(capture, loader, machine, targets, events)
    s = Service(registry=registry, capture=capture, loader=loader, loop=loop,
                machine=machine, events=events, targets=targets)
    s.now = 1000.0
    yield s
    loader.stop()


@pytest.fixture
def client(svc):
    # run_threads=False: the tests drive frames by hand so nothing is timing
    # dependent, but the app, the routes and the buses are all real.
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.svc = svc
        yield c


def advance(svc, n, seen=None, dt=0.05):
    det = svc.registry.get("fake")
    for _ in range(n):
        if seen is not None:
            det.set_script([seen])
        svc.now += dt
        svc.capture.tick(svc.now)
        svc.loop.step(svc.now)


def settle(svc, timeout=3.0):
    """Step frames until the background worker has handed its task over."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        advance(svc, 1)
        if svc.machine.phase not in (Phase.SWITCHING, Phase.LOADING_MODEL):
            return
        time.sleep(0.005)
    raise AssertionError(f"stuck in {svc.machine.phase}")


def spec_body(instruction_id="i1", spec_id="s1", model="fake", **over):
    spec = {"spec_id": spec_id, "model": model,
            "targets": [{"ref": "t1", "detect": ["pencil"]}]}
    spec.update(over)
    return {"instruction_id": instruction_id, "spec": spec}


# 1. POST /spec -----------------------------------------------------------
def test_a_valid_spec_is_accepted_immediately(client):
    r = client.post("/spec", json=spec_body())
    assert r.status_code == 202
    assert r.json()["spec_id"] == "s1"


def test_accepting_does_not_wait_for_the_model(client):
    """202 means queued, not applied. The orchestrator must never block on us."""
    client.post("/spec", json=spec_body())
    assert client.svc.loop.active is None


def test_a_posted_spec_reaches_the_pipeline(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    assert client.get("/health").json()["phase"] == "TRACKING"


@pytest.mark.parametrize("bad", [
    {"instruction_id": "i1"},
    {"spec": {"spec_id": "s1", "targets": []}},
    spec_body(targets=[]),
    spec_body(targets=[{"ref": "t", "detect": ["a"], "select": "vibes"}]),
    spec_body(mode="teleport"),
    {"instruction_id": "i1", "spec": {"spec_id": "s1",
     "targets": [{"ref": "t", "detect": ["a"]}], "junk": 1}},
])
def test_a_malformed_spec_is_rejected(client, bad):
    assert client.post("/spec", json=bad).status_code == 422


def test_an_unknown_model_is_rejected_with_the_alternatives(client):
    r = client.post("/spec", json=spec_body(model="rfdetr"))
    assert r.status_code == 422
    assert "fake" in r.json()["detail"]


def test_a_model_this_machine_cannot_run_is_rejected(client):
    r = client.post("/spec", json=spec_body(model="gpu_only"))
    assert r.status_code == 422
    assert "unavailable" in r.json()["detail"]


def test_a_rejection_also_shows_on_the_status_board(client):
    """A caller that ignores the HTTP error must still see it in the UI."""
    client.post("/spec", json=spec_body(model="rfdetr"))
    stages = [e.stage for e in client.svc.events.history()]
    assert "failed" in stages


def test_a_rejected_spec_never_reaches_the_pipeline(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    client.post("/spec", json=spec_body("i2", "s2", model="rfdetr"))
    settle(client.svc)
    advance(client.svc, 3, seen=boxes(PENCIL))
    assert client.get("/health").json()["spec_id"] == "s1"


def test_a_fixed_vocab_model_rejects_an_impossible_class(client):
    """Checked up front, without loading anything, so the LLM gets told at once."""
    det = client.svc.registry.get("fixed")
    type(det).classes = property(lambda self: ["person", "dog"])
    try:
        r = client.post("/spec", json=spec_body(model="fixed"))
        assert r.status_code == 422
        assert "pencil" in r.json()["detail"]
    finally:
        del type(det).classes


# 2. DELETE /spec ---------------------------------------------------------
def test_delete_stops_the_car(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    assert client.get("/health").json()["phase"] == "TRACKING"

    assert client.delete("/spec").status_code == 202
    advance(client.svc, 2, seen=boxes(PENCIL))
    h = client.get("/health").json()
    assert h["phase"] == "IDLE" and h["spec_id"] is None


def test_delete_on_an_idle_service_is_harmless(client):
    assert client.delete("/spec").status_code == 202
    advance(client.svc, 2)
    assert client.get("/health").json()["phase"] == "IDLE"


# 3. GET /models ----------------------------------------------------------
def test_models_lists_everything_with_its_availability(client):
    d = client.get("/models").json()
    by_name = {m["name"]: m for m in d["models"]}
    assert set(by_name) == {"fake", "fixed", "gpu_only"}
    assert by_name["fake"]["available"] is True
    assert by_name["gpu_only"]["available"] is False
    assert "reason" in by_name["gpu_only"]
    assert d["device"]


def test_models_says_which_are_open_vocabulary(client):
    """The LLM needs this to know when it may invent a class name."""
    fake = next(m for m in client.get("/models").json()["models"] if m["name"] == "fake")
    assert fake["open_vocab"] is True


# 4. GET /health ----------------------------------------------------------
def test_health_before_any_spec(client):
    h = client.get("/health").json()
    assert h["phase"] == "IDLE"
    assert h["model"] is None and h["spec_id"] is None


def test_health_reports_the_running_spec_and_model(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    h = client.get("/health").json()
    assert h["spec_id"] == "s1" and h["model"] == "fake"
    assert h["visible"] is True and h["frames"] > 0


def test_health_still_answers_when_the_camera_is_dead(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    client.svc.capture.dead = True
    for _ in range(40):
        client.svc.now += 0.1
        client.svc.loop.step(client.svc.now)
    h = client.get("/health").json()
    assert h["phase"] == "FAULT"


# 5. WebSockets -----------------------------------------------------------
def test_target_socket_delivers_states(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    with client.websocket_connect("/ws/target") as ws:
        advance(client.svc, 5, seen=boxes(PENCIL))
        msg = json.loads(ws.receive_text())
    assert msg["phase"] and "cx" in msg and "visible" in msg


def test_target_states_carry_everything_the_controller_needs(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    with client.websocket_connect("/ws/target") as ws:
        got = []
        for _ in range(3):
            advance(client.svc, 2, seen=boxes(PENCIL))
            got.append(json.loads(ws.receive_text()))
    required = {"ts", "phase", "visible", "cx", "cy", "area", "conf", "spec_id",
                "mode", "model", "target_ref", "track_id", "label"}
    assert all(required <= set(m) for m in got)


def test_a_stalled_controller_makes_the_pipeline_drop_not_block(client):
    """Level-triggered: states are discarded for a client that falls behind.

    The backlog a client can accumulate is bounded by the socket, not by us:
    the pipeline itself never queues more than one state per client and never
    waits on a slow reader. A controller acting on a stale position is worse
    than one that skipped a few.
    """
    client.post("/spec", json=spec_body())
    settle(client.svc)
    with client.websocket_connect("/ws/target") as ws:
        for cx in range(60, 600, 20):
            advance(client.svc, 2, seen=boxes(("pencil", cx, 240, 80)))
        time.sleep(0.2)
        assert client.svc.targets.dropped > 0, "pipeline kept a backlog for a slow client"
        assert json.loads(ws.receive_text())["phase"], "socket survived the burst"
    assert client.get("/health").json()["frames"] > 0


def test_event_socket_replays_what_a_late_client_missed(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 5, seen=boxes(PENCIL))
    with client.websocket_connect("/ws/events") as ws:
        stages = [json.loads(ws.receive_text())["stage"] for _ in range(3)]
    assert stages == ["prepared", "applied", "active"]


def test_two_clients_both_get_states(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    with client.websocket_connect("/ws/target") as a, client.websocket_connect("/ws/target") as b:
        advance(client.svc, 5, seen=boxes(PENCIL))
        assert json.loads(a.receive_text())["phase"]
        assert json.loads(b.receive_text())["phase"]


# 6. Video ----------------------------------------------------------------
def test_single_frame_endpoint_returns_a_jpeg(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 3, seen=boxes(PENCIL))
    r = client.get("/frame.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"  # JPEG magic
    assert len(r.content) > 1000


def test_video_renders_before_any_frame_arrives(client):
    """A blank feed with NO SIGNAL beats a broken image icon on stage."""
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_one_encode_is_shared_by_every_viewer(client):
    client.post("/spec", json=spec_body())
    settle(client.svc)
    advance(client.svc, 3, seen=boxes(PENCIL))
    before = client.app.state.streamer.encoded
    for _ in range(5):
        client.get("/frame.jpg")
    assert client.app.state.streamer.encoded - before <= 1


def test_the_debug_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and "perception" in r.text
