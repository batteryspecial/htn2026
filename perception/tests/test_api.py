"""The service as the agent will meet it: over HTTP.

The orchestrator is being written against this, so these double as its
contract. Nothing here reaches past what a network client can see.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from contracts import Phase
from server.api import Service, create_app
from tests.rig import Rig, boxes

THING = ("thing", 320, 240, 100)
OTHER = ("other", 100, 100, 60)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def client(rig):
    svc = Service(registry=rig.registry, capture=rig.capture, builder=rig.builder,
                  loop=rig.loop, machine=rig.machine, events=rig.event_bus,
                  states=rig.loop.state_bus)
    # run_threads=False: frames are driven by hand so nothing is timing
    # dependent, but the app, the routes and the buses are all real.
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.rig = rig
        yield c


def body(behavior_id="b1", kind="highlight", detect=("thing",), instruction_id="i1", **sub):
    return {"instruction_id": instruction_id,
            "behavior": {"behavior_id": behavior_id, "kind": kind,
                         "subject": {"detect": list(detect), **sub}}}


# 1. Starting behaviours --------------------------------------------------
def test_a_behavior_is_accepted_immediately(client):
    r = client.post("/behaviors", json=body())
    assert r.status_code == 202 and r.json()["behavior_id"] == "b1"


def test_accepting_does_not_wait_for_the_pipeline(client):
    """202 means queued, not running. The agent must never block on us."""
    client.post("/behaviors", json=body())
    assert client.rig.loop.behaviors == {}


def test_a_posted_behavior_starts_running(client):
    client.post("/behaviors", json=body())
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 1
    assert client.rig.status("b1").state == "active"


def test_several_behaviors_run_at_once(client):
    client.post("/behaviors", json=body("a", detect=("thing",)))
    client.post("/behaviors", json=body("b", detect=("other",), instruction_id="i2"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING, OTHER))
    listed = client.get("/behaviors").json()["behaviors"]
    assert {b["behavior_id"] for b in listed} == {"a", "b"}
    assert all(b["state"] == "active" for b in listed)


@pytest.mark.parametrize("bad", [
    {"instruction_id": "i1"},
    {"instruction_id": "i1", "behavior": {"behavior_id": "b", "kind": "highlight"}},
    body(detect=()),
    {"instruction_id": "i1", "behavior": {"behavior_id": "b", "kind": "nonsense",
                                          "subject": {"detect": ["a"]}}},
    {"instruction_id": "i1", "behavior": {"behavior_id": "b", "kind": "highlight",
                                          "subject": {"detect": ["a"]}, "junk": 1}},
])
def test_a_malformed_behavior_is_rejected(client, bad):
    assert client.post("/behaviors", json=bad).status_code == 422


def test_an_unimplemented_kind_says_what_is_available(client):
    r = client.post("/behaviors", json=body(kind="privacy"))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "not implemented" in detail and "highlight" in detail


def test_a_duplicate_behavior_id_is_refused(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(2, seen=boxes(THING))
    r = client.post("/behaviors", json=body("b1", instruction_id="i2"))
    assert r.status_code == 422 and "already running" in r.json()["detail"]


def test_a_rejection_also_reaches_the_status_channel(client):
    """An agent that ignored the HTTP error still sees it in the UI."""
    client.post("/behaviors", json=body(kind="pan_to"))
    assert "failed" in client.rig.stages


def test_a_rejected_behavior_never_starts(client):
    client.post("/behaviors", json=body("good"))
    client.rig.settle()
    client.post("/behaviors", json=body("bad", kind="watch", instruction_id="i2"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert set(client.rig.loop.behaviors) == {"good"}


# 2. Stopping -------------------------------------------------------------
def test_a_behavior_can_be_stopped(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert client.delete("/behaviors/b1").status_code == 202
    client.rig.settle()
    client.rig.frames(2, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 0


def test_stopping_one_leaves_the_others(client):
    client.post("/behaviors", json=body("a", detect=("thing",)))
    client.post("/behaviors", json=body("b", detect=("other",), instruction_id="i2"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING, OTHER))
    client.delete("/behaviors/a")
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING, OTHER))
    assert set(client.rig.loop.behaviors) == {"b"}
    assert client.rig.status("b").state == "active"


def test_stopping_something_that_is_not_running_is_a_404(client):
    assert client.delete("/behaviors/ghost").status_code == 404


def test_clear_stops_everything(client):
    client.post("/behaviors", json=body("a"))
    client.post("/behaviors", json=body("b", detect=("other",), instruction_id="i2"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING, OTHER))
    assert client.delete("/behaviors").status_code == 202
    client.rig.settle()
    client.rig.frames(2, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 0


# 3. Models ---------------------------------------------------------------
def test_models_lists_availability_and_vocabulary(client):
    d = client.get("/models").json()
    by_name = {m["name"]: m for m in d["models"]}
    assert by_name["fake"]["available"] is True
    assert by_name["gpu_only"]["available"] is False
    assert "reason" in by_name["gpu_only"] and d["device"]


def test_the_model_can_be_swapped(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert client.post("/model", json={"model": "other"}).status_code == 202
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    assert client.get("/health").json()["model"] == "other"


def test_a_model_swap_keeps_the_behaviors(client):
    """"switch to RF-DETR" changes how we see, not what we are looking for."""
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    client.post("/model", json={"model": "other"})
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    assert set(client.rig.loop.behaviors) == {"b1"}
    assert client.rig.status("b1").state == "active"


def test_an_unknown_model_names_the_alternatives(client):
    r = client.post("/model", json={"model": "rfdetr"})
    assert r.status_code == 422 and "fake" in r.json()["detail"]


def test_a_model_this_machine_cannot_run_is_refused(client):
    r = client.post("/model", json={"model": "gpu_only"})
    assert r.status_code == 422 and "unavailable" in r.json()["detail"]


# 4. Introspection --------------------------------------------------------
def test_health_before_anything_is_running(client):
    h = client.get("/health").json()
    assert h["behaviors"] == 0 and h["phase"] in ("IDLE", "BOOTING")


def test_health_reports_what_is_running(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    h = client.get("/health").json()
    assert h["behaviors"] == 1 and h["tracks"] == 1 and h["frames"] > 0
    assert h["attributes"] is True


def test_health_still_answers_when_the_camera_is_dead(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    client.rig.capture.dead = True
    for _ in range(40):
        client.rig.now += 0.1
        client.rig.loop.step(client.rig.now)
    assert client.get("/health").json()["phase"] == "FAULT"


def test_behaviors_endpoint_advertises_what_is_not_built_yet(client):
    """So the agent knows the menu without guessing."""
    kinds = client.get("/behaviors").json()["kinds"]
    assert "highlight" in kinds["available"] and "track" in kinds["available"]
    assert "privacy" in kinds["planned"]


# 5. Live channels --------------------------------------------------------
def test_state_socket_carries_tracks_and_behaviors(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    with client.websocket_connect("/ws/state") as ws:
        client.rig.frames(5, seen=boxes(THING))
        msg = json.loads(ws.receive_text())
    assert msg["phase"] and "tracks" in msg and "behaviors" in msg


def test_published_tracks_are_normalized(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    with client.websocket_connect("/ws/state") as ws:
        for _ in range(3):
            client.rig.frames(2, seen=boxes(THING))
            msg = json.loads(ws.receive_text())
    track = msg["tracks"][0]
    assert -1 <= track["cx"] <= 1 and -1 <= track["cy"] <= 1 and 0 < track["area"] < 1


def test_events_replay_to_a_late_client(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    with client.websocket_connect("/ws/events") as ws:
        first = json.loads(ws.receive_text())
    assert "stage" in first or "kind" in first


def test_a_behavior_firing_reaches_the_event_channel(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    kinds = [e.kind for e in client.rig.fired]
    assert "acquired" in kinds


def test_a_stalled_client_makes_the_pipeline_drop_not_block(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    with client.websocket_connect("/ws/state") as ws:
        for _ in range(40):
            client.rig.frames(2, seen=boxes(THING))
        time.sleep(0.2)
        assert client.rig.loop.state_bus.dropped > 0
        assert json.loads(ws.receive_text())["phase"]


# 6. Video ----------------------------------------------------------------
def test_frame_endpoint_returns_a_jpeg(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and len(r.content) > 500


def test_video_renders_before_any_frame_arrives(client):
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_one_encode_is_shared_by_every_viewer(client):
    client.post("/behaviors", json=body("b1"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    before = client.app.state.streamer.encoded
    for _ in range(5):
        client.get("/frame.jpg")
    assert client.app.state.streamer.encoded - before <= 1


def test_the_debug_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and "perception" in r.text
