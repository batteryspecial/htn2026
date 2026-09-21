"""The service as the agent meets it: over HTTP.

The orchestrator is written against this, so these double as its contract.
Nothing here reaches past what a network client can see, except to fake the
camera.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

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
                  loop=rig.loop, health=rig.health, events=rig.event_bus,
                  states=rig.state_bus)
    # run_threads=False: frames are driven by hand so nothing is timing
    # dependent, but the app, the routes and the buses are all real.
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.rig = rig
        yield c


def spec(kind="highlight", detect=("thing",), **sub):
    return {"kind": kind, "subject": {"detect": list(detect), **sub}}


# 1. Starting behaviours --------------------------------------------------
def test_the_server_assigns_the_id(client):
    r = client.post("/behaviors", json=spec())
    assert r.status_code == 201 and r.json()["id"]


def test_two_behaviors_get_different_ids(client):
    a = client.post("/behaviors", json=spec()).json()["id"]
    b = client.post("/behaviors", json=spec(detect=("other",))).json()["id"]
    assert a != b


def test_creating_does_not_wait_for_the_pipeline(client):
    """The id comes back at once; the agent must never block on us."""
    client.post("/behaviors", json=spec())
    assert client.rig.loop.behaviors == {}


def test_a_posted_behavior_starts_running(client):
    bid = client.post("/behaviors", json=spec()).json()["id"]
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 1
    assert client.rig.state_of(bid) == "ACTIVE"


def test_behavior_status_confirms_frame_loop_installation(client):
    bid = client.post("/behaviors", json=spec()).json()["id"]
    client.rig.settle()

    status = client.get(f"/behaviors/{bid}/status").json()

    assert status["id"] == bid
    assert status["status"] == "installed"


@pytest.mark.parametrize("bad", [
    {},
    {"kind": "highlight"},
    spec(detect=()),
    {"kind": "nonsense", "subject": {"detect": ["a"]}},
    {"kind": "highlight", "subject": {"detect": ["a"]}, "junk": 1},
    {"kind": "highlight", "subject": {"detect": ["a"], "pick": "vibes"}},
])
def test_a_malformed_behavior_is_rejected(client, bad):
    assert client.post("/behaviors", json=bad).status_code == 422


def test_every_kind_in_the_contract_is_implemented(client):
    """`keyboard` was the last one. Nothing in the contract is a promise the
    service cannot keep any more, so `planned` is empty."""
    kinds = client.get("/behaviors").json()["kinds"]
    assert kinds["planned"] == []
    assert "keyboard" in kinds["available"]


def test_a_rejected_behavior_never_starts(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.post("/behaviors", json=spec(kind="keyboard"))  # no 'text': refused
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert len(client.rig.loop.behaviors) == 1


# 2. Stopping -------------------------------------------------------------
def test_a_behavior_can_be_stopped(client):
    bid = client.post("/behaviors", json=spec()).json()["id"]
    client.rig.settle()
    assert client.delete(f"/behaviors/{bid}").status_code == 202
    client.rig.settle()
    assert client.get("/health").json()["behaviors"] == 0


def test_stopping_one_leaves_the_others(client):
    a = client.post("/behaviors", json=spec(detect=("thing",))).json()["id"]
    b = client.post("/behaviors", json=spec(detect=("other",))).json()["id"]
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING, OTHER))
    client.delete(f"/behaviors/{b}")
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING, OTHER))
    assert set(client.rig.loop.behaviors) == {a}


def test_stopping_something_that_is_not_running_is_a_404(client):
    assert client.delete("/behaviors/ghost").status_code == 404


def test_clear_stops_everything(client):
    client.post("/behaviors", json=spec())
    client.post("/behaviors", json=spec(detect=("other",)))
    client.rig.settle()
    assert client.delete("/behaviors").status_code == 202
    client.rig.settle()
    assert client.get("/health").json()["behaviors"] == 0


def test_listing_shows_what_is_running_and_what_is_planned(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    d = client.get("/behaviors").json()
    assert len(d["behaviors"]) == 1
    assert "highlight" in d["kinds"]["available"]
    assert d["kinds"]["planned"] == []


# 3. Model ----------------------------------------------------------------
def test_models_lists_availability(client):
    d = client.get("/models").json()
    by_name = {m["name"]: m for m in d["models"]}
    assert by_name["fake"]["available"] is True
    assert by_name["gpu_only"]["available"] is False
    assert "reason" in by_name["gpu_only"] and d["device"]


def test_models_reports_which_gestures_exist(client, monkeypatch):
    """The agent has to tell "no gestures at all" from "not that gesture".

    It cannot get this from the kind catalogue, which only says `pose_trigger`
    exists. Asked to report a raised *index finger*, the compiler then has to
    fill in a `gesture` param and settles for the nearest implemented value —
    the spec validates, the behaviour arms, and it never fires because the
    rule it actually installed watches wrists. Reporting the vocabulary is
    what lets it say "I only have hand_raised" instead.
    """
    # The rig loads no aux model at all, so no gesture can be observed.
    assert client.get("/models").json()["gestures"] == []

    # The list spans roles and narrows to what is loaded. With only a body
    # model, a finger gesture is still not on offer — reporting it would have
    # the agent install a behaviour that can never fire.
    monkeypatch.setattr(client.rig.registry, "has_role", lambda role: role == "pose")
    body_only = client.get("/models").json()["gestures"]
    assert "hand_raised" in body_only
    assert "index_finger_raised" not in body_only

    monkeypatch.setattr(client.rig.registry, "has_role", lambda role: role == "hands")
    hands_only = client.get("/models").json()["gestures"]
    assert "index_finger_raised" in hands_only
    assert "hand_raised" not in hands_only


def test_the_model_can_be_swapped(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    assert client.post("/model", json={"name": "other"}).status_code == 202
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert client.get("/health").json()["model"] == "other"


def test_a_model_swap_keeps_the_behaviors(client):
    """"switch to RF-DETR" changes how we see, not what we look for."""
    bid = client.post("/behaviors", json=spec()).json()["id"]
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    client.post("/model", json={"name": "other"})
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    assert set(client.rig.loop.behaviors) == {bid}
    assert client.rig.state_of(bid) == "ACTIVE"


def test_an_unknown_model_names_the_alternatives(client):
    r = client.post("/model", json={"name": "rfdetr"})
    assert r.status_code == 422 and "fake" in r.json()["detail"]


def test_a_model_this_machine_cannot_run_is_refused(client):
    r = client.post("/model", json={"name": "gpu_only"})
    assert r.status_code == 422 and "unavailable" in r.json()["detail"]


def test_a_vocabulary_miss_is_specific_enough_to_retry(client, monkeypatch):
    """The agent reads the reason and tries again, so "invalid" is useless."""
    client.post("/behaviors", json=spec())
    client.rig.settle()
    det = client.rig.registry.get("fake")
    monkeypatch.setattr(type(det), "classes", property(lambda self: ["person", "dog"]))
    r = client.post("/behaviors", json=spec(detect=("duck",)))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "duck" in detail and "vocabulary" in detail


# 4. Introspection --------------------------------------------------------
def test_health_is_small_and_honest(client):
    h = client.get("/health").json()
    assert set(h) >= {"status", "fps", "model", "camera_ok", "behaviors", "device"}


def test_health_reports_a_dead_camera(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    client.rig.capture.dead = True
    for _ in range(40):
        client.rig.now += 0.1
        client.rig.loop.step(client.rig.now)
    h = client.get("/health").json()
    assert h["status"] == "no_camera" and h["camera_ok"] is False


def test_a_dead_camera_is_announced_once(client):
    client.rig.frames(2, seen=boxes(THING))
    client.rig.capture.dead = True
    for _ in range(40):
        client.rig.now += 0.1
        client.rig.loop.step(client.rig.now)
    assert client.rig.types().count("camera_lost") == 1


def test_state_carries_tracks_and_behaviors(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    s = client.get("/state").json()
    assert s["tracks"] and s["behaviors"] and s["model"] == "fake"


def test_hud_text_round_trips(client):
    client.post("/hud", json={"text": "watch the duck"})
    client.rig.frames(2, seen=boxes(THING))
    assert client.get("/state").json()["hud"] == "watch the duck"


# 5. Queries --------------------------------------------------------------
def test_count_is_a_median_not_a_single_frame(client):
    """One bad frame must not change the answer the agent reports."""
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING, ("thing", 500, 100, 50)))
    r = client.post("/query/count",
                    json={"selector": {"detect": ["thing"]}, "window_s": 0.05})
    assert r.json()["count"] == 2


def test_look_returns_what_is_visible_with_attributes(client):
    client.rig.paint("red")
    client.post("/behaviors", json=spec(include=["red"]))
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    tracks = client.post("/query/look", json={}).json()["tracks"]
    assert tracks and "red" in tracks[0]["attributes"]


def test_look_can_be_narrowed(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING, OTHER))
    tracks = client.post("/query/look",
                         json={"selector": {"detect": ["other"]}}).json()["tracks"]
    assert [t["label"] for t in tracks] == ["other"]


def test_query_applies_pick_instead_of_only_filtering_labels(client):
    client.post("/behaviors", json=spec(detect=("person",)))
    client.rig.settle()
    client.rig.frames(4, seen=boxes(
        ("person", 180, 240, 60), ("person", 440, 240, 140)))

    tracks = client.post("/query/look", json={
        "selector": {"detect": ["person"], "pick": "largest"},
    }).json()["tracks"]

    assert len(tracks) == 1
    assert tracks[0]["area"] > 0.03


def test_query_rejects_an_unknown_reference_instead_of_returning_everyone(client):
    client.post("/behaviors", json=spec(detect=("person",)))
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))

    response = client.post("/query/look", json={
        "selector": {"detect": ["person"], "ref_id": "missing", "pick": "ref"},
    })

    assert response.status_code == 409
    assert "unknown reference" in response.json()["detail"]


def test_probe_without_a_frame_is_unavailable_not_a_zero_measurement(client):
    response = client.post("/probe", json={"phrases": ["person"]})

    assert response.status_code == 503
    assert "no frame" in response.json()["detail"]


# 6. Live channels --------------------------------------------------------
def test_state_socket_delivers_frames(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    with client.websocket_connect("/ws/state") as ws:
        client.rig.frames(4, seen=boxes(THING))
        msg = json.loads(ws.receive_text())
    assert "behaviors" in msg and "tracks" in msg


def test_events_reach_the_channel(client):
    client.post("/behaviors", json=spec(kind="track", pick="ref"))
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    with client.websocket_connect("/ws/events") as ws:
        first = json.loads(ws.receive_text())
    assert first["type"] and "id" in first and "notify" in first


def test_a_stalled_client_does_not_stall_the_pipeline(client):
    """A client that stops reading must not hold up the frame loop.

    The drop policy itself is pinned deterministically at the bus level in
    test_runtime; what matters here is that the socket survives a burst and
    the pipeline kept going while nobody was listening.
    """
    client.post("/behaviors", json=spec())
    client.rig.settle()
    with client.websocket_connect("/ws/state") as ws:
        before = client.rig.loop.timings.frames
        for _ in range(40):
            client.rig.frames(2, seen=boxes(THING))
        assert client.rig.loop.timings.frames - before == 80
        assert json.loads(ws.receive_text())["ts"]


# 7. Video ----------------------------------------------------------------
def test_the_enriched_frame_is_a_jpeg(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and len(r.content) > 500


def test_video_renders_before_any_frame_arrives(client):
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_the_snapshot_has_no_overlays(client):
    """The agent's vision model must see the world, not our annotations."""
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    enriched = client.get("/frame.jpg").content
    raw = client.get("/snapshot").content
    assert raw[:2] == b"\xff\xd8" and raw != enriched


def test_a_missing_event_snapshot_is_a_404(client):
    assert client.get("/snapshots/nope.jpg").status_code == 404


def test_one_encode_is_shared_by_every_viewer(client):
    client.post("/behaviors", json=spec())
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    before = client.app.state.streamer.encoded
    for _ in range(5):
        client.get("/frame.jpg")
    assert client.app.state.streamer.encoded - before <= 1


def test_the_debug_page_loads(client):
    assert client.get("/").status_code == 200


# 8. References -----------------------------------------------------------
def test_an_uploaded_reference_is_registered_and_matches_the_subject(client):
    import cv2
    import numpy as np

    image = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    assert encoded
    result = client.post("/references", files={
        "file": ("target.jpg", jpeg.tobytes(), "image/jpeg")},
        data={"label": "my red target"})
    assert result.status_code == 201
    ref_id = result.json()["ref_id"]
    bid = client.post("/behaviors", json=spec(ref_id=ref_id)).json()["id"]
    client.rig.paint("red", "blue")
    client.rig.settle()
    client.rig.frames(6, seen=boxes(
        ("thing", 160, 240, 100), ("thing", 480, 240, 100)))

    refs = client.get("/references").json()["references"]
    assert [(r["ref_id"], r["label"]) for r in refs] == [
        (ref_id, "my red target")]
    assert client.get(result.json()["thumb_url"]).content[:2] == b"\xff\xd8"
    assert client.rig.view(bid).matches == 1
    assert "behavior_failed" not in client.rig.types()


def test_a_frame_reference_preserves_its_vector_id_and_label(client):
    client.post("/behaviors", json=spec(detect=("person",)))
    client.rig.paint("red")
    client.rig.settle()
    client.rig.frames(4, seen=boxes(("person", 320, 240, 100)))
    result = client.post("/references", json={
        "from": "largest_person", "label": "the operator"})
    assert result.status_code == 201
    ref_id = result.json()["ref_id"]
    client.rig.settle()

    refs = client.get("/references").json()["references"]
    assert [(r["ref_id"], r["label"]) for r in refs] == [
        (ref_id, "the operator")]
    assert ref_id in client.rig.loop.world.ref_vectors
    assert "behavior_failed" not in client.rig.types()


def test_a_reference_that_cannot_be_encoded_is_rejected_before_201(tmp_path):
    rig = Rig(tmp_path, attributes=False)
    svc = Service(registry=rig.registry, capture=rig.capture, builder=rig.builder,
                  loop=rig.loop, health=rig.health, events=rig.event_bus,
                  states=rig.state_bus)
    try:
        with TestClient(create_app(svc, run_threads=False)) as client:
            response = client.post("/references", files={
                "file": ("target.jpg", b"not-an-image", "image/jpeg")})
            assert response.status_code == 422

        import cv2
        import numpy as np

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        encoded, jpeg = cv2.imencode(".jpg", image)
        assert encoded
        with TestClient(create_app(svc, run_threads=False)) as client:
            response = client.post("/references", files={
                "file": ("target.jpg", jpeg.tobytes(), "image/jpeg")})
            assert response.status_code == 422
            assert "CLIP" in response.json()["detail"]
            assert client.get("/references").json()["references"] == []
    finally:
        rig.close()
