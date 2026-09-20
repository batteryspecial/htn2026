"""The seams other people's code touches.

Two callers exist that this service does not control: the operator console and
the agent. Both now speak `BehaviorSpec` natively — the TaskSpec shim that used
to sit in front of `/behaviors` is gone, along with `POST /spec` and
`WS /ws/status`. These tests pin the edges both callers land on, so a change
here fails in CI rather than on stage.
"""

import pytest
from fastapi.testclient import TestClient

from server.api import Service, create_app
from tests.rig import Rig, boxes

THING = ("thing", 320, 240, 100)


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
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.rig = rig
        yield c


def track(detect=("thing",), label="the thing", pick="largest"):
    return {"kind": "track",
            "subject": {"detect": list(detect), "pick": pick},
            "render": {"label": label}}


# 1. The browser can reach us at all -------------------------------------
def test_cross_origin_requests_are_allowed(client):
    """The UI is a page on another origin. Without this every call it makes
    fails before reaching a route, and the failure looks like "server down"."""
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_a_preflight_is_answered(client):
    r = client.options("/behaviors", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code < 400
    assert r.headers.get("access-control-allow-origin") == "*"


# 2. The retired shim -----------------------------------------------------
@pytest.mark.parametrize("method,path", [
    ("post", "/spec"),
    ("delete", "/spec"),
])
def test_the_taskspec_routes_are_gone(client, method, path):
    """`server/legacy.py` mapped every target to one `track`, which put six of
    the eight behaviour kinds out of reach of the only UI. Both callers compile
    to `BehaviorSpec` now, so the shim is deleted rather than deprecated — a
    second intake shape is a second thing to keep in step."""
    assert client.request(method.upper(), path).status_code in (404, 405)


def test_the_legacy_stage_socket_is_gone(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, RuntimeError)):
        with client.websocket_connect("/ws/status"):
            pass


# 3. Applying an objective ------------------------------------------------
def test_posting_a_behavior_starts_tracking(client):
    r = client.post("/behaviors", json=track())
    assert r.status_code == 201
    assert r.json()["id"]
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 1


def test_clearing_then_posting_replaces_the_objective(client):
    """An instruction means "this is the whole objective now". The console
    clears before it posts, so the second instruction does not stack on the
    first. This is the sequence `PerceptionClient.applyProgram` performs."""
    client.post("/behaviors", json=track(label="a"))
    client.rig.settle()

    client.delete("/behaviors")
    client.post("/behaviors", json=track(detect=("other",), label="b"))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))

    behaviors = client.get("/state").json()["behaviors"]
    assert len(behaviors) == 1 and behaviors[0]["label"] == "b"


def test_one_instruction_can_install_several_behaviors(client):
    """"Guard the table: laptop, phone, wallet" is three behaviours, each with
    its own subject and its own alert. The old contract capped it at two
    targets and collapsed them all to `track`."""
    for label in ("laptop", "phone", "wallet"):
        r = client.post("/behaviors", json={
            "kind": "watch",
            "subject": {"detect": [label], "pick": "largest"},
            "params": {"triggers": [{"type": "missing", "after_s": 2.0}]},
            "render": {"label": label},
        })
        assert r.status_code == 201

    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    assert len(client.get("/state").json()["behaviors"]) == 3


def test_a_class_the_model_cannot_see_is_refused_with_a_reason(client, monkeypatch):
    det = client.rig.registry.get("fake")
    monkeypatch.setattr(type(det), "classes", property(lambda self: ["person"]))

    # The vocabulary is read off the resident detector, so the loop has to have
    # built its world before the check has anything to consult.
    client.post("/behaviors", json=track(detect=("person",)))
    client.rig.settle()

    r = client.post("/behaviors", json=track(detect=("duck",)))
    assert r.status_code == 422 and "duck" in r.json()["detail"]


def test_a_malformed_behavior_is_refused_not_crashed(client):
    assert client.post("/behaviors", json={"kind": "track"}).status_code == 422
    assert client.post("/behaviors", json={"subject": {"detect": []}}).status_code == 422


# 4. Events the console draws on ------------------------------------------
def test_acquiring_a_target_reaches_the_event_stream(client):
    """The console's headline number is instruction sent -> `acquired`, so
    this event arriving is the metric the project is measured on."""
    with client.websocket_connect("/ws/events") as ws:
        client.post("/behaviors", json=track(pick="ref"))
        client.rig.settle()
        client.rig.frames(6, seen=boxes(THING))

        seen = set()
        for _ in range(12):
            seen.add(ws.receive_json()["type"])
            if "acquired" in seen:
                break
    assert "acquired" in seen


# 5. What the agent needs to discover -------------------------------------
def test_the_agent_can_learn_what_kinds_exist(client):
    kinds = client.get("/behaviors").json()["kinds"]
    assert set(kinds["available"]) >= {"highlight", "track", "watch",
                                       "count_line", "privacy", "pan_to",
                                       "pose_trigger", "keyboard"}
    assert kinds["planned"] == []


def test_the_agent_can_learn_the_vocabulary(client):
    d = client.get("/models").json()
    assert d["device"] and isinstance(d["models"], list)
    assert all("open_vocab" in m and "available" in m for m in d["models"])


def test_every_refusal_carries_a_reason_the_agent_can_act_on(client):
    """"invalid" is useless to a model that has to retry."""
    for payload, expect in [
        ({"kind": "keyboard", "subject": {"detect": ["x"]}}, "text"),
        ({"kind": "watch", "subject": {"detect": ["x"]}, "params": {"triggers": []}},
         "triggers"),
    ]:
        r = client.post("/behaviors", json=payload)
        assert r.status_code == 422
        assert expect in r.json()["detail"].lower() or expect in r.json()["detail"]


# 6. "all" — the request the old contract could not express ----------------
def test_highlight_reports_every_match(client):
    """"Detect all humans" was structurally inexpressible through the shim:
    every select rule picked exactly one, and every target became a `track`,
    which follows one thing by definition. It looked like the detector had
    failed. A `highlight` with `pick: all` is the whole answer."""
    r = client.post("/behaviors", json={
        "kind": "highlight",
        "subject": {"detect": ["thing"], "pick": "all"},
        "render": {"label": "all things"},
    })
    assert r.status_code == 201

    client.rig.settle()
    client.rig.frames(4, seen=boxes(("thing", 200, 200, 60), ("thing", 450, 300, 60)))

    view = client.get("/state").json()["behaviors"][0]
    assert view["kind"] == "highlight" and view["matches"] == 2
