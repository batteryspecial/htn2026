"""The seams other people's code touches.

Two callers exist that this service does not control: the operator UI, built
against the older TaskSpec contract, and the agent, being finished now. These
tests pin the edges both of them land on, so a change here fails in CI rather
than on stage.
"""

import json

import pytest
from fastapi.testclient import TestClient

from server.api import Service, create_app
from server.legacy import to_behaviors, to_stage
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


def taskspec(**over):
    spec = {
        "spec_id": "s1",
        "mode": "follow",
        "model": "yoloe",
        "targets": [{"ref": "the thing", "detect": ["thing"], "select": "largest"}],
    }
    spec.update(over)
    return spec


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


# 2. Legacy TaskSpec translation -----------------------------------------
def test_a_taskspec_becomes_a_tracking_behavior():
    behaviors, notes = to_behaviors(taskspec())
    assert len(behaviors) == 1
    assert behaviors[0].kind == "track"
    assert behaviors[0].subject.detect == ["thing"]
    assert behaviors[0].render.label == "the thing"


def test_two_targets_become_two_behaviors():
    behaviors, _ = to_behaviors(taskspec(targets=[
        {"ref": "pencil", "detect": ["pencil"]},
        {"ref": "eraser", "detect": ["eraser"]},
    ]))
    assert [b.subject.detect for b in behaviors] == [["pencil"], ["eraser"]]


def test_verify_becomes_an_include_and_says_so():
    behaviors, notes = to_behaviors(taskspec(targets=[{
        "ref": "red shoes", "detect": ["person"],
        "verify": {"class": "person", "text": "a person in red shoes", "min_score": 0.3},
    }]))
    assert behaviors[0].subject.include == ["a person in red shoes"]
    assert behaviors[0].subject.min_score == 0.3
    assert any("exclude" in n for n in notes), "the weaker regime should be flagged"


def test_relate_survives_the_rename():
    behaviors, _ = to_behaviors(taskspec(targets=[{
        "ref": "person with shoe", "detect": ["person", "shoe"],
        "relate": {"keep": "person", "if_contains": "shoe"},
    }]))
    assert behaviors[0].subject.relate.contains == "shoe"


def test_locked_becomes_a_reference_pick():
    behaviors, _ = to_behaviors(taskspec(targets=[
        {"ref": "him", "detect": ["person"], "select": "locked"}]))
    assert behaviors[0].subject.pick == "ref"


def test_an_unmappable_select_degrades_loudly():
    """`highest_conf` has no equivalent. Silently picking something else is
    how a demo does the wrong thing and nobody knows why."""
    behaviors, notes = to_behaviors(taskspec(targets=[
        {"ref": "x", "detect": ["thing"], "select": "highest_conf"}]))
    assert behaviors[0].subject.pick == "largest"
    assert any("highest_conf" in n for n in notes)


@pytest.mark.parametrize("bad", [{}, {"targets": []}, {"targets": [{"ref": "x"}]}])
def test_a_broken_taskspec_raises_rather_than_half_translating(bad):
    with pytest.raises(ValueError):
        to_behaviors(bad)


# 3. The legacy endpoint end to end ---------------------------------------
def test_posting_a_taskspec_starts_tracking(client):
    r = client.post("/spec", json={"instruction_id": "i1", "spec": taskspec()})
    assert r.status_code == 202
    assert r.json()["behavior_ids"]
    client.rig.settle()
    client.rig.frames(5, seen=boxes(THING))
    assert client.get("/health").json()["behaviors"] == 1


def test_a_bare_taskspec_without_the_envelope_also_works(client):
    """The compiler emits the spec itself; the orchestrator wraps it. Accept
    both rather than making the caller guess."""
    assert client.post("/spec", json=taskspec()).status_code == 202


def test_a_new_taskspec_replaces_the_old_objective(client):
    """The old contract meant "this is the whole objective now"."""
    client.post("/spec", json=taskspec(targets=[{"ref": "a", "detect": ["thing"]}]))
    client.rig.settle()
    client.post("/spec", json=taskspec(spec_id="s2",
                                       targets=[{"ref": "b", "detect": ["other"]}]))
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    behaviors = client.get("/state").json()["behaviors"]
    assert len(behaviors) == 1 and behaviors[0]["label"] == "b"


def test_a_taskspec_the_model_cannot_see_is_refused_with_a_reason(client, monkeypatch):
    det = client.rig.registry.get("fake")
    monkeypatch.setattr(type(det), "classes", property(lambda self: ["person"]))
    client.post("/spec", json=taskspec())
    client.rig.settle()
    r = client.post("/spec", json=taskspec(targets=[{"ref": "x", "detect": ["duck"]}]))
    assert r.status_code == 422 and "duck" in r.json()["detail"]


def test_a_malformed_taskspec_is_refused_not_crashed(client):
    r = client.post("/spec", json={"spec": {"targets": []}})
    assert r.status_code == 422 and "TaskSpec" in r.json()["detail"]


# 4. Legacy stage stream ---------------------------------------------------
def test_acquired_appears_as_the_active_stage(client):
    client.post("/spec", json={"instruction_id": "i1", "spec": taskspec(
        targets=[{"ref": "x", "detect": ["thing"], "select": "locked"}])})
    client.rig.settle()
    with client.websocket_connect("/ws/status") as ws:
        client.rig.frames(6, seen=boxes(THING))
        msg = json.loads(ws.receive_text())
    assert msg["stage"] == "active" and "ts" in msg


def test_events_with_no_old_equivalent_are_dropped():
    """Inventing a stage name would put unknown entries on the UI's strip."""
    from contracts import Event

    assert to_stage(Event(id="e", type="count_changed")) is None
    assert to_stage(Event(id="e", type="crossed")) is None
    assert to_stage(Event(id="e", type="acquired"))["stage"] == "active"


# 5. What the agent needs to discover -------------------------------------
def test_the_agent_can_learn_what_kinds_exist(client):
    kinds = client.get("/behaviors").json()["kinds"]
    assert set(kinds["available"]) >= {"highlight", "track", "watch",
                                       "count_line", "privacy", "pan_to",
                                       "pose_trigger"}
    assert set(kinds["planned"]) >= {"keyboard"}


def test_the_agent_can_learn_the_vocabulary(client):
    d = client.get("/models").json()
    assert d["device"] and isinstance(d["models"], list)
    assert all("open_vocab" in m and "available" in m for m in d["models"])


def test_every_refusal_carries_a_reason_the_agent_can_act_on(client):
    """"invalid" is useless to a model that has to retry."""
    for payload, expect in [
        ({"kind": "keyboard", "subject": {"detect": ["x"]}}, "not implemented"),
        ({"kind": "watch", "subject": {"detect": ["x"]}, "params": {"triggers": []}},
         "triggers"),
    ]:
        r = client.post("/behaviors", json=payload)
        assert r.status_code == 422
        assert expect in r.json()["detail"].lower() or expect in r.json()["detail"]
