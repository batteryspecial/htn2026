"""The HTTP and WebSocket surface.

The rule this file exists to hold: **HTTP is HTTP, WebSockets are
WebSockets.** `/chat` is a request that returns a reply. `/ws/trace` is a
stream the browser subscribes to. A turn's intermediate steps go on the
socket and are never dribbled into the HTTP response, and neither endpoint
answers on the other's protocol.

The second rule is that nothing here reasons: routes marshal input, hand it
to the runner, and marshal output.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from tests.fake_model import calls, says
from tools.perception import Perception


@pytest.fixture
def client(pipeline, tmp_path, monkeypatch):
    """The real app, with perception in-process and the watcher off.

    `watch_events=False` because the event stream is a WebSocket out to the
    pipeline; it is tested against its own fake in `test_events.py`, and a
    reconnect loop chattering in the background would make these flaky.
    """
    from memory.store import PhraseMemory

    monkeypatch.setattr("app.api.Perception",
                        lambda: Perception(transport=pipeline.transport()))
    monkeypatch.setattr("app.api.PhraseMemory",
                        lambda: PhraseMemory(uri=str(tmp_path / "db"), embed_model=""))

    with TestClient(create_app(watch_events=False)) as test_client:
        yield test_client


@pytest.fixture
def model(monkeypatch):
    """Script the model this app's turns will run on."""
    def install(scripted):
        monkeypatch.setattr("agent.graph.chat_model", lambda: scripted)
        monkeypatch.setattr("agent.graph.vision_model", lambda: scripted)
        return scripted
    return install


# 1. Status -----------------------------------------------------------------

def test_health_reports_the_model_and_whether_a_key_is_present(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["provider"] == "openai"
    # conftest blanks the key, so this is the "it will boot and say so" case.
    assert body["has_key"] is False


def test_health_reports_the_pipeline_it_is_pointed_at(client):
    body = client.get("/health").json()

    assert body["perception"]["base"] == "http://perception.test"
    assert body["perception"]["up"] is True


def test_a_pipeline_that_is_down_is_reported_not_a_500(client, pipeline):
    """Perception being unreachable is news for the operator, not a crash."""
    pipeline.down = True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["perception"]["up"] is False
    assert "cannot reach perception" in response.json()["perception"]["detail"]


def test_the_memory_can_be_inspected_for_tuning(client):
    body = client.get("/memory", params={"subject": "yellow rubber duck"}).json()

    assert body["subject"] == "yellow rubber duck"
    assert any(s["phrase"] == "yellow duck" for s in body["suggestions"])


# 2. Turns ------------------------------------------------------------------

def test_a_chat_turn_returns_the_reply_and_how_long_it_took(client, model):
    model(says("Hello."))

    body = client.post("/chat", data={"text": "hello"}).json()

    assert body["reply"] == "Hello."
    assert body["turn"].startswith("turn-")
    assert isinstance(body["seconds"], float)
    assert body["outcome"] == "reply"
    assert body["behaviors"] == []


def test_a_chat_turn_actually_configures_the_pipeline(client, model, pipeline):
    model(calls(("start_behavior",
                 {"kind": "highlight", "subject": {"detect": ["person"]}}))
          .then(says("Done.")))

    body = client.post("/chat", data={"text": "highlight everyone"}).json()

    assert pipeline.behaviors["b1"]["kind"] == "highlight"
    assert body["outcome"] == "applied"
    assert body["behaviors"][0]["id"] == "b1"
    assert body["behaviors"][0]["spec"]["kind"] == "highlight"
    assert body["behaviors"][0]["spec"]["subject"]["detect"] == ["person"]


def test_an_empty_turn_is_refused_rather_than_sent_to_the_model(client, model):
    scripted = model(says("should not be called"))

    response = client.post("/chat", data={"text": "   "})

    assert response.status_code == 422
    assert "say something" in response.json()["detail"]
    assert scripted.seen == []


def test_a_photo_with_no_words_is_a_turn(client, model, pipeline):
    """"Track this" is sometimes just a picture."""
    model(says("Following him."))

    response = client.post("/chat", data={"text": ""},
                           files={"images": ("him.jpg", b"\xff\xd8jpeg", "image/jpeg")})

    assert response.status_code == 200
    assert "/references" in pipeline.paths("POST")


def test_the_console_contract_still_works(client, model):
    """`/instruction` is what LIVE mode shipped against; it must not break."""
    model(calls(("start_behavior", {
        "kind": "track", "subject": {"detect": ["duck"]},
    })).then(says("Tracking the duck.")))

    body = client.post("/instruction", json={"text": "track the duck"}).json()

    assert body["reply"] == "Tracking the duck."
    assert body["instruction_id"].startswith("turn-")


def test_an_empty_instruction_is_refused(client):
    response = client.post("/instruction", json={"text": ""})

    assert response.status_code == 422


# 3. Streams ----------------------------------------------------------------

def test_the_trace_replays_what_a_late_browser_missed(client):
    """A console opened mid-demo should not show an empty panel."""
    from app.trace import BUS

    BUS.emit("user", "track the duck", turn="turn-1")
    BUS.emit("reply", "Tracking the duck.", turn="turn-1")

    with client.websocket_connect("/ws/trace") as socket:
        first = socket.receive_json()
        second = socket.receive_json()

    assert (first["kind"], first["label"]) == ("user", "track the duck")
    assert (second["kind"], second["label"]) == ("reply", "Tracking the duck.")
    assert first["turn"] == "turn-1"


def test_the_trace_carries_the_structured_data_with_the_entry(client):
    from app.trace import BUS

    BUS.emit("tool_result", "start_behavior", "b1 · highlight",
             turn="turn-1", behavior_id="b1")

    with client.websocket_connect("/ws/trace") as socket:
        entry = socket.receive_json()

    assert entry["data"]["behavior_id"] == "b1"
    assert entry["id"].startswith("t")


def test_the_status_strip_only_carries_stages_it_understands(client, model):
    """Inventing a stage name for a tool call would put junk on the strip."""
    model(calls(("start_behavior",
                 {"kind": "highlight", "subject": {"detect": ["person"]}}))
          .then(says("Highlighting everyone.")))

    with client.websocket_connect("/ws/status") as socket:
        client.post("/chat", data={"text": "highlight everyone"})

        stages = [socket.receive_json() for _ in range(2)]

    assert [s["stage"] for s in stages] == ["received", "applied"]
    assert all(s["instruction_id"].startswith("turn-") for s in stages)


def test_a_failure_reaches_the_status_strip_as_failed(client, model):
    from tests.fake_model import ScriptedModel

    model(ScriptedModel(error=RuntimeError("503 from the provider")))

    with client.websocket_connect("/ws/status") as socket:
        client.post("/chat", data={"text": "track the duck"})

        assert socket.receive_json()["stage"] == "received"
        assert socket.receive_json()["stage"] == "failed"


# 4. Protocol discipline ----------------------------------------------------

def test_a_websocket_route_does_not_answer_http(client):
    """Shockingly common mistake; worth a test that will notice."""
    assert client.get("/ws/trace").status_code in (404, 405, 426)


def test_an_http_route_does_not_answer_a_websocket(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, AssertionError, RuntimeError)):
        with client.websocket_connect("/chat"):
            pass


def test_the_console_on_another_origin_can_reach_the_api(client):
    """Without CORS every call fails before reaching a route, and the failure
    looks like "server down" rather than "CORS"."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "*"
