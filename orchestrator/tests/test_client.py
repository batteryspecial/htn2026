"""The HTTP client's contract with the rest of the orchestrator.

One rule holds everywhere in `tools/perception.py`: **never raise at the
agent**. A tool that throws ends the turn; a tool that returns
`{"ok": false, "error": "..."}` lets the agent read the reason and try
something else. Most of this file is that rule, checked against each way a
call can go wrong.

The second rule — HTTP is HTTP — shows up here as an absence: nothing in this
file opens a socket. The event stream is tested in `test_events.py`.
"""

from __future__ import annotations

import httpx
import pytest

from tests.fake_pipeline import FakePipeline
from tools.perception import Perception

SPEC = {
    "kind": "highlight",
    "subject": {"detect": ["person"]},
    "render": {"label": "everyone"},
}


# 1. The happy path ---------------------------------------------------------

async def test_a_started_behavior_comes_back_with_its_id(perception, pipeline):
    result = await perception.start_behavior(SPEC)

    assert result["ok"] is True
    assert result["id"] == "b1"
    assert pipeline.behaviors["b1"]["kind"] == "highlight"


async def test_the_listing_carries_what_is_running(perception, pipeline):
    await perception.start_behavior(SPEC)

    result = await perception.list_behaviors()

    assert result["ok"] is True
    assert [b["id"] for b in result["behaviors"]] == ["b1"]
    # The agent reads `kinds` to know what it must not ask for.
    assert "keyboard" in result["kinds"]["planned"]


async def test_a_202_with_a_body_is_still_ok(perception):
    """DELETE /behaviors answers 202 {"cleared": true}, not 204."""
    result = await perception.clear_behaviors()

    assert result == {"ok": True, "cleared": True}


async def test_an_empty_body_is_ok_rather_than_a_parse_failure():
    """A 204 carries nothing to decode, and that is a success."""
    async def no_content(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    client = Perception(transport=httpx.MockTransport(no_content))
    try:
        assert await client.set_hud("watching the duck") == {"ok": True}
    finally:
        await client.aclose()


# 2. Every way it can fail --------------------------------------------------

async def test_a_422_surfaces_the_sentence_perception_wrote(perception):
    """The whole reason perception's refusals are specific."""
    result = await perception.start_behavior({**SPEC, "kind": "keyboard"})

    assert result["ok"] is False
    assert result["status"] == 422
    assert "not implemented yet" in result["error"]
    assert "keyboard" in result["error"]


async def test_a_param_refusal_names_the_missing_param(perception):
    result = await perception.start_behavior(
        {"kind": "watch", "subject": {"detect": ["duck"]}})

    assert result["ok"] is False
    assert result["error"] == "watch needs a non-empty 'triggers' list"


async def test_a_404_on_an_unknown_behavior_is_reported_not_swallowed(perception):
    result = await perception.stop_behavior("b99")

    assert result["ok"] is False
    assert "b99" in result["error"]


async def test_a_dead_pipeline_is_an_error_string_not_an_exception(perception, pipeline):
    pipeline.down = True

    result = await perception.list_behaviors()

    assert result["ok"] is False
    assert "cannot reach perception" in result["error"]
    assert perception.base in result["error"]


async def test_a_timeout_says_so_in_seconds(pipeline):
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = Perception(transport=httpx.MockTransport(slow))
    try:
        result = await client.probe(["a yellow duck"])
    finally:
        await client.aclose()

    assert result["ok"] is False
    assert "did not answer /probe" in result["error"]


async def test_a_non_json_error_body_still_produces_a_reason():
    """A proxy or a crashed worker answers HTML, not `{"detail": ...}`."""
    def html_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<h1>Internal Server Error</h1>")

    client = Perception(transport=httpx.MockTransport(html_500))
    try:
        result = await client.health()
    finally:
        await client.aclose()

    assert result["ok"] is False
    assert "Internal Server Error" in result["error"]


@pytest.mark.parametrize("body,expected", [
    ({"detail": "label 'duck' is not in the coco vocabulary"},
     "label 'duck' is not in the coco vocabulary"),
    ({"message": "something else"}, "something else"),
    ({"detail": [{"loc": ["body", "kind"], "msg": "field required"}]},
     "field required"),
])
async def test_the_most_specific_message_in_the_body_wins(pipeline, body, expected):
    pipeline.overrides[("POST", "/behaviors")] = (422, body)
    client = Perception(transport=pipeline.transport())
    try:
        result = await client.start_behavior(SPEC)
    finally:
        await client.aclose()

    assert expected in result["error"]


# 3. Request shapes ---------------------------------------------------------

async def test_probe_is_capped_at_six_phrases(perception, pipeline):
    """ProbeRequest rejects more than six; the client trims rather than 422s."""
    await perception.probe([f"phrase {i}" for i in range(10)])

    sent = pipeline.last("POST", "/probe")
    assert len(sent["json"]["phrases"]) == 6


async def test_count_sends_the_selector_and_the_window(perception, pipeline):
    await perception.count({"detect": ["person"]}, window_s=2.5)

    sent = pipeline.last("POST", "/query/count")
    assert sent["json"] == {"selector": {"detect": ["person"]}, "window_s": 2.5}


async def test_look_with_no_selector_asks_for_everything(perception, pipeline):
    await perception.look()

    assert pipeline.last("POST", "/query/look")["json"] == {"selector": None}


async def test_a_reference_from_a_photo_is_multipart(perception, pipeline):
    result = await perception.add_reference_from_image(b"\xff\xd8jpeg", "duck.jpg")

    assert result["ok"] is True
    assert result["ref_id"] == "r1"
    # Multipart, so the handler saw no JSON body — that is the point.
    assert pipeline.last("POST", "/references")["json"] is None


async def test_a_reference_from_the_frame_is_json(perception, pipeline):
    await perception.add_reference_from_frame("largest_person", label="him")

    assert pipeline.last("POST", "/references")["json"] == {
        "from": "largest_person", "label": "him"}


async def test_absolute_leaves_a_full_url_alone(perception):
    assert perception.absolute("/snapshots/e1.jpg").endswith("/snapshots/e1.jpg")
    assert perception.absolute("http://elsewhere/x.jpg") == "http://elsewhere/x.jpg"


async def test_a_snapshot_that_is_missing_returns_none_rather_than_raising(pipeline):
    pipeline.overrides[("GET", "/snapshot")] = (404, {"detail": "no frame"})
    client = Perception(transport=pipeline.transport())
    try:
        assert await client.snapshot() is None
    finally:
        await client.aclose()
