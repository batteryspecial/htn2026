"""Who owns a turn, and what happens around it.

The runner is thin on purpose — build the messages, run the graph, publish
the result — so what is worth testing is the three things it does that the
graph cannot:

* an uploaded photo becomes a `ref_id` *before* the model thinks, so the
  agent is handed one rather than having to work out it needs one;
* turns are serialised, because a user turn and an event turn arriving
  together would both read "what is running" and race to replace it;
* a turn that blows up still answers the operator.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.runner import AgentRunner, Attachment
from tests.fake_model import ScriptedModel, calls, says

JPEG = b"\xff\xd8\xff\xe0jpeg-bytes"


@pytest.fixture
def runner(perception, memory):
    return AgentRunner(perception, memory)


@pytest.fixture
def scripted(monkeypatch):
    def install(model):
        monkeypatch.setattr("agent.graph.chat_model", lambda: model)
        return model
    return install


# 1. A plain turn -----------------------------------------------------------

async def test_a_turn_comes_back_with_an_id_a_reply_and_a_duration(runner, scripted):
    scripted(says("Highlighting everyone."))

    result = await runner.user_turn("highlight everyone")

    assert result.turn.startswith("turn-")
    assert result.reply == "Highlighting everyone."
    assert result.seconds >= 0.0


async def test_each_turn_gets_its_own_id(runner, scripted):
    scripted(says("ok"))

    first = await runner.user_turn("one")
    second = await runner.user_turn("two")

    assert first.turn != second.turn


async def test_what_the_operator_said_opens_the_trace(runner, scripted, trace):
    scripted(says("ok"))

    await runner.user_turn("watch the duck")

    opening = trace[0]
    assert opening.kind == "user"
    assert opening.label == "watch the duck"


# 2. Photos -----------------------------------------------------------------

async def test_an_uploaded_photo_is_registered_before_the_model_thinks(
        runner, scripted, pipeline):
    """"Track that one" is two steps, and this is the first."""
    model = scripted(says("Following him."))

    await runner.user_turn("track this person", [
        Attachment("him.jpg", "image/jpeg", JPEG)])

    # Registered first...
    assert pipeline.paths("POST")[0] == "/references"
    # ...and the id was handed over with instructions for using it.
    told = model.prompt_text()
    assert "registered as reference r1" in told
    assert 'pick "ref"' in told


async def test_the_photo_itself_is_given_to_the_model(runner, scripted):
    """A registered reference is not a substitute for seeing the picture."""
    model = scripted(says("Following him."))

    await runner.user_turn("track this one", [
        Attachment("him.jpg", "image/jpeg", JPEG)])

    human = next(m for m in model.seen[0] if not isinstance(m.content, str))
    assert any(p["type"] == "image_url" for p in human.content)


async def test_registering_a_reference_is_traced(runner, scripted, trace):
    scripted(says("ok"))

    await runner.user_turn("track this one", [
        Attachment("him.jpg", "image/jpeg", JPEG)])

    entry = next(e for e in trace if e.label == "add_reference")
    assert entry.data["ref_id"] == "r1"


async def test_a_photo_that_cannot_be_registered_does_not_stop_the_turn(
        runner, scripted, pipeline):
    pipeline.overrides[("POST", "/references")] = (
        422, {"detail": "could not decode that image"})
    model = scripted(says("I could not use that photo."))

    result = await runner.user_turn("track this one", [
        Attachment("broken.jpg", "image/jpeg", b"not-a-jpeg")])

    assert "could not decode that image" in model.prompt_text()
    assert "You can still see the image." in model.prompt_text()
    assert result.reply == "I could not use that photo."


async def test_the_trace_opens_with_the_operator_then_the_registration(
        runner, scripted, trace):
    """What was said, then what was done about it — in that order.

    Both carry the turn id, so the console can group them; the registration
    happens before the graph starts and would otherwise be an orphan sitting
    above the line that caused it.
    """
    scripted(says("ok"))

    await runner.user_turn("track this one", [
        Attachment("him.jpg", "image/jpeg", JPEG)])

    assert [(e.kind, e.label) for e in trace[:2]] == [
        ("user", "track this one"), ("tool_result", "add_reference")]
    assert trace[0].turn == trace[1].turn


# 3. Event turns ------------------------------------------------------------

async def test_an_event_turn_is_marked_as_one_and_carries_its_snapshot(
        runner, scripted, trace):
    model = scripted(says("Someone is reaching for the duck."))

    result = await runner.event_turn(
        "The pipeline reported: a person came near the duck.", JPEG)

    assert trace[0].kind == "event"
    assert "You were woken by the pipeline" in model.prompt_text()
    assert result.reply == "Someone is reaching for the duck."


async def test_an_event_with_no_snapshot_still_runs(runner, scripted):
    scripted(says("The duck is gone."))

    result = await runner.event_turn("The pipeline reported: missing.", None)

    assert result.reply == "The duck is gone."


# 4. Serialisation ----------------------------------------------------------

class ConcurrencyProbe(ScriptedModel):
    """Fails loudly if two turns are ever inside the graph at once."""

    def __init__(self, *script):
        super().__init__(*script)
        self.inside = 0
        self.overlapped = False

    async def ainvoke(self, messages, **kw):
        self.inside += 1
        self.overlapped = self.overlapped or self.inside > 1
        try:
            await asyncio.sleep(0.01)
            return await super().ainvoke(messages, **kw)
        finally:
            self.inside -= 1


async def test_two_turns_arriving_together_do_not_interleave(runner, scripted, pipeline):
    """Both would read "what is running" and race to replace it."""
    model = scripted(ConcurrencyProbe(*says("Done.").script))

    replies = await asyncio.gather(
        runner.user_turn("highlight everyone"),
        runner.user_turn("count the people"),
    )

    assert not model.overlapped
    assert [r.reply for r in replies] == ["Done.", "Done."]
    # Grounding reads three endpoints in a fixed order. Two clean runs of it,
    # rather than six calls shuffled together, is the lock doing its job.
    assert pipeline.paths("GET") == ["/behaviors", "/health", "/models"] * 2


# 5. Failure ----------------------------------------------------------------

async def test_a_turn_that_blows_up_still_answers_the_operator(runner, monkeypatch):
    async def explode(*_a, **_kw):
        raise RuntimeError("the graph fell over")

    monkeypatch.setattr(runner.graph, "ainvoke", explode)

    result = await runner.user_turn("track the duck")

    assert "That failed" in result.reply
    assert "the graph fell over" in result.reply


async def test_a_turn_that_blows_up_is_traced_as_an_error(runner, monkeypatch, trace):
    async def explode(*_a, **_kw):
        raise RuntimeError("the graph fell over")

    monkeypatch.setattr(runner.graph, "ainvoke", explode)

    await runner.user_turn("track the duck")

    assert any(e.kind == "error" and e.label == "turn failed" for e in trace)
