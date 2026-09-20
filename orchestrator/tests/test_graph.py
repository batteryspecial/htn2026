"""The agent loop: ground → think → act ⇄ think → finish.

What makes this better than a one-shot compiler is `ground`: before the model
sees the instruction it is handed what is actually running, what the detector
is, and what wordings have been measured for a subject like this one. Then it
can probe, read the answer, and only then commit. Most of this file checks
that the model really is told those things, and that the loop survives every
way a step can fail.

The model is scripted (`fake_model.py`). No provider is reachable from a test
and none should be.
"""

from __future__ import annotations

import pytest

from agent.graph import build_graph, opening_messages
from config import CFG
from tests.fake_model import calls, says
from tools.registry import build_tools


@pytest.fixture
def run(perception, memory, monkeypatch):
    """Drive one turn with a scripted model, and hand back the final state."""
    graph = build_graph(perception, memory)

    async def drive(model, instruction="track the duck", origin="user",
                    images=None):
        monkeypatch.setattr("agent.graph.chat_model", lambda: model)
        return await graph.ainvoke({
            "messages": opening_messages(instruction, origin, images or []),
            "turn": "turn-1",
            "origin": origin,
            "instruction": instruction,
            "image_urls": images or [],
            "steps": 0,
            "started": 0.0,
        }, {"recursion_limit": CFG.max_steps * 2 + 8})

    return drive


HIGHLIGHT = {"kind": "highlight", "subject": {"detect": ["person"]}}


# 1. Grounding --------------------------------------------------------------

async def test_the_model_is_told_what_is_already_running(run, perception, pipeline):
    await perception.start_behavior(
        {"kind": "track", "subject": {"detect": ["duck"]},
         "render": {"label": "the duck"}})
    model = says("Already on it.")

    await run(model, "what are you doing?")

    told = model.prompt_text()
    assert 'b1: track "the duck" [ACTIVE]' in told


async def test_a_behaviour_matching_nothing_is_called_out_as_such(run, pipeline):
    """ACTIVE with zero matches is the failure that looks like success."""
    pipeline.behaviors["b1"] = {
        "id": "b1", "kind": "highlight", "state": "ACTIVE",
        "spec": "duck", "label": "the duck", "matches": 0,
        "track_ids": [], "since": 0.0, "detail": None, "notify": True,
    }
    model = says("Nothing is matching.")

    await run(model, "is it working?")

    assert "matching nothing, the wording may be wrong" in model.prompt_text()


async def test_the_model_is_told_which_detector_is_live(run):
    model = says("ok")

    await run(model)

    assert "Detector: yoloe" in model.prompt_text()
    assert "28 fps" in model.prompt_text()


async def test_a_fixed_vocabulary_detector_is_flagged_before_anything_is_tried(run, pipeline):
    """With coco active, a descriptive phrase in `detect` is a guaranteed 422."""
    for entry in pipeline.models_payload["models"]:
        entry["active"] = entry["name"] == "coco"
    pipeline.health_payload["model"] = "coco"
    model = says("ok")

    await run(model)

    assert "FIXED VOCABULARY" in model.prompt_text()


async def test_the_measured_prior_is_handed_over_before_the_model_thinks(run):
    model = says("ok")

    await run(model, "track the yellow rubber duck")

    told = model.prompt_text()
    assert "Measured before" in told
    assert '"yellow duck"' in told


async def test_a_prior_is_traced_so_the_operator_sees_the_retrieval(run, trace):
    await run(says("ok"), "track the yellow rubber duck")

    thoughts = [e for e in trace if e.kind == "thought"]
    assert any(e.label == "recalled phrasings" for e in thoughts)


async def test_a_dead_pipeline_is_reported_to_the_model_not_hidden(run, pipeline):
    pipeline.down = True
    model = says("The camera service is not answering.")

    state = await run(model)

    told = model.prompt_text()
    assert "WARNING: the pipeline is not answering" in told
    assert "do not pretend a behaviour started" in told
    assert state["reply"] == "The camera service is not answering."


# 2. The loop ---------------------------------------------------------------

async def test_an_answer_with_no_tool_calls_finishes_in_one_step(run, pipeline):
    state = await run(says("There are three people."), "how many people?")

    assert state["reply"] == "There are three people."
    assert state["steps"] == 1
    # Grounding reads; a turn that installs nothing writes nothing.
    assert pipeline.paths("POST") == []


async def test_a_tool_call_runs_and_the_loop_comes_back_for_another_thought(run, pipeline):
    model = calls(("start_behavior", HIGHLIGHT))
    model = model.then(says("Highlighting everyone."))

    state = await run(model)

    assert state["reply"] == "Highlighting everyone."
    assert pipeline.behaviors["b1"]["kind"] == "highlight"
    assert state["steps"] == 2


async def test_several_tool_calls_in_one_step_run_in_order(run, pipeline):
    """Stop the old one, then start the new one — that pair must not race."""
    model = calls(
        ("clear_behaviors", {}),
        ("start_behavior", {"kind": "track", "subject": {"detect": ["dog"]}}),
    )
    model = model.then(says("Now following the dog."))

    await run(model, "follow the dog instead")

    assert pipeline.paths()[-2:] == ["/behaviors", "/behaviors"]
    assert [c["method"] for c in pipeline.calls[-2:]] == ["DELETE", "POST"]


async def test_the_model_reads_a_refusal_and_fixes_its_own_call(run, pipeline):
    """The behaviour that makes specific 422s worth writing."""
    model = calls(("start_behavior", {
        "kind": "watch", "subject": {"detect": ["rubber duck"]}}))
    model = model.then(calls(("start_behavior", {
        "kind": "watch",
        "subject": {"detect": ["rubber duck"]},
        "params": {"triggers": [{"type": "missing", "after_s": 2.0}]},
    }))).then(says("Watching the duck."))

    state = await run(model, "tell me if the duck disappears")

    # The refusal reached the model verbatim, and the retry landed.
    assert "watch needs a non-empty 'triggers' list" in model.prompt_text(1)
    assert state["reply"] == "Watching the duck."
    assert list(pipeline.behaviors) == ["b1"]


async def test_the_loop_stops_at_the_step_cap_rather_than_spinning(run, trace):
    """A demo that thinks for twenty steps has already lost."""
    model = calls(("list_behaviors", {}))  # asks forever

    state = await run(model)

    assert state["steps"] == CFG.max_steps
    assert any(e.kind == "error" and e.label == "step limit" for e in trace)


async def test_whatever_was_installed_survives_the_step_cap(run, pipeline):
    model = calls(("start_behavior", HIGHLIGHT))
    model = model.then(calls(("list_behaviors", {})))

    await run(model)

    assert list(pipeline.behaviors) == ["b1"]


# 3. When a step fails ------------------------------------------------------

async def test_a_model_that_errors_ends_the_turn_with_an_explanation(run):
    from tests.fake_model import ScriptedModel

    state = await run(ScriptedModel(error=RuntimeError("503 upstream")))

    assert "the model call failed" in state["reply"]
    assert "503 upstream" in state["reply"]


async def test_a_model_error_is_traced_as_an_error(run, trace):
    from tests.fake_model import ScriptedModel

    await run(ScriptedModel(error=RuntimeError("503 upstream")))

    assert any(e.kind == "error" and e.label == "model" for e in trace)


async def test_no_api_key_is_said_plainly_instead_of_crashing_the_service(perception, memory):
    """`conftest` blanks the key, so this is the real `chat_model` path."""
    graph = build_graph(perception, memory)

    state = await graph.ainvoke({
        "messages": opening_messages("track the duck", "user"),
        "turn": "turn-1", "origin": "user", "instruction": "track the duck",
        "image_urls": [], "steps": 0, "started": 0.0,
    })

    assert "no API key" in state["reply"]
    assert ".env" in state["reply"]


async def test_a_tool_that_does_not_exist_is_reported_to_the_model(run):
    model = calls(("summon_a_pony", {}))
    model = model.then(says("I cannot do that."))

    await run(model)

    assert "No such tool: summon_a_pony." in model.prompt_text(1)


async def test_a_tool_that_raises_does_not_end_the_turn(run, monkeypatch):
    """A raising tool would kill the turn; the model gets the reason instead."""
    async def exploding(*_a, **_kw):
        raise ValueError("the selector was nonsense")

    original = build_tools

    def patched(perception, memory, turn=""):
        tools = original(perception, memory, turn)
        for tool in tools:
            if tool.name == "look":
                tool.coroutine = exploding
        return tools

    monkeypatch.setattr("agent.graph.build_tools", patched)

    model = calls(("look", {}))
    model = model.then(says("Something went wrong looking."))

    state = await run(model)

    assert "the selector was nonsense" in model.prompt_text(1)
    assert "Check the arguments and try again" in model.prompt_text(1)
    assert state["reply"] == "Something went wrong looking."


async def test_a_turn_that_says_nothing_still_answers_something(run):
    state = await run(says(""))

    assert state["reply"] == "Done."


# 4. Finishing --------------------------------------------------------------

async def test_the_reply_and_the_elapsed_time_are_published(run, trace):
    await run(says("Tracking the duck."))

    reply = next(e for e in trace if e.kind == "reply")
    timer = next(e for e in trace if e.kind == "timer")
    assert reply.label == "Tracking the duck."
    assert reply.turn == "turn-1"
    assert "seconds" in timer.data


async def test_the_models_own_words_are_traced_when_it_also_calls_a_tool(run, trace):
    model = calls(("list_behaviors", {}), text="Let me check what is running.")
    model = model.then(says("Nothing is running."))

    await run(model)

    thoughts = [e.label for e in trace if e.kind == "thought"]
    assert "Let me check what is running." in thoughts


# 5. The opening messages ---------------------------------------------------

def test_the_system_prompt_comes_first_and_carries_the_documents():
    messages = opening_messages("track the duck", "user")

    system = messages[0].content
    assert "You are the agent layer of a live camera system" in system
    # SELECTORS.md and BEHAVIORS.md are pulled in whole.
    assert "detect" in system and "watch" in system
    assert messages[1].content == "track the duck"


def test_an_event_turn_is_told_to_look_before_it_speaks():
    messages = opening_messages("something moved", "event")

    assert "You were woken by the pipeline" in messages[1].content


def test_an_uploaded_image_rides_along_with_the_instruction():
    messages = opening_messages("track this one", "user",
                                ["data:image/jpeg;base64,abc"])

    parts = messages[-1].content
    assert parts[0] == {"type": "text", "text": "track this one"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg")
