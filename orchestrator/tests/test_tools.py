"""The agent's tools.

Two properties are worth more than the rest put together:

1. **A tool answers in prose, and never raises.** A tool that throws ends the
   turn. Every failure here has to come back as a sentence the model can act
   on — which means the sentence has to carry perception's own reason, not a
   generic "call failed".
2. **The argument schemas are the shared contracts.** `start_behavior` takes
   a `BehaviorSpec`, so the model is shown the same field descriptions the
   pipeline validates against. If that ever became a hand-written copy it
   would drift, and the drift would only surface as 422s on stage.
"""

from __future__ import annotations

import json

import pytest

from contracts import Selector
from tools.registry import build_tools


@pytest.fixture
def tools(perception, memory):
    return {t.name: t for t in build_tools(perception, memory, turn="turn-1")}


HIGHLIGHT = {
    "kind": "highlight",
    "subject": {"detect": ["person"]},
    "render": {"label": "everyone"},
}


# 1. Shape ------------------------------------------------------------------

def test_every_capability_the_agent_needs_has_a_tool(tools):
    assert set(tools) == {
        "start_behavior", "stop_behavior", "clear_behaviors", "list_behaviors",
        "probe_phrases", "count_objects", "look", "describe_scene",
        "add_reference", "set_model", "set_hud", "say",
    }


def test_start_behavior_is_typed_by_the_shared_contract(tools):
    """Not a copy of the schema — the schema."""
    schema = tools["start_behavior"].args_schema.model_json_schema()

    assert set(schema["properties"]) >= {"kind", "subject", "params", "render", "notify"}
    selector = schema["$defs"]["Selector"]
    assert selector["properties"]["detect"]["maxItems"] == 8
    assert selector["properties"]["min_score"]["default"] == 0.75
    assert selector["additionalProperties"] is False


def test_the_selectors_worked_examples_reach_the_model(tools):
    """Pydantic carries a class docstring into the schema as `description`.

    That is the whole of `Selector`'s documentation — the five worked examples
    and the sentence about include/exclude — arriving in the tool definition
    the model is given. Field-level `#:` comments do *not* travel; anything
    the agent must know about a single field belongs in SELECTORS.md, which
    is in the system prompt.
    """
    schema = tools["start_behavior"].args_schema.model_json_schema()
    described = schema["$defs"]["Selector"]["description"]

    assert '{"detect": ["person"], "ref_id": "r1", "pick": "ref"}' in described
    assert "are CLIP checks on the subject" in described


def test_the_kinds_offered_are_the_ones_the_contract_declares(tools):
    schema = tools["start_behavior"].args_schema.model_json_schema()
    kinds = schema["properties"]["kind"]["enum"]

    assert set(kinds) == {"highlight", "track", "watch", "count_line",
                          "privacy", "pan_to", "pose_trigger", "keyboard"}


# 2. Installing -------------------------------------------------------------

async def test_starting_a_behavior_reports_the_id_back_to_the_model(tools, pipeline):
    answer = await tools["start_behavior"].ainvoke(HIGHLIGHT)

    assert "b1" in answer
    assert pipeline.behaviors["b1"]["kind"] == "highlight"


async def test_starting_a_behavior_traces_the_call_and_its_result(tools, trace):
    await tools["start_behavior"].ainvoke(HIGHLIGHT)

    kinds = [(e.kind, e.label) for e in trace]
    assert ("tool_call", "start_behavior") in kinds
    assert ("tool_result", "start_behavior") in kinds
    result = next(e for e in trace if e.kind == "tool_result")
    assert result.data["behavior_id"] == "b1"
    assert result.turn == "turn-1"


async def test_the_traced_call_carries_the_spec_that_was_sent(tools, trace):
    await tools["start_behavior"].ainvoke(HIGHLIGHT)

    call = next(e for e in trace if e.kind == "tool_call")
    assert call.data["spec"]["subject"]["detect"] == ["person"]


async def test_a_refusal_comes_back_as_the_reason_and_an_instruction_to_retry(tools):
    answer = await tools["start_behavior"].ainvoke(
        {"kind": "watch", "subject": {"detect": ["duck"]}})

    assert answer.startswith("REFUSED:")
    assert "triggers" in answer
    assert "try again" in answer


async def test_an_unbuilt_kind_is_refused_with_the_list_of_built_ones(tools):
    answer = await tools["start_behavior"].ainvoke(
        {"kind": "keyboard", "subject": {"detect": ["keyboard"]}})

    assert "not implemented yet" in answer
    assert "highlight" in answer


async def test_a_watch_with_a_near_trigger_installs(tools, pipeline):
    """The kind the old adapter could not reach at all."""
    answer = await tools["start_behavior"].ainvoke({
        "kind": "watch",
        "subject": {"detect": ["rubber duck"], "pick": "largest"},
        "params": {"triggers": [
            {"type": "near",
             "other": {"detect": ["person"]},
             "margin": 0.1},
        ]},
    })

    assert "b1" in answer
    assert pipeline.behaviors["b1"]["state"] == "ARMING"


async def test_installing_files_the_wording_for_next_time(tools, memory):
    await tools["start_behavior"].ainvoke({
        "kind": "track",
        "subject": {"detect": ["a yellow rubber duck"], "pick": "largest"},
        "render": {"label": "the duck"},
    })

    remembered = memory.suggest("the duck")
    assert any(r.phrase == "a yellow rubber duck" and r.outcome == "acquired"
               for r in remembered)


async def test_stopping_one_behavior_leaves_the_others(tools, pipeline):
    await tools["start_behavior"].ainvoke(HIGHLIGHT)
    await tools["start_behavior"].ainvoke(HIGHLIGHT)

    answer = await tools["stop_behavior"].ainvoke({"behavior_id": "b1"})

    assert "Stopped b1" in answer
    assert list(pipeline.behaviors) == ["b2"]


async def test_stopping_something_that_is_not_there_explains_rather_than_raises(tools):
    answer = await tools["stop_behavior"].ainvoke({"behavior_id": "b42"})

    assert "Could not stop b42" in answer
    assert "b42" in answer


async def test_clearing_empties_the_pipeline(tools, pipeline):
    await tools["start_behavior"].ainvoke(HIGHLIGHT)

    answer = await tools["clear_behaviors"].ainvoke({})

    assert "idle" in answer
    assert pipeline.behaviors == {}


async def test_listing_returns_json_the_model_can_read(tools):
    await tools["start_behavior"].ainvoke(HIGHLIGHT)

    answer = await tools["list_behaviors"].ainvoke({})

    assert json.loads(answer)[0]["id"] == "b1"


# 3. Asking the pipeline ----------------------------------------------------

async def test_a_probe_reports_every_score_including_the_zeros(tools, pipeline):
    pipeline.probe_scores = {"a yellow rubber duck": 0.22}

    answer = await tools["probe_phrases"].ainvoke({
        "subject": "the duck",
        "phrases": ["duck", "a yellow rubber duck", "toy"],
    })

    assert '"duck" 0.00 (nothing)' in answer
    assert '"a yellow rubber duck" 0.22' in answer
    assert 'Use "a yellow rubber duck".' in answer


async def test_a_probe_is_written_to_memory_zeros_and_all(tools, pipeline, memory):
    """The zeros are the valuable half: they stop the next run repeating them."""
    pipeline.probe_scores = {"a yellow rubber duck": 0.22}

    await tools["probe_phrases"].ainvoke({
        "subject": "the duck", "phrases": ["duck", "a yellow rubber duck"]})

    phrases = {r.phrase: r for r in memory.suggest("the duck", limit=20)}
    assert phrases["duck"].found == 0
    assert phrases["a yellow rubber duck"].max_conf == pytest.approx(0.22)


async def test_a_failed_probe_tells_the_model_to_carry_on(tools, pipeline):
    pipeline.down = True

    answer = await tools["probe_phrases"].ainvoke({
        "subject": "the duck", "phrases": ["duck"]})

    assert "Probe failed" in answer
    assert "best guess" in answer


async def test_counting_answers_with_the_window_it_used(tools, pipeline):
    pipeline.count_answer = 3

    answer = await tools["count_objects"].ainvoke({
        "selector": {"detect": ["person"]}, "window_s": 2.0})

    assert answer.startswith("3, as a median over 2s")
    assert "14 samples" in answer


async def test_looking_at_nothing_says_so_plainly(tools):
    assert "Nothing is being tracked" in await tools["look"].ainvoke({})


async def test_looking_returns_the_tracks(tools, pipeline):
    pipeline.tracks = [{"track_id": 7, "label": "person", "conf": 0.9}]

    answer = await tools["look"].ainvoke({"selector": {"detect": ["person"]}})

    assert json.loads(answer)[0]["track_id"] == 7


async def test_describe_hands_over_the_grounding_and_forbids_contradicting_it(tools, pipeline):
    pipeline.scene_objects = [
        {"label": "laptop", "where": "centre", "size": "large", "conf": 0.8},
    ]

    answer = await tools["describe_scene"].ainvoke({"question": "what's on the table?"})

    assert "laptop" in answer
    assert "do not contradict it" in answer


# 4. References, model, HUD, speech ----------------------------------------

async def test_a_reference_comes_back_with_how_to_use_it(tools):
    answer = await tools["add_reference"].ainvoke({"source": "largest_person"})

    assert "r1" in answer
    assert 'pick "ref"' in answer


async def test_switching_model_warns_not_to_restart_paused_behaviours(tools, pipeline):
    answer = await tools["set_model"].ainvoke({"name": "coco"})

    assert "Switched to coco" in answer
    assert "do not restart them" in answer
    assert pipeline.health_payload["model"] == "coco"


async def test_an_unknown_model_is_refused_with_the_names_that_exist(tools):
    answer = await tools["set_model"].ainvoke({"name": "yolov99"})

    assert "Could not switch model" in answer
    assert "yoloe" in answer


async def test_the_hud_is_set_and_echoed(tools, pipeline):
    answer = await tools["set_hud"].ainvoke({"text": "WATCHING THE DUCK"})

    assert "WATCHING THE DUCK" in answer
    assert pipeline.hud == "WATCHING THE DUCK"


async def test_say_publishes_to_the_trace_rather_than_the_pipeline(tools, trace, pipeline):
    await tools["say"].ainvoke({"text": "Someone is reaching for the duck."})

    spoken = [e for e in trace if e.kind == "say"]
    assert spoken[0].label == "Someone is reaching for the duck."
    # Nothing was sent downstream: speech is the frontend's job.
    assert pipeline.calls == []


# 5. The rule that holds for all of them ------------------------------------

@pytest.mark.parametrize("name,args", [
    ("start_behavior", HIGHLIGHT),
    ("stop_behavior", {"behavior_id": "b1"}),
    ("clear_behaviors", {}),
    ("list_behaviors", {}),
    ("probe_phrases", {"subject": "the duck", "phrases": ["duck"]}),
    ("count_objects", {"selector": {"detect": ["person"]}}),
    ("look", {}),
    ("describe_scene", {"question": "what is there?"}),
    ("add_reference", {}),
    ("set_model", {"name": "coco"}),
    ("set_hud", {"text": "x"}),
])
async def test_no_tool_raises_when_the_pipeline_is_down(tools, pipeline, name, args):
    pipeline.down = True

    answer = await tools[name].ainvoke(args)

    assert isinstance(answer, str) and answer
    assert "cannot reach perception" in answer.lower() or "could not" in answer.lower()


def test_a_selector_summarises_itself_for_the_trace():
    """What the operator sees in the trace instead of JSON."""
    selector = Selector(detect=["person"], include=["a red jacket"], ref_id="r1")

    assert selector.summary() == "person ✓a red jacket ref:r1"
