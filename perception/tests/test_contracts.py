"""Schema guardrails.

A malformed instruction must be refused at the door, not halfway down the
pipeline. These also pin the shape the orchestrator is written against.
"""

import pytest
from pydantic import ValidationError

from contracts import BehaviorSpec, Event, RenderSpec, Selector


def sel(**over):
    return Selector(**{"detect": ["person"], **over})


# 1. Selector -------------------------------------------------------------
def test_defaults_are_the_simple_case():
    s = sel()
    assert s.pick == "all" and s.include == [] and s.exclude == []
    assert not s.needs_attributes and not s.needs_reference


def test_relate_adds_its_class_to_the_detector_prompts():
    """"the person in red shoes" needs the shoe detected too."""
    s = sel(relate={"contains": "shoe"})
    assert s.prompts() == ["person", "shoe"]


def test_prompts_are_deduplicated():
    assert sel(detect=["person", "person"], relate={"contains": "person"}).prompts() == ["person"]


def test_attribute_texts_cover_include_and_exclude():
    s = sel(include=["a person in red"], exclude=["a person in black"])
    assert s.attribute_texts() == ["a person in red", "a person in black"]
    assert s.needs_attributes


def test_a_reference_is_recognised_either_way():
    assert sel(ref_id="r1").needs_reference
    assert sel(pick="ref").needs_reference


def test_summary_is_readable():
    s = sel(include=["a yellow duck"], exclude=["a black jacket"], relate={"contains": "shoe"})
    text = s.summary()
    assert "person" in text and "yellow duck" in text and "shoe" in text


@pytest.mark.parametrize("over", [
    {"detect": []},
    {"detect": ["a"] * 9},
    {"pick": "vibes"},
    {"include": ["a"] * 5},
    {"min_score": 1.5},
    {"ref_min_sim": -0.1},
    {"relate": {"contains": "shoe", "lower_frac": 0}},
    {"nonsense": 1},
])
def test_a_bad_selector_is_rejected(over):
    with pytest.raises(ValidationError):
        sel(**over)


# 2. BehaviorSpec ---------------------------------------------------------
def test_a_behavior_carries_no_id():
    """The server assigns ids, so two agents cannot collide on a name."""
    assert "id" not in BehaviorSpec(kind="highlight", subject={"detect": ["a"]}).model_dump()


def test_render_defaults_to_something_visible():
    r = BehaviorSpec(kind="highlight", subject={"detect": ["a"]}).render
    assert r.boxes and r.mask and not r.trail and r.color is None


def test_notify_defaults_on():
    assert BehaviorSpec(kind="highlight", subject={"detect": ["a"]}).notify is True


@pytest.mark.parametrize("over", [
    {"kind": "teleport"},
    {"subject": {"detect": []}},
    {"subject": {}},
    {"render": {"color": "#FFF", "bogus": 1}},
    {"junk": 1},
])
def test_a_bad_behavior_is_rejected(over):
    with pytest.raises(ValidationError):
        BehaviorSpec(**{"kind": "highlight", "subject": {"detect": ["a"]}, **over})


def test_params_are_free_form():
    """Kind-specific settings are validated per kind, so adding a behaviour
    never means editing the shared contract."""
    b = BehaviorSpec(kind="count_line", subject={"detect": ["person"]},
                     params={"line": [[0, 0.5], [1, 0.5]]})
    assert b.params["line"][0] == [0, 0.5]


# 3. Event ----------------------------------------------------------------
def test_a_system_event_has_no_behavior():
    assert Event(id="e1", type="camera_lost").behavior_id is None


def test_an_event_serialises_for_the_wire():
    payload = Event(id="e1", type="near", detail="someone near the duck",
                    behavior_id="b3", snapshot_url="/snapshots/e1.jpg").model_dump(mode="json")
    assert payload["type"] == "near" and payload["snapshot_url"].endswith(".jpg")
    assert payload["notify"] is True


def test_an_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        Event(id="e1", type="exploded")


def test_render_colour_is_free_text_but_parsed_safely():
    """A typo in a colour must not reach the drawing code as a crash."""
    from render.layers import hex_to_bgr

    assert hex_to_bgr(RenderSpec(color="#00FF00").color) == (0, 255, 0)
    assert hex_to_bgr(RenderSpec(color="chartreuse").color) is not None
