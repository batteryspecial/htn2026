"""What the model is told about the device, every turn.

`situation()` is pure — no pipeline, no LLM, no memory — so these run anywhere
`agent.prompts` imports.
"""

from __future__ import annotations

from agent.prompts import situation


def test_the_gesture_vocabulary_is_named():
    """Knowing only that `pose_trigger` exists is what broke the finger guard.

    The compiler still had to fill in a `gesture`, and `hand_raised` was the
    only implemented value, so "report when someone raises their index finger"
    compiled to it. The spec validated, the behaviour armed, and it watched
    wrists forever. Naming the vocabulary is what lets the agent refuse.
    """
    text = situation([], None, True, ["hand_raised"])
    assert "hand_raised" in text
    assert "no others" in text, "the list has to read as exhaustive, not as examples"


def test_an_empty_vocabulary_is_not_a_gesture():
    text = situation([], None, True, [])
    assert "cannot fire" in text
    assert "hand_raised" not in text


def test_no_answer_from_the_pipeline_claims_nothing():
    """`None` is "we did not get told", which must not render as "there are
    none" — that would have the agent refuse a gesture it can actually do."""
    text = situation([], None, True, None)
    assert "gesture" not in text.lower()


def test_gestures_are_optional_for_existing_callers():
    """The argument is additive: three-argument calls still work."""
    assert situation([], None, True)
